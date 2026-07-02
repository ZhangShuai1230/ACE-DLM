"""Compute-to-clean benchmark: per method (baseline vs steered), sample until 1000
generations pass seq-rep-4 <= 1.92%, then report the compute spent and the clean-PPL.
"""
from __future__ import annotations
import argparse, json
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer
from tqdm.auto import tqdm 
from steering import build_elf, sampling_args, generate, load_direction
from metrics import seq_rep_4, gen_ppl

HERE = Path(__file__).resolve().parent
DEFAULT_CKPT = HERE / "models" / "ELF-B-owt-torch"
OUT = Path(os.environ.get("ELF_OUT", HERE / "results"))
OUT.mkdir(parents=True, exist_ok=True)

CRIT = 0.0192   # human 95th-percentile seq-rep-4


def collect_clean(model, *, steer_d, lam, tokenizer, args, target, cap, device, bs=50, seed=42):
    accepted, total = [], 0
    gen = torch.Generator(device=device).manual_seed(seed)
    pbar = tqdm(total=target, desc="clean (reject-to-N)")
    while len(accepted) < target and total < cap:
        texts = generate(model, n=bs, steer_d=steer_d, lam=lam, tokenizer=tokenizer,
                         args=args, device=device, bs=bs, generator=gen)
        total += len(texts)
        before = len(accepted)
        accepted.extend([t for t in texts if seq_rep_4(t) <= CRIT])
        pbar.update(min(len(accepted), target) - min(before, target))
    pbar.close()
    return accepted[:target], total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT),
                    help="path to a model dir (or file); EMA weights loaded; default models/ELF-B-owt-torch")
    ap.add_argument("--steer-d", default=None,
                    help="cached d from estimate.py (default results/direction_<size>.pt)")
    ap.add_argument("--size", default="B")
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--cap", type=int, default=10000, help="max generations before giving up")
    ap.add_argument("--seed", type=int, default=42,
                    help="fixes the sampling noise")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained("t5-small")
    model = build_elf(a.size, a.ckpt, dev)
    args = sampling_args()
    steer_d = a.steer_d or str(OUT / f"direction_{a.size}.pt")
    d = load_direction(steer_d, str(dev))
    print(f"loaded steering direction from {steer_d}")

    out = {"ckpt": a.ckpt, "criterion_seq_rep_4": CRIT, "target": a.target, "lam": a.lam}
    for tag, lam, sd in [("baseline", 0.0, None), (f"steered(lam={a.lam})", a.lam, d)]:
        clean, total = collect_clean(model, steer_d=sd, lam=lam, tokenizer=tok, args=args,
                                     target=a.target, cap=a.cap, device=str(dev), seed=a.seed)
        ppl = gen_ppl(clean, dev)[0] if len(clean) >= 20 else None
        out[tag] = {"generations": total, "clean": len(clean),
                    "accept_rate": round(len(clean) / max(total, 1), 4), "clean_ppl": ppl}
        print(f"=== {tag} ===\n{json.dumps(out[tag], indent=2)}")
    dest = a.out or (OUT / f"benchmark_{a.size}.json")
    json.dump(out, open(dest, "w"), indent=2); print(f"saved {dest}")


if __name__ == "__main__":
    main()
