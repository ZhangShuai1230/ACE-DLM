"""Generate unconditional ELF samples under plain vs steered self-conditioning and report
the defect metrics (rep, Gen-PPL) for each.
"""
from __future__ import annotations
import argparse, json
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer
from steering import build_elf, sampling_args, estimate_direction, generate, load_direction
from metrics import summarize, gen_ppl

HERE = Path(__file__).resolve().parent
DEFAULT_CKPT = HERE / "models" / "ELF-B-owt-torch"
OUT = Path(os.environ.get("ELF_OUT", HERE / "results"))
OUT.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT),
                    help="path to a model dir (or file); EMA weights loaded; default models/ELF-B-owt-torch")
    ap.add_argument("--size", default="B")
    ap.add_argument("--n", type=int, default=500, help="samples to evaluate")
    ap.add_argument("--steer-d", default=None, help="cached d from estimate.py (default results/direction_<size>.pt)")
    ap.add_argument("--n-est", type=int, default=400, help="samples to estimate d (fallback only)")
    ap.add_argument("--lam", type=float, default=2.0, help="steering strength")
    ap.add_argument("--seed", type=int, default=42,help="fixes the sampling noise")
    ap.add_argument("--no-ppl", action="store_true", help="skip Gen-PPL (faster)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained("t5-small")
    model = build_elf(a.size, a.ckpt, dev)
    args = sampling_args()

    steer_d = a.steer_d or str(OUT / f"direction_{a.size}.pt")
    if Path(steer_d).exists():
        d = load_direction(steer_d, str(dev))
        print(f"loaded steering direction from {steer_d}")
    else:
        d = estimate_direction(model, n=a.n_est, tokenizer=tok, args=args, device=str(dev), seed=a.seed)
        print(f"{steer_d} not found -- estimated d inline (run scripts/estimate.sh to cache it)")

    out = {"ckpt": a.ckpt, "n": a.n, "lam": a.lam}
    for tag, lam in [("baseline", 0.0), (f"steered(lam={a.lam})", a.lam)]:
        texts = generate(model, n=a.n, steer_d=(d if lam else None), lam=lam,
                         tokenizer=tok, args=args, device=str(dev), seed=a.seed)
        s = summarize(texts)
        if not a.no_ppl:
            s["gen_ppl"], s["entropy"] = gen_ppl(texts, dev)
        out[tag] = s
        print(f"\n=== {tag} ===")
        print(json.dumps(s, indent=2))

    dest = a.out or (OUT / f"sample_{a.size}.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {dest}")


if __name__ == "__main__":
    main()
