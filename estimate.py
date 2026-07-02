"""Estimate the label-free steering direction d and cache it to results/direction_<size>.pt.
d is config-invariant, so sample.py and benchmark.py reuse it.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer
from steering import build_elf, sampling_args, estimate_direction

HERE = Path(__file__).resolve().parent
DEFAULT_CKPT = HERE / "models" / "ELF-B-owt-torch"
OUT = Path(os.environ.get("ELF_OUT", HERE / "results"))
OUT.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT),
                    help="path to a model dir (or file); EMA weights loaded; default models/ELF-B-owt-torch")
    ap.add_argument("--size", default="B")
    ap.add_argument("--n-est", type=int, default=400, help="samples to estimate d")
    ap.add_argument("--seed", type=int, default=42, help="fixes the sampling noise")
    ap.add_argument("--out", default=None, help="default results/direction_<size>.pt")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained("t5-small")
    model = build_elf(a.size, a.ckpt, dev)
    args = sampling_args()

    d = estimate_direction(model, n=a.n_est, tokenizer=tok, args=args, device=str(dev), seed=a.seed)

    dest = Path(a.out) if a.out else (OUT / f"direction_{a.size}.pt")
    torch.save({"d": d.cpu(), "size": a.size, "n_est": a.n_est}, dest)
    print(f"saved {dest}  (size={a.size}, n_est={a.n_est}, dim={d.numel()})")


if __name__ == "__main__":
    main()
