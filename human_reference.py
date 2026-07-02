"""Derive the 1.92% acceptance bar (human 95th-percentile seq-rep-4) from real XSum
articles via metrics.seq_rep_4.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from metrics import seq_rep_4, ACCEPT_THRESHOLD

OUT = Path(os.environ.get("ELF_OUT", Path(__file__).resolve().parent / "results"))
OUT.mkdir(parents=True, exist_ok=True)


def load_human(n, min_words=50, split="validation"):
    """Yield up to ``n`` real XSum source articles (``document``) of >= ``min_words`` words."""
    ds = load_dataset("EdinburghNLP/xsum", split=split, streaming=True, trust_remote_code=True)
    out = []
    for row in ds:
        text = row["document"]
        if text and len(text.split()) >= min_words:
            out.append(text)
        if len(out) >= n:
            break
    return out


def percentile(sorted_vals, p):
    """Linear-interpolated percentile (p in [0, 100]); matches numpy's default."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (p / 100.0) * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="number of human documents")
    ap.add_argument("--min-words", type=int, default=50, help="minimum words per document")
    a = ap.parse_args()

    human = load_human(a.n, a.min_words)
    reps = sorted(seq_rep_4(t) for t in human)

    median = percentile(reps, 50)
    p95 = percentile(reps, 95)
    frac_above_5pct = sum(r > 0.05 for r in reps) / len(reps)

    print(f"human documents       : {len(reps)}")
    print(f"seq-rep-4 median      : {100 * median:.3f}%")
    print(f"seq-rep-4 95th pct    : {100 * p95:.3f}%  (acceptance bar)")
    print(f"fraction above 5%     : {100 * frac_above_5pct:.2f}%")
    print(f"metrics.ACCEPT_THRESHOLD = {100 * ACCEPT_THRESHOLD:.2f}%")
    print(f"derived bar matches ACCEPT_THRESHOLD: {abs(p95 - ACCEPT_THRESHOLD) < 5e-4}")

    res = {"n_documents": len(reps),
           "rep_median_pct": round(100 * median, 3),
           "rep_p95_pct": round(100 * p95, 3),
           "frac_above_5pct": round(frac_above_5pct, 4),
           "accept_threshold_pct": round(100 * ACCEPT_THRESHOLD, 3),
           "matches_accept_threshold": bool(abs(p95 - ACCEPT_THRESHOLD) < 5e-4)}
    (OUT / "human_reference.json").write_text(json.dumps(res, indent=2))
    print(f"saved {OUT / 'human_reference.json'}")


if __name__ == "__main__":
    main()
