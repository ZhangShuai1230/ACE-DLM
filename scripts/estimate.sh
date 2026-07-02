#!/usr/bin/env bash
# Estimate the steering direction d once on ELF-B -> results/direction_B.pt.

set -euo pipefail
export CUDA_VISIBLE_DEVICES=5

HERE="$(dirname "$0")/.."
echo "[estimate] B: difference-of-means steering direction d (reused across all sizes)"
python "$HERE/estimate.py" --size B --ckpt "$HERE/models/ELF-B-owt-torch" "$@"