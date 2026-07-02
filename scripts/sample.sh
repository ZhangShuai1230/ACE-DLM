#!/usr/bin/env bash
# Baseline vs steered generation for B, M, L -> results/sample_<size>.json.
# All three sizes steer with the SAME d estimated on ELF-B (results/direction_B.pt).

set -euo pipefail
export CUDA_VISIBLE_DEVICES=7

HERE="$(dirname "$0")/.."
DB="$HERE/results/direction_B.pt"
for SC in B:ELF-B-owt-torch M:ELF-M-owt-torch L:ELF-L-owt-torch; do
  S="${SC%%:*}"; CK="${SC##*:}"
  echo "[sample] $S: baseline vs steered with d_B"
  python "$HERE/sample.py" --size "$S" --ckpt "$HERE/models/$CK" --n 1000 --lam 2.0 --steer-d "$DB" "$@"
done
