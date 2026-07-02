#!/usr/bin/env bash
# Compute-to-clean benchmark for B, M, L -> results/benchmark_<size>.json.
# All three sizes steer with the SAME d estimated on ELF-B (results/direction_B.pt).

set -euo pipefail
export CUDA_VISIBLE_DEVICES=6 

HERE="$(dirname "$0")/.."
DB="$HERE/results/direction_B.pt"
for SC in B:ELF-B-owt-torch M:ELF-M-owt-torch L:ELF-L-owt-torch; do
  S="${SC%%:*}"; CK="${SC##*:}"
  echo "[benchmark] $S: compute-to-clean with d_B"
  python "$HERE/benchmark.py" --size "$S" --ckpt "$HERE/models/$CK" --lam 2.0 --target 1000 --steer-d "$DB" "$@"
done
