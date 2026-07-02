#!/usr/bin/env bash
# Derive the 1.92% human acceptance bar from XSum -> results/human_reference.json.

set -euo pipefail
echo "[human_reference] deriving the acceptance bar from XSum"
python "$(dirname "$0")/../human_reference.py" "$@"
