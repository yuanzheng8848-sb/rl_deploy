#!/bin/bash
# Run standalone BC evaluation (no dependency on train.py)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false

PYTHON_BIN="${PYTHON_BIN:-/home/sj/miniconda3/envs/zy/bin/python}"

"${PYTHON_BIN}" eval_bc_standalone.py \
  --demo_path=./demo/collected/success \
  --checkpoint_path=./checkpoints_bc \
  --eval_steps=1000 \
  --batch_size=256 \
  "$@"
