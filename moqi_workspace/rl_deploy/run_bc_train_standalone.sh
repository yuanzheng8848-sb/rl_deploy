#!/bin/bash
# Run standalone BC training (no dependency on train.py)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false

PYTHON_BIN="${PYTHON_BIN:-/home/sj/miniconda3/envs/zy/bin/python}"

"${PYTHON_BIN}" train_bc_standalone.py \
  --demo_path=./demo/collected/success \
  --checkpoint_path=./checkpoints_bc \
  --bc_steps=5000 \
  --batch_size=256 \
  --log_period=10 \
  --checkpoint_period=500 \
  "$@"
