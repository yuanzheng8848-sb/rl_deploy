#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false

PYTHON_BIN="${PYTHON_BIN:-/home/sj/miniconda3/envs/zy/bin/python}"

# Sequential eval iterates over all stages with checkpoints.
# Per-stage train_arm and gripper holds come from _STAGE_DEFAULTS in train.py.

"${PYTHON_BIN}" train.py \
  --eval \
  --render \
  --check
  --eval_sequential \
  "$@"
