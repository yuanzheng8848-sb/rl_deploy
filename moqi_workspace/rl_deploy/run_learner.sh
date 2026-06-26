#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false

PYTHON_BIN="${PYTHON_BIN:-/home/sj/miniconda3/envs/zy/bin/python}"

# Usage: ./run_learner.sh [additional args...]
#
# Training mode: bimanual, keyboard-only rewards (SPACE=fail, ENTER=success)
# Optionally load a BC checkpoint as initialization:
#   --checkpoint_path=./checkpoints_bc  (loads BC warmup)
#
# Or continue RL training from RL checkpoint:
#   --checkpoint_path=./checkpoints_rl

"${PYTHON_BIN}" train.py \
  --learner \
  "$@"
