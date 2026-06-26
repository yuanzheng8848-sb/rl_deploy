#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false

PYTHON_BIN="${PYTHON_BIN:-/home/sj/miniconda3/envs/zy/bin/python}"

# Usage: ./run_actor.sh [additional args...]
#
# Training mode: bimanual, keyboard-only rewards
# Spacemouse teleop enabled (SPACE=fail, ENTER=success, H=toggle handoff focus)

"${PYTHON_BIN}" train.py \
  --actor \
  --render \
  --random_steps=0 \
  "$@"
