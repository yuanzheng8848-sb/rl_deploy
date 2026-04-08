#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false

python train_pick_place_2.py \
  --actor \
  --enable_two_stage_training=true \
  --random_steps=0 \
  --render
