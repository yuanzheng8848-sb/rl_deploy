#!/bin/bash
set -euo pipefail

export XLA_PYTHON_CLIENT_PREALLOCATE=false

python train_pick_place.py \
  --eval \
  --render
