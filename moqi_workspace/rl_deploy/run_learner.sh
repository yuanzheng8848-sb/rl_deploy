#!/bin/bash
set -euo pipefail

export XLA_PYTHON_CLIENT_PREALLOCATE=false

python train_pick_place.py \
  --learner \
  --exp_name=openarm_rl_test \
  --training_starts=100 \
  --demo_pkl_variant=v2 \
  --demo_drop_over_limit_transitions
