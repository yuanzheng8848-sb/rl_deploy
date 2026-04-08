#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false

python train_pick_place_3.py \
  --learner \
  --enable_two_stage_training=true \
  --exp_name=openarm_rl_2stage_gripper_policy \
  --training_starts=100 \
  --demo_pkl_variant=v2 \
  --demo_drop_over_limit_transitions
