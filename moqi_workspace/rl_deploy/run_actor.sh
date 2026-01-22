#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python train_rl.py --actor --arm=right  --exp_name=openarm_rl_test --random_steps=0 --training_starts=200 --ip=10.255.23.70
