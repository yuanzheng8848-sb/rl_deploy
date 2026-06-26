export MUJOCO_GL=glfw && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python async_drq_sim.py "$@" \
    --actor \
    --render \
    --exp_name=serl_dev_drq_sim_test_resnet \
    --seed 0 \
    --random_steps 100 \
    --encoder_type resnet-pretrained \
    --actor_max_step_hz 3
    # actor_max_step_hz limits the actor to ~2 env steps/sec to match the learner.
    # Set to 0 (or override on the command line) for unlimited speed.
    # to disable wandb upload, append --debug
