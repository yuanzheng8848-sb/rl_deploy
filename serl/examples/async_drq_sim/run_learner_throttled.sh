export MUJOCO_GL=glfw && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
python async_drq_sim.py "$@" \
    --learner \
    --exp_name=serl_dev_drq_sim_test_resnet \
    --seed 0 \
    --training_starts 100 \
    --critic_actor_ratio 4 \
    --encoder_type resnet-pretrained \
    --demo_path franka_lift_cube_image_20_trajs.pkl
    # to disable wandb upload, append --debug
