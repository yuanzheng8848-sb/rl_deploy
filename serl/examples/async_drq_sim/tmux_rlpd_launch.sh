#!/bin/bash
set -e

# use the default values if the env variables are not set
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXAMPLE_DIR=${EXAMPLE_DIR:-"$SCRIPT_DIR"}
CONDA_ENV=${CONDA_ENV:-"serl"}

cd "$EXAMPLE_DIR"
echo "Running from $(pwd)"

download_file() {
    local url="$1"
    local output="$2"
    local tmp_output="${output}.tmp"
    rm -f "$tmp_output"
    if command -v curl >/dev/null 2>&1; then
        curl -L -f --retry 3 -o "$tmp_output" "$url"
    else
        wget -O "$tmp_output" "$url"
    fi
    mv "$tmp_output" "$output"
}

# check if the pkl file exists, else download it
RESNET_FILE="resnet10_params.pkl"
if [ ! -s "$RESNET_FILE" ]; then
    PRETRAINED_FILE="$REPO_DIR/serl_launcher/serl_launcher/networks/$RESNET_FILE"
    if [ -f "$PRETRAINED_FILE" ]; then
        echo "$RESNET_FILE not found in $(pwd). Copying from $PRETRAINED_FILE..."
        cp "$PRETRAINED_FILE" "$RESNET_FILE"
    fi
fi
if [ ! -s "$RESNET_FILE" ]; then
    download_file \
        "https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl" \
        "$RESNET_FILE"
fi

# download trajectory data for offline RL
DATA_FILE="franka_lift_cube_image_20_trajs.pkl"
if [ ! -s "$DATA_FILE" ]; then
    download_file \
        "https://github.com/rail-berkeley/serl/releases/download/franka_sim_lift_cube_demos/franka_lift_cube_image_20_trajs.pkl" \
        "$DATA_FILE"
fi

# check if both file exists else throw error
if [ ! -s "$RESNET_FILE" ] || [ ! -s "$DATA_FILE" ]; then
    echo "Error: $RESNET_FILE or $DATA_FILE does not exist"
    exit 1
fi

# Create a new tmux session
tmux new-session -d -s serl_session

# Split the window vertically
tmux split-window -v

# Navigate to the activate the conda environment in the first pane
tmux send-keys -t serl_session:0.0 "conda activate $CONDA_ENV && bash run_actor.sh" C-m

# Navigate to the activate the conda environment in the second pane
tmux send-keys -t serl_session:0.1 "conda activate $CONDA_ENV && bash run_learner.sh --demo_path franka_lift_cube_image_20_trajs.pkl" C-m

# Attach to the tmux session
tmux attach-session -t serl_session

# kill the tmux session by running the following command
# tmux kill-session -t serl_session
