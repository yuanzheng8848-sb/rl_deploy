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
FILE="resnet10_params.pkl"
if [ ! -s "$FILE" ]; then
    PRETRAINED_FILE="$REPO_DIR/serl_launcher/serl_launcher/networks/$FILE"
    if [ -f "$PRETRAINED_FILE" ]; then
        echo "$FILE not found in $(pwd). Copying from $PRETRAINED_FILE..."
        cp "$PRETRAINED_FILE" "$FILE"
    fi
fi
if [ ! -s "$FILE" ]; then
    echo "$FILE not found in $(pwd). Downloading..."
    download_file \
        "https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl" \
        "$FILE"
fi

# if pretrained weights file not exists, throw error
if [ ! -s "$FILE" ]; then
    echo "Error: $FILE not found."
    exit 1
fi

# Create a new tmux session
tmux new-session -d -s serl_session

# Split the window vertically
tmux split-window -v

# Navigate to the activate the conda environment in the first pane
tmux send-keys -t serl_session:0.0 "conda activate $CONDA_ENV && bash run_actor.sh" C-m

# Navigate to the activate the conda environment in the second pane
tmux send-keys -t serl_session:0.1 "conda activate $CONDA_ENV && bash run_learner.sh" C-m

# Attach to the tmux session
tmux attach-session -t serl_session

# kill the tmux session by running the following command
# tmux kill-session -t serl_session
