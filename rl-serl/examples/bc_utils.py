"""Shared helpers for rl-serl BC training and evaluation."""
import copy
import glob
import os
import pickle as pkl
from pathlib import Path

import numpy as np


GRIPPER_ACTION_INDICES = (6, 13)
CONTINUOUS_ACTION_INDICES = tuple(i for i in range(14) if i not in GRIPPER_ACTION_INDICES)


def continuous_action(action):
    action = action_vector(action)
    return action[..., CONTINUOUS_ACTION_INDICES]


def gripper_classes(action):
    action = action_vector(action)
    left = np.clip(np.rint(action[..., 6]).astype(np.int32) + 1, 0, 2)
    right = np.clip(np.rint(action[..., 13]).astype(np.int32) + 1, 0, 2)
    joint = left * 3 + right
    return left, right, joint


def action_vector(action):
    action = np.asarray(action, dtype=np.float32)
    if action.ndim > 1 and action.shape[0] == 1:
        action = action[0]
    return action


def split_predicted_action(action):
    action = np.asarray(action, dtype=np.float32)
    return continuous_action(action), *gripper_classes(action)


def iter_pkl_paths(dir_path):
    if not dir_path:
        return []
    return sorted(glob.glob(os.path.join(os.fspath(dir_path), "*.pkl")))


def load_transition_paths(dir_path, max_trajs=0):
    paths = iter_pkl_paths(dir_path)
    if max_trajs and max_trajs > 0:
        paths = paths[:max_trajs]
    return paths


def load_trajectories(dir_path, max_trajs=0):
    trajectories = []
    for path in load_transition_paths(dir_path, max_trajs=max_trajs):
        with open(path, "rb") as handle:
            traj = pkl.load(handle)
        if isinstance(traj, list):
            trajectories.append((Path(path), traj))
    return trajectories


def prepare_transition(transition, skip_zero_action=False):
    action = np.asarray(transition.get("actions"), dtype=np.float32)
    if action.shape[-1] != 14:
        return None
    if skip_zero_action and float(np.linalg.norm(action)) == 0.0:
        return None
    prepared = copy.deepcopy(transition)
    prepared["actions"] = action
    prepared["rewards"] = np.asarray(prepared.get("rewards", 0.0), dtype=np.float32)
    prepared["masks"] = np.asarray(prepared.get("masks", 1.0), dtype=np.float32)
    prepared["dones"] = bool(prepared.get("dones", False))
    prepared["grasp_penalty"] = np.asarray(
        prepared.get("grasp_penalty", 0.0), dtype=np.float32
    )
    return prepared


def load_demo_dir_into_buffer(dir_path, data_store, skip_zero_action=False, max_trajs=0):
    loaded_files = 0
    loaded_transitions = 0
    skipped = 0
    for _, traj in load_trajectories(dir_path, max_trajs=max_trajs):
        loaded_files += 1
        # Skip the first frame of each demo (index 0) due to discontinuities
        for idx, transition in enumerate(traj):
            if idx == 0:
                skipped += 1
                continue
            prepared = prepare_transition(transition, skip_zero_action=skip_zero_action)
            if prepared is None:
                skipped += 1
                continue
            data_store.insert(prepared)
            loaded_transitions += 1
    return loaded_files, loaded_transitions, skipped


def summarize(values):
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "median": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
    }
