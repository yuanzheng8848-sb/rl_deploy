#!/usr/bin/env python3

import argparse
import copy
import os
import pickle as pkl
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from pynput import keyboard


RL_DEPLOY_DIR = Path(__file__).resolve().parents[1]
if str(RL_DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(RL_DEPLOY_DIR))

from train import create_env, maybe_add_grasp_penalty  # noqa: E402


DEMO_ROOT = RL_DEPLOY_DIR / "demo" / "collected"
SUCCESS_DIR = DEMO_ROOT / "success"
FAILURE_DIR = DEMO_ROOT / "failure"
RAW_IMAGE_ROOT = DEMO_ROOT / "raw_images"
RAW_SUCCESS_DIR = RAW_IMAGE_ROOT / "success"
RAW_FAILURE_DIR = RAW_IMAGE_ROOT / "failure"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record bimanual 3DX demos in the exact training transition format."
    )
    parser.add_argument("--max-episodes", type=int, default=0, help="0 means unlimited.")
    parser.add_argument(
        "--success-dir",
        type=Path,
        default=SUCCESS_DIR,
        help="Directory for successful trajectory .pkl files.",
    )
    parser.add_argument(
        "--failure-dir",
        type=Path,
        default=FAILURE_DIR,
        help="Directory for failed trajectory .pkl files.",
    )
    parser.add_argument(
        "--discard-shorter-than",
        type=int,
        default=2,
        help="Discard trajectories with fewer than this many transitions.",
    )
    return parser.parse_args()


def make_keyboard_state():
    return {
        "label": None,
        "quit_requested": False,
    }


def start_keyboard_listener(state):
    def on_press(key):
        try:
            if key == keyboard.Key.enter:
                state["label"] = "success"
                print("[Recorder] ENTER -> mark trajectory SUCCESS")
            elif key == keyboard.Key.space:
                state["label"] = "failure"
                print("[Recorder] SPACE -> mark trajectory FAILURE")
            elif key == keyboard.Key.esc:
                state["quit_requested"] = True
                print("[Recorder] ESC -> quit after current loop")
        except Exception:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


def snapshot_raw_images(env):
    base_env = env.unwrapped
    latest = getattr(base_env, "latest_images_raw", None)
    if not isinstance(latest, dict):
        return {}

    snapshot = {}
    for key, value in latest.items():
        if value is None:
            continue
        snapshot[key] = np.array(value, copy=True)
    return snapshot


def save_raw_frames(target_dir: Path, trajectory_stem: str, raw_frames):
    if not raw_frames:
        print(f"[Recorder] no raw frames captured for {trajectory_stem}, skipping raw image export.")
        return None

    trajectory_dir = target_dir / trajectory_stem
    saved_count = 0
    for frame_idx, frame_dict in enumerate(raw_frames):
        if not isinstance(frame_dict, dict):
            continue
        for camera_name, image in frame_dict.items():
            if image is None:
                continue
            camera_dir = trajectory_dir / camera_name
            camera_dir.mkdir(parents=True, exist_ok=True)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_path = camera_dir / f"frame_{frame_idx:06d}.png"
            if cv2.imwrite(str(image_path), image_bgr):
                saved_count += 1

    if saved_count == 0:
        print(f"[Recorder] raw image export produced no files for {trajectory_stem}.")
        return None

    print(f"[Recorder] saved raw images -> {trajectory_dir} ({saved_count} files)")
    return trajectory_dir


def save_trajectory(target_dir: Path, raw_target_dir: Path, trajectory, raw_frames, label: str):
    target_dir.mkdir(parents=True, exist_ok=True)
    raw_target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trajectory_stem = f"{label}_{timestamp}"
    file_path = target_dir / f"{trajectory_stem}.pkl"
    with open(file_path, "wb") as handle:
        pkl.dump(trajectory, handle)
    save_raw_frames(raw_target_dir, trajectory_stem, raw_frames)
    print(f"[Recorder] saved {label} trajectory -> {file_path} ({len(trajectory)} transitions)")
    return file_path


def finalize_trajectory(trajectory, success: bool):
    if not trajectory:
        return trajectory

    finalized = [copy.deepcopy(transition) for transition in trajectory]
    for transition in finalized:
        transition["rewards"] = np.asarray(0.0, dtype=np.float32)
        transition["masks"] = np.asarray(1.0, dtype=np.float32)
        transition["dones"] = False

    finalized[-1]["rewards"] = np.asarray(1.0 if success else 0.0, dtype=np.float32)
    finalized[-1]["masks"] = np.asarray(0.0, dtype=np.float32)
    finalized[-1]["dones"] = True
    return finalized


def main():
    args = parse_args()
    os.makedirs(args.success_dir, exist_ok=True)
    os.makedirs(args.failure_dir, exist_ok=True)
    os.makedirs(RAW_SUCCESS_DIR, exist_ok=True)
    os.makedirs(RAW_FAILURE_DIR, exist_ok=True)

    env = create_env(
        fake_env=False,
        enable_classifier=False,
        enable_intervention=True,
    )
    keyboard_state = make_keyboard_state()
    listener = start_keyboard_listener(keyboard_state)

    print("[Recorder] Ready.")
    print("[Recorder] Use the same dual 3DX controls as training.")
    print("[Recorder] Press ENTER to save current trajectory as success.")
    print("[Recorder] Press SPACE to save current trajectory as failure.")
    print("[Recorder] Press ESC to quit.")

    episodes_saved = 0
    current_trajectory = []
    obs, _ = env.reset()
    current_raw_frames = [snapshot_raw_images(env)]

    try:
        while True:
            if keyboard_state["quit_requested"]:
                break
            if args.max_episodes > 0 and episodes_saved >= args.max_episodes:
                break

            label = keyboard_state["label"]
            if label is not None:
                keyboard_state["label"] = None
                if len(current_trajectory) < args.discard_shorter_than:
                    print(
                        f"[Recorder] trajectory too short ({len(current_trajectory)}), discarded."
                    )
                else:
                    success = label == "success"
                    finalized = finalize_trajectory(current_trajectory, success=success)
                    save_trajectory(
                        args.success_dir if success else args.failure_dir,
                        RAW_SUCCESS_DIR if success else RAW_FAILURE_DIR,
                        finalized,
                        current_raw_frames,
                        label=label,
                    )
                    episodes_saved += 1
                current_trajectory = []
                obs, _ = env.reset()
                current_raw_frames = [snapshot_raw_images(env)]
                continue

            action = np.zeros(env.action_space.shape, dtype=np.float32)
            next_obs, reward, done, truncated, info = env.step(action)

            if isinstance(info, dict) and info.get("intervention_idle", False):
                continue

            sampled_transition = info.get("sampled_transition") if isinstance(info, dict) else None
            if sampled_transition is not None:
                transition = {
                    "observations": sampled_transition["observations"],
                    "actions": np.asarray(sampled_transition["actions"], dtype=np.float32),
                    "next_observations": sampled_transition["next_observations"],
                    "rewards": np.asarray(sampled_transition["rewards"], dtype=np.float32),
                    "masks": np.asarray(
                        1.0 - float(sampled_transition["dones"]),
                        dtype=np.float32,
                    ),
                    "dones": bool(sampled_transition["dones"]),
                    "infos": copy.deepcopy(sampled_transition.get("infos", info)),
                }
            elif "intervene_action" in info:
                transition = {
                    "observations": obs,
                    "actions": np.asarray(info["intervene_action"], dtype=np.float32),
                    "next_observations": next_obs,
                    "rewards": np.asarray(reward, dtype=np.float32),
                    "masks": np.asarray(1.0 - float(done), dtype=np.float32),
                    "dones": bool(done),
                    "infos": copy.deepcopy(info),
                }
            else:
                obs = next_obs
                if done or truncated:
                    obs, _ = env.reset()
                continue

            maybe_add_grasp_penalty(transition)
            current_trajectory.append(copy.deepcopy(transition))
            obs = next_obs
            current_raw_frames.append(snapshot_raw_images(env))

            if done or truncated:
                print(
                    "[Recorder] environment episode ended before manual label; discarding current trajectory."
                )
                current_trajectory = []
                obs, _ = env.reset()
                current_raw_frames = [snapshot_raw_images(env)]

    finally:
        listener.stop()
        env.close()


if __name__ == "__main__":
    main()
