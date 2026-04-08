import copy
import csv
import json
import pickle as pkl
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


FILE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = FILE_DIR / "demos_origin"
DEST_FILE = FILE_DIR / "merged_demos_2.pkl"
LOG_DIR = FILE_DIR / "processing_logs_2"

# Keep processed demo action semantics aligned with OpenArmEnv.step().
TRAIN_MAX_POS_DELTA_M = 0.01
TRAIN_MAX_ROT_DELTA_RAD = 0.05
TRAIN_MAX_GRIPPER_DELTA = 0.5236

# Trim leading/trailing silence in raw recordings.
TRANS_IDLE_THRESH = 0.002
ROT_IDLE_THRESH = 0.01
GRIPPER_IDLE_THRESH = 0.01


def resize_images(obs_dict):
    obs = copy.deepcopy(obs_dict)
    for key, value in list(obs.items()):
        if key.startswith("image_") and isinstance(value, np.ndarray):
            if value.shape[:2] != (128, 128):
                obs[key] = cv2.resize(value, (128, 128))
    return obs


def compute_body_delta(current_ee_pose, target_ee_pose):
    delta_pos = target_ee_pose[:3] - current_ee_pose[:3]
    rot_curr = R.from_quat(current_ee_pose[3:7])
    rot_target = R.from_quat(target_ee_pose[3:7])
    rot_delta = rot_curr.inv() * rot_target
    delta_euler = rot_delta.as_euler("xyz")
    delta_pos_body = rot_curr.inv().apply(delta_pos)
    return np.concatenate([delta_pos_body, delta_euler])


def translation_and_rotation_magnitude(current_state, next_state):
    current_state = np.asarray(current_state)
    next_state = np.asarray(next_state)
    delta_pos = np.linalg.norm(next_state[:3] - current_state[:3])
    rot_curr = R.from_quat(current_state[3:7])
    rot_next = R.from_quat(next_state[3:7])
    delta_rot = np.linalg.norm((rot_curr.inv() * rot_next).as_euler("xyz"))
    delta_grip = abs(float(next_state[7]) - float(current_state[7]))
    return float(delta_pos), float(delta_rot), float(delta_grip)


def is_idle_transition(transition):
    obs_state = np.asarray(transition["observations"]["state"])
    next_obs = transition.get("next_observations")
    if next_obs is not None:
        next_state = np.asarray(next_obs["state"])
        pos_mag, rot_mag, grip_mag = translation_and_rotation_magnitude(obs_state, next_state)
        return (
            pos_mag < TRANS_IDLE_THRESH
            and rot_mag < ROT_IDLE_THRESH
            and grip_mag < GRIPPER_IDLE_THRESH
        )

    action = np.asarray(transition["actions"])
    pos_mag = float(np.linalg.norm(action[:3]))
    rot_mag = float(np.linalg.norm(action[3:6]))
    grip_mag = float(abs(action[6])) if action.shape[0] >= 7 else 0.0
    return (
        pos_mag < TRANS_IDLE_THRESH
        and rot_mag < ROT_IDLE_THRESH
        and grip_mag < GRIPPER_IDLE_THRESH
    )


def find_active_bounds(transitions):
    if not transitions:
        return 0, 0

    start_idx = 0
    for i in range(len(transitions)):
        if not is_idle_transition(transitions[i]):
            start_idx = i
            break

    end_idx = len(transitions)
    for i in range(len(transitions) - 1, start_idx - 1, -1):
        if not is_idle_transition(transitions[i]):
            end_idx = i + 1
            break

    return start_idx, end_idx


def to_relative_state(raw_state, t_reset_inv):
    raw_state = np.asarray(raw_state)
    pos = raw_state[:3]
    quat = raw_state[3:7]
    gripper = raw_state[7:8]

    transform = np.eye(4)
    transform[:3, :3] = R.from_quat(quat).as_matrix()
    transform[:3, 3] = pos
    transform_rel = t_reset_inv @ transform
    euler_rel = R.from_matrix(transform_rel[:3, :3]).as_euler("xyz")
    return np.concatenate([gripper, transform_rel[:3, 3], euler_rel, np.zeros(6)])


def action_within_env_limits(action_physical):
    action_physical = np.asarray(action_physical)
    pos_ok = np.all(np.abs(action_physical[:3]) <= TRAIN_MAX_POS_DELTA_M + 1e-9)
    rot_ok = np.all(np.abs(action_physical[3:6]) <= TRAIN_MAX_ROT_DELTA_RAD + 1e-9)
    return bool(pos_ok and rot_ok)


def build_processed_transition(src_transition, dst_transition, t_reset_inv):
    src_raw = np.asarray(src_transition["observations"]["state"])
    dst_raw = np.asarray(dst_transition["observations"]["state"])
    action_physical = np.concatenate(
        [compute_body_delta(src_raw[:7], dst_raw[:7]), [dst_transition["actions"][6]]]
    ).astype(np.float32)

    obs = resize_images(src_transition["observations"])
    next_obs = resize_images(dst_transition["observations"])
    obs["state"] = to_relative_state(obs["state"], t_reset_inv)
    next_obs["state"] = to_relative_state(next_obs["state"], t_reset_inv)

    reward = np.asarray(max(float(np.asarray(src_transition["rewards"])), float(np.asarray(dst_transition["rewards"]))), dtype=np.float32)
    done = bool(src_transition.get("dones", False) or dst_transition.get("dones", False))
    return {
        "observations": obs,
        "next_observations": next_obs,
        "actions": action_physical,
        "rewards": reward,
        "dones": done,
        "masks": np.asarray(0.0 if done else 1.0, dtype=np.float32),
    }, action_physical


def aggregate_episode(transitions, kept_indices):
    output = []
    if len(kept_indices) < 2:
        return output, {
            "deleted_frames": 0,
            "adjacent_keeps": 0,
            "over_limit_pairs": 0,
        }

    reset_state = np.asarray(transitions[kept_indices[0]]["observations"]["state"])
    t_reset = np.eye(4)
    t_reset[:3, :3] = R.from_quat(reset_state[3:7]).as_matrix()
    t_reset[:3, 3] = reset_state[:3]
    t_reset_inv = np.linalg.inv(t_reset)

    deleted_frames = 0
    adjacent_keeps = 0
    over_limit_pairs = 0

    anchor_pos = 0
    while anchor_pos < len(kept_indices) - 1:
        src_idx = kept_indices[anchor_pos]
        best_next_pos = anchor_pos + 1

        for candidate_pos in range(anchor_pos + 1, len(kept_indices)):
            dst_idx = kept_indices[candidate_pos]
            src_state = np.asarray(transitions[src_idx]["observations"]["state"])
            dst_state = np.asarray(transitions[dst_idx]["observations"]["state"])
            delta = np.concatenate(
                [compute_body_delta(src_state[:7], dst_state[:7]), [transitions[dst_idx]["actions"][6]]]
            )

            if action_within_env_limits(delta):
                best_next_pos = candidate_pos
            else:
                break

        if best_next_pos == anchor_pos + 1:
            adjacent_keeps += 1

        src_transition = transitions[src_idx]
        dst_transition = transitions[kept_indices[best_next_pos]]
        new_transition, action_physical = build_processed_transition(src_transition, dst_transition, t_reset_inv)
        if not action_within_env_limits(action_physical):
            over_limit_pairs += 1

        output.append(new_transition)
        deleted_frames += max(0, best_next_pos - anchor_pos - 1)
        anchor_pos = best_next_pos

    return output, {
        "deleted_frames": deleted_frames,
        "adjacent_keeps": adjacent_keeps,
        "over_limit_pairs": over_limit_pairs,
    }


def process_demos_2():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source directory {SOURCE_DIR} does not exist.")

    files = sorted(SOURCE_DIR.glob("*.pkl"))
    print(f"Processing {len(files)} demo files from {SOURCE_DIR}")
    print(
        "Constraints: "
        f"max_pos={TRAIN_MAX_POS_DELTA_M:.4f}m "
        f"max_rot={TRAIN_MAX_ROT_DELTA_RAD:.4f}rad"
    )

    merged = []
    summary_rows = []

    for file_path in tqdm(files, desc="process_demos_2"):
        with open(file_path, "rb") as f:
            transitions = pkl.load(f)

        if not transitions:
            continue

        start_idx, end_idx = find_active_bounds(transitions)
        kept_indices = list(range(start_idx, end_idx))
        episode_transitions, stats = aggregate_episode(transitions, kept_indices)
        merged.extend(episode_transitions)

        summary_rows.append(
            {
                "file": file_path.name,
                "original_frames": len(transitions),
                "trimmed_frames": max(0, end_idx - start_idx),
                "output_transitions": len(episode_transitions),
                "deleted_frames": stats["deleted_frames"],
                "adjacent_keeps": stats["adjacent_keeps"],
                "over_limit_pairs": stats["over_limit_pairs"],
            }
        )

    with open(DEST_FILE, "wb") as f:
        pkl.dump(merged, f)

    csv_path = LOG_DIR / "processing_summary_2.csv"
    json_path = LOG_DIR / "processing_summary_2.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "original_frames",
                "trimmed_frames",
                "output_transitions",
                "deleted_frames",
                "adjacent_keeps",
                "over_limit_pairs",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    aggregate_summary = {
        "source_dir": str(SOURCE_DIR),
        "dest_file": str(DEST_FILE),
        "num_files": len(summary_rows),
        "num_transitions": len(merged),
        "total_original_frames": int(sum(row["original_frames"] for row in summary_rows)),
        "total_trimmed_frames": int(sum(row["trimmed_frames"] for row in summary_rows)),
        "total_output_transitions": int(sum(row["output_transitions"] for row in summary_rows)),
        "total_deleted_frames": int(sum(row["deleted_frames"] for row in summary_rows)),
        "total_adjacent_keeps": int(sum(row["adjacent_keeps"] for row in summary_rows)),
        "total_over_limit_pairs": int(sum(row["over_limit_pairs"] for row in summary_rows)),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(aggregate_summary, f, ensure_ascii=True, indent=2)

    print(f"Saved merged demos to {DEST_FILE}")
    print(f"Saved logs to {csv_path} and {json_path}")
    print(json.dumps(aggregate_summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    process_demos_2()
