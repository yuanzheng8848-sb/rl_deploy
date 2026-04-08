import pickle as pkl
from dataclasses import dataclass

import numpy as np


TRAINING_IMAGE_KEYS = ["image_primary", "image_left", "image_right"]
DEMO_PATH_V1 = "/home/sj/Desktop/zy/moqi_workspace/rl_deploy/demo/merged_demos.pkl"
DEMO_PATH_V2 = "/home/sj/Desktop/zy/moqi_workspace/rl_deploy/demo/merged_demos_2.pkl"
SCALE_POS = 0.01
SCALE_ROT = 0.05
GRIPPER_SCALE = 0.5236


@dataclass
class DemoLoadStats:
    total_raw: int = 0
    loaded: int = 0
    over_limit: int = 0
    skipped: int = 0

    def summary(self) -> str:
        return (
            f"loaded={self.loaded}, over_limit={self.over_limit}, "
            f"skipped={self.skipped}, total_raw={self.total_raw}"
        )


def resolve_demo_path(demo_pkl_variant: str, demo_path: str) -> str:
    if demo_pkl_variant == "v1":
        return DEMO_PATH_V1
    if demo_pkl_variant == "v2":
        return DEMO_PATH_V2
    return demo_path


def _normalize_demo_action(action_physical):
    action_physical = np.asarray(action_physical, dtype=np.float32)
    action_pos_norm = action_physical[:3] / SCALE_POS
    action_rot_norm = action_physical[3:6] / SCALE_ROT
    gripper_norm = (float(action_physical[6]) / GRIPPER_SCALE) + 1.0

    over_limit = (
        np.any(np.abs(action_pos_norm) > 1.0)
        or np.any(np.abs(action_rot_norm) > 1.0)
        or abs(float(gripper_norm)) > 1.0
    )

    action_pos_norm = np.clip(action_pos_norm, -1.0, 1.0)
    action_rot_norm = np.clip(action_rot_norm, -1.0, 1.0)
    gripper_norm = np.clip(gripper_norm, -1.0, 1.0)

    final_action = np.concatenate(
        [action_pos_norm, action_rot_norm, np.array([gripper_norm], dtype=np.float32)]
    ).astype(np.float32)
    return final_action, over_limit


def _mask_observation_dict(source_dict, valid_keys):
    target_dict = {}
    for key, value in source_dict.items():
        if key not in valid_keys and key != "state":
            continue
        if key == "image_left":
            target_dict[key] = np.zeros_like(value)
        elif key == "state":
            target_dict[key] = np.asarray(value, dtype=np.float32)
        else:
            target_dict[key] = np.asarray(value)
    return target_dict


def convert_demo_transition(
    raw_transition,
    training_image_keys=TRAINING_IMAGE_KEYS,
    drop_over_limit_transitions=False,
):
    valid_keys = set(training_image_keys) | {"state"}

    obs_state = np.asarray(raw_transition["observations"]["state"], dtype=np.float32)
    if raw_transition["next_observations"] is not None:
        next_obs_state = np.asarray(
            raw_transition["next_observations"]["state"], dtype=np.float32
        )
        next_obs_source = {
            k: v for k, v in raw_transition["next_observations"].items() if k != "state"
        }
    else:
        next_obs_state = obs_state
        next_obs_source = {
            k: v for k, v in raw_transition["observations"].items() if k != "state"
        }

    final_action, over_limit = _normalize_demo_action(raw_transition["actions"])
    if over_limit and drop_over_limit_transitions:
        return None, {"over_limit": True, "skipped": True}

    obs_dict = _mask_observation_dict(raw_transition["observations"], valid_keys)
    next_obs_dict = _mask_observation_dict(next_obs_source, valid_keys)
    obs_dict["state"] = obs_state
    next_obs_dict["state"] = next_obs_state

    done = bool(raw_transition["dones"])
    reward_val = np.asarray(raw_transition["rewards"], dtype=np.float32)
    if reward_val.shape != ():
        reward_val = np.asarray(float(reward_val.reshape(-1)[0]), dtype=np.float32)

    transition = {
        "observations": obs_dict,
        "next_observations": next_obs_dict,
        "actions": final_action,
        "rewards": reward_val,
        "masks": np.asarray(0.0 if done else 1.0, dtype=np.float32),
        "dones": done,
    }
    return transition, {"over_limit": bool(over_limit), "skipped": False}


def load_demo_transitions(
    demo_path,
    training_image_keys=TRAINING_IMAGE_KEYS,
    drop_over_limit_transitions=False,
):
    with open(demo_path, "rb") as f:
        raw_transitions = pkl.load(f)

    stats = DemoLoadStats(total_raw=len(raw_transitions))
    transitions = []
    for raw_transition in raw_transitions:
        transition, info = convert_demo_transition(
            raw_transition,
            training_image_keys=training_image_keys,
            drop_over_limit_transitions=drop_over_limit_transitions,
        )
        if info["over_limit"]:
            stats.over_limit += 1
        if info["skipped"]:
            stats.skipped += 1
            continue
        transitions.append(transition)
        stats.loaded += 1
    return transitions, stats


def insert_transitions_into_buffer(buffer, transitions):
    for transition in transitions:
        buffer.insert(transition)

