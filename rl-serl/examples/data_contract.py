"""Strict on-disk transition contract for the current OpenArm architecture."""

import copy

import numpy as np


DATA_CONTRACT = {
    "name": "openarm_velocity_v1",
    "version": 1,
    "action_semantics": "normalized_cartesian_velocity",
    "transition_hz": 20.0,
    "translation_scale_m_s": 0.05,
    "rotation_scale_rad_s": 0.25,
    "rotation_representation": "rotvec_action_quaternion_xyzw_pose",
    "reward_semantics": "sparse_terminal_success",
}

REPLAY_FIELDS = (
    "observations",
    "next_observations",
    "actions",
    "rewards",
    "masks",
    "dones",
)
PERSISTED_FIELDS = frozenset((*REPLAY_FIELDS, "infos"))
PERSISTED_INFO_FIELDS = frozenset(("data_contract", "trajectory_label", "terminal_success"))


def stamp_transition(transition):
    """Return a copy explicitly marked as data from the current architecture."""
    stamped = copy.deepcopy(transition)
    infos = dict(stamped.get("infos", {}))
    infos["data_contract"] = dict(DATA_CONTRACT)
    stamped["infos"] = infos
    return stamped


def validate_transition(transition, source="transition"):
    """Reject legacy/ambiguous data instead of attempting compatibility conversion."""
    if not isinstance(transition, dict):
        raise ValueError(f"{source}: transition must be a dict")
    actual_fields = set(transition)
    if actual_fields != PERSISTED_FIELDS:
        missing = sorted(PERSISTED_FIELDS - actual_fields)
        extra = sorted(actual_fields - PERSISTED_FIELDS)
        raise ValueError(f"{source}: persisted fields mismatch: missing={missing}, extra={extra}")
    infos = transition.get("infos")
    if not isinstance(infos, dict):
        raise ValueError(f"{source}: infos must be a dict")
    extra_info = sorted(set(infos) - PERSISTED_INFO_FIELDS)
    if extra_info:
        raise ValueError(f"{source}: unknown persisted info fields: {extra_info}")
    label_fields = {"trajectory_label", "terminal_success"}
    if bool(label_fields & set(infos)) and not label_fields.issubset(infos):
        raise ValueError(f"{source}: trajectory label metadata must be complete")
    contract = infos.get("data_contract")
    if contract != DATA_CONTRACT:
        raise ValueError(
            f"{source}: incompatible or missing data contract; expected "
            f"{DATA_CONTRACT['name']!r} v{DATA_CONTRACT['version']}. "
            "Legacy transition data is intentionally unsupported."
        )
    action = np.asarray(transition.get("actions"), dtype=np.float32)
    if action.shape != (14,) or not np.all(np.isfinite(action)):
        raise ValueError(f"{source}: actions must be one finite float32 vector with shape (14,)")
    if np.any(action < -1.00001) or np.any(action > 1.00001):
        raise ValueError(f"{source}: normalized velocity actions must be within [-1, 1]")
    for key in ("observations", "next_observations"):
        obs = transition.get(key)
        if not isinstance(obs, dict):
            raise ValueError(f"{source}: missing dict field {key!r}")
    return transition


def to_replay_transition(transition):
    """Explicitly project persisted data onto the exact learner schema."""
    return {key: copy.deepcopy(transition[key]) for key in REPLAY_FIELDS}


def finalize_labeled_trajectory(trajectory, label):
    """Apply the same sparse terminal reward semantics used by online RLPD."""
    if label not in ("success", "failure"):
        raise ValueError(f"Unknown trajectory label {label!r}")
    finalized = []
    last_idx = len(trajectory) - 1
    for idx, transition in enumerate(trajectory):
        item = copy.deepcopy(transition)
        terminal = idx == last_idx
        reward = float(label == "success" and terminal)
        item["rewards"] = np.asarray(reward, dtype=np.float32)
        item["masks"] = np.asarray(0.0 if terminal else 1.0, dtype=np.float32)
        item["dones"] = bool(terminal)
        infos = dict(item.get("infos", {}))
        infos["trajectory_label"] = label
        infos["terminal_success"] = bool(reward)
        item["infos"] = infos
        finalized.append(stamp_transition(item))
    return finalized
