import copy
import sys
from pathlib import Path

import numpy as np
import pytest
from gymnasium import spaces


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from data_contract import (
    DATA_CONTRACT,
    finalize_labeled_trajectory,
    stamp_transition,
    to_replay_transition,
    validate_transition,
)
from rl_launcher.data.replay_buffer import ReplayBuffer


def _transition():
    return {
        "observations": {"state": np.zeros(1, dtype=np.float32)},
        "actions": np.zeros(14, dtype=np.float32),
        "next_observations": {"state": np.zeros(1, dtype=np.float32)},
        "rewards": np.asarray(0.0, dtype=np.float32),
        "masks": np.asarray(1.0, dtype=np.float32),
        "dones": False,
        "infos": {},
    }


def test_current_transition_is_stamped_and_accepted():
    stamped = stamp_transition(_transition())
    assert stamped["infos"]["data_contract"] == DATA_CONTRACT
    assert validate_transition(stamped) is stamped


def test_legacy_transition_without_contract_is_rejected():
    with pytest.raises(ValueError, match="Legacy transition data is intentionally unsupported"):
        validate_transition(_transition(), source="old.pkl[0]")


def test_wrong_timing_or_action_semantics_is_rejected():
    stamped = stamp_transition(_transition())
    wrong = copy.deepcopy(stamped)
    wrong["infos"]["data_contract"]["transition_hz"] = 5.0
    with pytest.raises(ValueError, match="incompatible or missing data contract"):
        validate_transition(wrong)


def test_non_normalized_velocity_action_is_rejected():
    stamped = stamp_transition(_transition())
    stamped["actions"][0] = 1.1
    with pytest.raises(ValueError, match=r"within \[-1, 1\]"):
        validate_transition(stamped)


def test_success_demo_has_one_terminal_sparse_reward():
    finalized = finalize_labeled_trajectory([_transition()] * 3, "success")
    assert [float(item["rewards"]) for item in finalized] == [0.0, 0.0, 1.0]
    assert [bool(item["dones"]) for item in finalized] == [False, False, True]
    assert [float(item["masks"]) for item in finalized] == [1.0, 1.0, 0.0]
    assert all(item["infos"]["trajectory_label"] == "success" for item in finalized)


def test_failure_demo_terminates_without_success_reward():
    finalized = finalize_labeled_trajectory([_transition()] * 2, "failure")
    assert [float(item["rewards"]) for item in finalized] == [0.0, 0.0]
    assert [bool(item["dones"]) for item in finalized] == [False, True]
    assert [float(item["masks"]) for item in finalized] == [1.0, 0.0]


def test_persisted_metadata_is_explicitly_removed_before_replay():
    persisted = stamp_transition({
        **_transition(),
        "rewards": np.asarray(0.0, dtype=np.float32),
        "masks": np.asarray(1.0, dtype=np.float32),
        "dones": False,
    })
    replay_item = to_replay_transition(persisted)
    assert set(replay_item) == {
        "observations", "next_observations", "actions", "rewards", "masks", "dones"
    }


def test_replay_rejects_extra_fields_instead_of_silently_dropping_them():
    obs_space = spaces.Dict({"state": spaces.Box(-1, 1, shape=(1,), dtype=np.float32)})
    buffer = ReplayBuffer(obs_space, spaces.Box(-1, 1, shape=(14,), dtype=np.float32), 4)
    item = {
        "observations": {"state": np.zeros(1, dtype=np.float32)},
        "next_observations": {"state": np.zeros(1, dtype=np.float32)},
        "actions": np.zeros(14, dtype=np.float32),
        "rewards": np.asarray(0.0, dtype=np.float32),
        "masks": np.asarray(1.0, dtype=np.float32),
        "dones": False,
        "unused": 123,
    }
    with pytest.raises(ValueError, match=r"extra=\['unused'\]"):
        buffer.insert(item)


def test_persisted_transition_rejects_unknown_fields_before_projection():
    persisted = stamp_transition({
        **_transition(),
        "rewards": np.asarray(0.0, dtype=np.float32),
        "masks": np.asarray(1.0, dtype=np.float32),
        "dones": False,
    })
    persisted["unused"] = 123
    with pytest.raises(ValueError, match=r"extra=\['unused'\]"):
        validate_transition(persisted)
