import os
import sys
from unittest import mock

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [
    os.path.join(ROOT, "rl_robot_infra"),
    ROOT,
]

from openarm_env.camera.local_camera import build_cameras
from openarm_env.envs.openarm_env import DefaultOpenArmConfig, OpenArmEnv


class NoCameraConfig(DefaultOpenArmConfig):
    REALSENSE_CAMERAS = {}


def test_virtual_env_does_not_create_http_session_or_send_requests():
    with mock.patch("requests.Session", side_effect=AssertionError("HTTP session created")):
        env = OpenArmEnv(env_mode="virtual", hz=1000, config=NoCameraConfig())
        assert env.session is None
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        obs, reward, terminated, truncated, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        assert env.observation_space.contains(obs)
        assert reward == 0.0
        assert terminated is False
        assert truncated is False
        assert info["state_stale"] is False
        env.close()


def test_virtual_env_uses_gymnasium_truncation_for_time_limit():
    env = OpenArmEnv(
        env_mode="virtual",
        hz=1000,
        config=NoCameraConfig(),
        max_episode_length=1,
    )
    env.reset()
    _, _, terminated, truncated, _ = env.step(
        np.zeros(env.action_space.shape, dtype=np.float32)
    )
    assert terminated is False
    assert truncated is True
    env.close()


def test_tcp_velocity_is_estimated_from_pose_delta():
    env = OpenArmEnv(env_mode="virtual", hz=1000, config=NoCameraConfig())
    env.reset()
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[0] = 1.0
    obs, _, _, _, _ = env.step(action)
    assert obs["state"]["tcp_vel"].shape == (2, 6)
    assert obs["state"]["tcp_vel"][0, 0] > 0.0
    env.close()


def test_virtual_camera_builder_uses_explicit_config_only():
    cameras = build_cameras(
        {
            "image_primary": {
                "name": "head",
                "type": "usb",
                "width": 32,
                "height": 24,
                "fps": 1,
            }
        },
        virtual=True,
    )
    assert [name for name, _ in cameras] == ["image_primary"]
    frame = cameras[0][1].get_data()
    assert frame[0].shape == (24, 32, 3)
