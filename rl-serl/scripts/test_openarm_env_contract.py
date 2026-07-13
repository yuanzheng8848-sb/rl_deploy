import os
import sys
from unittest import mock

import numpy as np
import yaml


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [
    os.path.join(ROOT, "rl_robot_infra"),
    os.path.join(ROOT, "rl_launcher"),
    os.path.join(ROOT, "examples"),
    ROOT,
]

import gymnasium as gym

from openarm_control.gripper import GripperCalibration
from openarm_env.camera.local_camera import build_cameras, load_deployment_camera_config
from openarm_env.envs.openarm_env import DefaultOpenArmConfig, OpenArmEnv
from openarm_env.envs.reward_wrappers import MultiCameraBinaryRewardClassifierWrapper
from openarm_env.envs import wrappers as openarm_wrappers
from openarm_env.utils.transformations import construct_twist_rotation_matrix
from rl_launcher.wrappers import ChunkingWrapper, SERLObsWrapper


class NoCameraConfig(DefaultOpenArmConfig):
    CAMERAS = {}


class InvalidBaseCameraConfig(DefaultOpenArmConfig):
    CAMERAS = {"image_primary": {"name": "head", "type": "usb"}}


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


def test_base_env_rejects_camera_configuration():
    try:
        OpenArmEnv(env_mode="virtual", config=InvalidBaseCameraConfig())
    except NotImplementedError:
        pass
    else:
        raise AssertionError("camera acquisition must be owned by LocalOpenArmEnv")


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
    assert cameras[0][1].read_rgb().shape == (24, 32, 3)


def test_camera_config_rejects_legacy_aliases_and_loads_deployment_mapping():
    try:
        build_cameras({"image_primary": "local-head"}, virtual=True)
    except TypeError:
        pass
    else:
        raise AssertionError("legacy camera aliases must not be accepted")

    cameras = load_deployment_camera_config()
    assert set(cameras) == {"image_primary", "image_left", "image_right"}
    assert cameras["image_primary"]["type"] == "usb"
    assert cameras["image_left"]["serial"] == "150622074105"


def test_gripper_calibration_owns_raw_units_and_applies_hysteresis():
    calibration = GripperCalibration(
        open_position=[-1.0, -1.0],
        closed_position=[0.05, 0.05],
        open_threshold=[-0.65, -0.65],
        close_threshold=[-0.30, -0.30],
    )
    np.testing.assert_allclose(calibration.target_from_closed([False, True]), [-1.0, 0.05])
    np.testing.assert_array_equal(calibration.update_from_position([-1.0, 0.0]), [False, True])
    np.testing.assert_array_equal(calibration.update_from_position([-0.5, -0.5]), [False, True])
    np.testing.assert_array_equal(calibration.update_from_position([0.0, -1.0]), [True, False])


def test_control_config_is_single_source_for_servo_home_and_gripper_calibration():
    config_path = os.path.join(ROOT, "rl_robot_infra", "openarm_configs", "control.yaml")
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert set(config) == {"servo", "home", "gripper"}
    assert config["servo"]["backend"] == "analytic"
    assert config["servo"]["hz"] == 100.0
    assert len(config["home"]["joint_position"]) == 14
    calibration = GripperCalibration.from_config(config["gripper"])
    np.testing.assert_allclose(calibration.target_from_closed([False, True]), [-1.0, 0.05])


def test_crop_batch_shape_supports_non_square_network_size():
    frames = np.zeros((2, 40, 60, 3), dtype=np.uint8)
    with mock.patch.object(openarm_wrappers, "MODEL_IMAGE_SIZE", (32, 24)):
        cropped = openarm_wrappers.crop_rgb_image(frames, crop_ratio=0.5)
    assert cropped.shape == (2, 24, 32, 3)


def test_dual_device_toggle_requests_are_merged_once_and_both_consumed():
    devices = [
        {"intervention_toggle_requested": True},
        {"intervention_toggle_requested": True},
    ]
    assert openarm_wrappers.consume_intervention_toggle_requests(devices) is True
    assert all("intervention_toggle_requested" not in device for device in devices)
    assert openarm_wrappers.consume_intervention_toggle_requests(devices) is False


def test_tcp_twist_basis_change_has_no_translation_coupling():
    pose = np.array([1.2, -0.7, 0.4, 0.0, 0.0, 0.0, 1.0])
    transform = construct_twist_rotation_matrix(pose)
    local_twist = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    np.testing.assert_allclose(transform, np.eye(6), atol=1e-12)
    np.testing.assert_allclose(transform @ local_twist, local_twist, atol=1e-12)


def test_tcp_twist_basis_change_rotates_linear_and_angular_parts_consistently():
    quat = openarm_wrappers.R.from_euler("z", np.pi / 2).as_quat()
    pose = np.concatenate(([0.3, -0.2, 0.5], quat))
    transform = construct_twist_rotation_matrix(pose)
    local_twist = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    expected_world = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0])

    np.testing.assert_allclose(transform @ local_twist, expected_world, atol=1e-7)
    np.testing.assert_allclose(transform.T @ expected_world, local_twist, atol=1e-7)


def test_dual_relative_action_transform_round_trips_without_touching_grippers():
    env = OpenArmEnv(env_mode="virtual", hz=1000, config=NoCameraConfig())
    wrapper = openarm_wrappers.DualRelativeFrame(env)
    quat = openarm_wrappers.R.from_euler("xyz", [0.2, -0.3, 0.4]).as_quat()
    poses = np.array(
        [
            [0.4, 0.2, 0.3, *quat],
            [-0.4, -0.2, 0.3, *quat],
        ]
    )
    wrapper._update_twist_transforms(poses)
    action = np.linspace(-1.0, 1.0, 14, dtype=np.float32)

    world_action = wrapper.transform_action(action)
    recovered = wrapper.transform_action_inv(world_action)

    np.testing.assert_allclose(recovered, action, atol=1e-6)
    np.testing.assert_allclose(world_action[[6, 13]], action[[6, 13]], atol=0.0)
    env.close()


def test_intervention_builds_world_command_once_and_policy_command_for_replay():
    base_env = OpenArmEnv(env_mode="virtual", hz=1000, config=NoCameraConfig())
    relative = openarm_wrappers.DualRelativeFrame(base_env)
    quat = openarm_wrappers.R.from_euler("z", np.pi / 2).as_quat()
    poses = np.array([[0.2, 0.1, 0.3, *quat], [-0.2, 0.1, 0.3, *quat]])
    relative._update_twist_transforms(poses)
    with mock.patch.object(openarm_wrappers, "InputDevice", None), mock.patch.object(
        openarm_wrappers, "ecodes", None
    ):
        intervention = openarm_wrappers.DualSpacemouseIntervention(
            relative,
            trans_denom=1.0,
            deadzone=0.0,
        )
    intervention._left["axes"]["x"] = 1.0

    policy_action, world_action = intervention._build_intervene_actions()

    np.testing.assert_allclose(world_action[:3], [1.0, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(policy_action[:3], [0.0, -1.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(relative.transform_action(policy_action), world_action, atol=1e-7)
    intervention.close()


def test_intervention_runs_four_80hz_target_updates_per_20hz_transition():
    base_env = OpenArmEnv(env_mode="virtual", hz=20, config=NoCameraConfig())
    relative = openarm_wrappers.DualRelativeFrame(base_env)
    relative._update_twist_transforms(base_env.currpos)
    with mock.patch.object(openarm_wrappers, "InputDevice", None), mock.patch.object(
        openarm_wrappers, "ecodes", None
    ):
        intervention = openarm_wrappers.DualSpacemouseIntervention(
            relative,
            trans_denom=1.0,
            deadzone=0.0,
            control_hz=80.0,
        )
    intervention._intervention_mode = True
    intervention._target_pose_ref = base_env.currpos.copy()
    intervention._left["axes"]["x"] = 1.0

    clock = {"now": 0.0}

    def monotonic():
        return clock["now"]

    def sleep(duration):
        clock["now"] += max(float(duration), 0.0)

    captured = {}

    def sample_step(action):
        captured["action"] = np.array(action, copy=True)
        return None

    with mock.patch.object(openarm_wrappers.time, "monotonic", side_effect=monotonic), mock.patch.object(
        openarm_wrappers.time, "sleep", side_effect=sleep
    ), mock.patch.object(intervention, "_poll_devices_once"), mock.patch.object(
        intervention, "_update_servo_target"
    ) as send_target, mock.patch.object(
        intervention, "_sample_env_like_step", side_effect=sample_step
    ):
        result = intervention._run_intervention_window(window_start=0.0)

    assert result is None
    assert send_target.call_count == 4
    np.testing.assert_allclose(captured["action"][:3], [1.0, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(intervention._target_pose_ref[0, 0], 0.0025, atol=1e-7)
    intervention.close()


def test_intervention_consumes_relative_axes_but_preserves_absolute_axes():
    base_env = OpenArmEnv(env_mode="virtual", hz=20, config=NoCameraConfig())
    with mock.patch.object(openarm_wrappers, "InputDevice", None), mock.patch.object(
        openarm_wrappers, "ecodes", None
    ):
        intervention = openarm_wrappers.DualSpacemouseIntervention(base_env)
    intervention._left["axes"]["x"] = 12.0
    intervention._left["axes"]["y"] = 34.0
    intervention._left["relative_axes"].add("x")

    intervention._consume_relative_axes()

    assert intervention._left["axes"]["x"] == 0.0
    assert intervention._left["axes"]["y"] == 34.0
    intervention.close()


def test_local_rotation_action_matches_world_frame_rotvec_integration():
    from openarm_env.envs.openarm_env import integrate_pose_velocity

    current_rotation = openarm_wrappers.R.from_euler("z", np.pi / 2)
    pose = np.tile(
        np.concatenate(([0.0, 0.0, 0.2], current_rotation.as_quat())),
        (2, 1),
    )
    transform = construct_twist_rotation_matrix(pose[0])
    local_action = np.zeros(14, dtype=np.float32)
    local_action[3] = 0.4
    world_action = local_action.copy()
    world_action[:6] = transform @ local_action[:6]

    updated = integrate_pose_velocity(
        pose,
        world_action,
        np.array([0.05, 0.25]),
        dt=0.2,
    )
    expected = current_rotation * openarm_wrappers.R.from_rotvec([0.4 * 0.05, 0.0, 0.0])
    actual = openarm_wrappers.R.from_quat(updated[0, 3:])

    np.testing.assert_allclose((actual * expected.inv()).as_rotvec(), 0.0, atol=1e-7)


def test_velocity_action_integrates_to_same_pose_at_20_and_80_hz():
    from openarm_env.envs.openarm_env import integrate_pose_velocity

    initial = np.tile(
        np.array([0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        (2, 1),
    )
    action = np.array(
        [0.4, -0.2, 0.1, 0.2, -0.1, 0.3, -1.0] * 2,
        dtype=np.float32,
    )
    velocity_scale = np.array([0.05, 0.25], dtype=np.float32)

    pose_20hz = initial.copy()
    for _ in range(20):
        pose_20hz = integrate_pose_velocity(pose_20hz, action, velocity_scale, dt=1.0 / 20.0)
    pose_80hz = initial.copy()
    for _ in range(80):
        pose_80hz = integrate_pose_velocity(pose_80hz, action, velocity_scale, dt=1.0 / 80.0)

    np.testing.assert_allclose(pose_20hz[:, :3], pose_80hz[:, :3], atol=1e-6)
    for arm_idx in (0, 1):
        rot_20hz = openarm_wrappers.R.from_quat(pose_20hz[arm_idx, 3:])
        rot_80hz = openarm_wrappers.R.from_quat(pose_80hz[arm_idx, 3:])
        np.testing.assert_allclose(
            (rot_20hz * rot_80hz.inv()).as_rotvec(),
            0.0,
            atol=1e-6,
        )


class _IdleEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 255, shape=(4,), dtype=np.uint8)
    action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)

    def step(self, action):
        return np.zeros(4, dtype=np.uint8), 0.0, False, False, {"intervention_idle": True}

    def reset(self, **kwargs):
        return np.zeros(4, dtype=np.uint8), {}


def test_classifier_skips_intervention_idle_frames():
    classifier = mock.Mock(return_value=1)
    env = MultiCameraBinaryRewardClassifierWrapper(_IdleEnv(), classifier)
    _, reward, terminated, _, info = env.step(np.zeros(1, dtype=np.float32))
    assert reward == 0
    assert terminated is False
    assert info["succeed"] is False
    classifier.assert_not_called()


def test_policy_adapter_replays_the_installed_wrapper_chain():
    base_env = OpenArmEnv(env_mode="virtual", hz=1000, config=NoCameraConfig())
    env = openarm_wrappers.DualRelativeFrame(base_env)
    env = openarm_wrappers.Quat2EulerWrapper(env)
    env = SERLObsWrapper(env, proprio_keys=["tcp_pose", "tcp_vel", "gripper_pose"])
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    env.reset()

    adapter = openarm_wrappers.OpenArmPolicyObsAdapter(env)
    adapted = adapter(base_env.refresh_obs())
    assert env.observation_space.contains(adapted)
    env.close()


def test_openarm_experiment_builds_complete_virtual_wrapper_stack():
    from experiments.openarm_pickplace.config import TrainConfig

    env = TrainConfig().get_environment(env_mode="virtual", classifier=False)
    obs, _ = env.reset()
    assert env.unwrapped.hz == 20.0
    assert env.unwrapped.max_episode_length == 1600
    assert env.observation_space.contains(obs)
    next_obs, reward, terminated, truncated, _ = env.step(
        np.zeros(env.action_space.shape, dtype=np.float32)
    )
    assert env.observation_space.contains(next_obs)
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    env.close()


class _FakeResponse:
    def __init__(self, payload=None, status_code=200):
        self._payload = payload or {"ok": True}
        self.status_code = status_code
        self.text = str(self._payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.text)


class _FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append((url.rsplit("/", 1)[-1], json, timeout))
        route = self.calls[-1][0]
        if route == "state":
            pose = [[0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]] * 2
            return _FakeResponse({"pose": pose, "gripper_closed": [False, True], "timestamp": 1.0})
        if route == "start":
            return _FakeResponse({"ok": True, "backend": "analytic"})
        return _FakeResponse()

    def close(self):
        pass


def test_real_env_owns_one_servo_session_and_uses_boolean_gripper_contract():
    session = _FakeSession()
    with mock.patch("requests.Session", return_value=session), mock.patch(
        "openarm_env.envs.openarm_env.time.sleep"
    ):
        env = OpenArmEnv(env_mode="real", hz=1000, config=NoCameraConfig())
        env.reset()
        start_payload = next(payload for route, payload, _ in session.calls if route == "start")
        assert set(start_payload) == {"arr", "gripper_closed"}
        assert start_payload["gripper_closed"] == [False, False]
        assert env.control_backend == "analytic"

        action = np.zeros(14, dtype=np.float32)
        action[6] = 1.0
        action[13] = -1.0
        env.step(action)
        target_payload = [payload for route, payload, _ in session.calls if route == "target"][-1]
        assert target_payload["gripper_closed"] == [True, False]

        env.reset()
        routes = [route for route, _, _ in session.calls]
        assert routes.count("start") == 2
        assert routes.count("stop") == 1
        env.close()
        routes = [route for route, _, _ in session.calls]
        assert routes.count("stop") == 2


def test_real_env_accumulates_velocity_on_target_reference_not_stale_feedback():
    session = _FakeSession()
    with mock.patch("requests.Session", return_value=session), mock.patch(
        "openarm_env.envs.openarm_env.time.sleep"
    ):
        env = OpenArmEnv(env_mode="real", hz=20, config=NoCameraConfig())
        env.reset()
        action = np.zeros(14, dtype=np.float32)
        action[0] = 1.0
        env.step(action)
        env.step(action)

        targets = [payload["arr"] for route, payload, _ in session.calls if route == "target"]
        assert len(targets) >= 2
        np.testing.assert_allclose(targets[-2][0][0], 0.0025, atol=1e-7)
        np.testing.assert_allclose(targets[-1][0][0], 0.0050, atol=1e-7)
        env.close()
