"""Gymnasium interface for the bimanual OpenArm environment."""

import time
from typing import Dict, Literal

import gymnasium as gym
import numpy as np
import requests
from scipy.spatial.transform import Rotation


EnvMode = Literal["real", "virtual"]


def euler_2_quat(euler: np.ndarray) -> np.ndarray:
    return Rotation.from_euler("xyz", euler).as_quat()


def quat_2_euler(quat: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(quat).as_euler("xyz")


class DefaultOpenArmConfig:
    """Default OpenArm environment configuration."""

    SERVER_URL: str = "http://127.0.0.1:5000/"
    CAMERAS: Dict[str, dict] = {}

    # Maximum Cartesian velocities for a normalized action of magnitude 1.
    # Units: metres/second and radians/second.
    ACTION_VELOCITY_SCALE: np.ndarray = np.array([0.05, 0.25])
    VIRTUAL_RESET_POSE: np.ndarray = np.zeros((6,))

    ABS_POSE_LIMIT_HIGH: np.ndarray = np.array([0.5, 0.5, 0.8, 3.14, 3.14, 3.14])
    ABS_POSE_LIMIT_LOW: np.ndarray = np.array([-0.5, -0.5, 0.0, -3.14, -3.14, -3.14])

    GRIPPER_OPEN_THRESHOLD: float = -0.3
    GRIPPER_CLOSE_THRESHOLD: float = 0.3

def apply_binary_gripper_logic(
    raw_val: float,
    prev_binary_state: int,
    open_threshold: float,
    close_threshold: float,
):
    next_binary_state = int(prev_binary_state)
    if raw_val >= close_threshold:
        next_binary_state = 1
    elif raw_val <= open_threshold:
        next_binary_state = 0
    return next_binary_state


def integrate_pose_velocity(
    pose: np.ndarray,
    action: np.ndarray,
    velocity_scale: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Integrate a normalized 14-D Cartesian velocity command for ``dt`` seconds."""
    updated = np.asarray(pose, dtype=np.float32).reshape(2, 7).copy()
    action = np.asarray(action, dtype=np.float32).reshape(14)
    scale = np.asarray(velocity_scale, dtype=np.float32).reshape(2)
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt!r}")
    for arm_idx in (0, 1):
        arm_action = action[arm_idx * 7 : (arm_idx + 1) * 7]
        updated[arm_idx, :3] += arm_action[:3] * scale[0] * dt
        # Rotational actions are angular-vector components, matching tcp_vel.
        # Integrate them with the SO(3) exponential map instead of interpreting
        # them as sequential XYZ Euler rotations.
        rot_delta = Rotation.from_rotvec(arm_action[3:6] * scale[1] * dt)
        rot_curr = Rotation.from_quat(updated[arm_idx, 3:])
        updated[arm_idx, 3:] = (rot_delta * rot_curr).as_quat()
    return updated


def get_gripper_thresholds(config_or_env):
    if hasattr(config_or_env, "gripper_open_threshold") and hasattr(
        config_or_env, "gripper_close_threshold"
    ):
        return float(config_or_env.gripper_open_threshold), float(
            config_or_env.gripper_close_threshold
        )
    source = config_or_env.config if hasattr(config_or_env, "config") else config_or_env
    if not hasattr(source, "GRIPPER_OPEN_THRESHOLD") or not hasattr(
        source, "GRIPPER_CLOSE_THRESHOLD"
    ):
        raise AttributeError("config_or_env must provide gripper thresholds via env fields or config")
    return float(source.GRIPPER_OPEN_THRESHOLD), float(source.GRIPPER_CLOSE_THRESHOLD)


class OpenArmEnv(gym.Env):
    def __init__(
        self,
        hz=20,
        env_mode: EnvMode = "real",
        config: DefaultOpenArmConfig = None,
        max_episode_length=100,
    ):
        if env_mode not in ("real", "virtual"):
            raise ValueError(f"env_mode must be 'real' or 'virtual', got {env_mode!r}")

        self.hz = float(hz)
        if not np.isfinite(self.hz) or self.hz <= 0.0:
            raise ValueError(f"hz must be finite and positive, got {hz!r}")
        self.control_dt = 1.0 / self.hz
        self.env_mode = env_mode
        self.is_virtual = env_mode == "virtual"
        self.config = config or DefaultOpenArmConfig()
        self.max_episode_length = int(max_episode_length)
        self.arm = "both"

        self.session = None if self.is_virtual else requests.Session()
        self.url = self.config.SERVER_URL

        self.action_velocity_scale = np.asarray(
            self.config.ACTION_VELOCITY_SCALE, dtype=np.float32
        ).reshape(2)

        self.xyz_bounding_box = gym.spaces.Box(
            self.config.ABS_POSE_LIMIT_LOW[:3],
            self.config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )

        virtual_reset_pose = np.asarray(self.config.VIRTUAL_RESET_POSE, dtype=np.float32)
        single_arm_quat = euler_2_quat(virtual_reset_pose[3:])
        single_arm_reset = np.concatenate([virtual_reset_pose[:3], single_arm_quat])
        self.resetpos = np.vstack([single_arm_reset, single_arm_reset]).astype(np.float32)

        self.currpos = self.resetpos.copy()
        self.target_pose_ref = self.currpos.copy()
        self.currvel = np.zeros((2, 6), dtype=np.float32)
        self._last_tcp_pose_for_vel = None
        self._last_tcp_pose_time = None
        self.state_stale = False

        self.gripper_binary_state = np.zeros((2,), dtype=int)
        self.gripper_open_threshold, self.gripper_close_threshold = get_gripper_thresholds(
            self.config
        )
        self.servo_running = False
        self.control_backend = None

        self.curr_path_length = 0
        self.cycle_count = 0
        self.latest_images = {}
        self._next_step_deadline = None

        tcp_shape = (2, 7)
        gripper_shape = (2, 1)
        action_dim = 14
        img_spaces = {
            name: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)
            for name in self.config.CAMERAS.keys()
        }
        self.observation_space = gym.spaces.Dict(
            {
                "images": gym.spaces.Dict(img_spaces),
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=tcp_shape, dtype=np.float32
                        ),
                        "tcp_vel": gym.spaces.Box(
                            -np.inf, np.inf, shape=(2, 6), dtype=np.float32
                        ),
                        "gripper_pose": gym.spaces.Box(
                            -1, 1, shape=gripper_shape, dtype=np.float32
                        ),
                    }
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            -np.ones((action_dim,), dtype=np.float32),
            np.ones((action_dim,), dtype=np.float32),
            dtype=np.float32,
        )

        if self.config.CAMERAS:
            self.init_cameras(self.config.CAMERAS)

        if self.is_virtual:
            print(f"Initialized OpenArm Env (virtual, offline) - Arm: {self.arm}")
        else:
            print(f"Initialized OpenArm Env (real) connected to {self.url} - Arm: {self.arm}")

    def _apply_gripper_action(self, raw_val: float, arm_idx: int) -> bool:
        self.gripper_binary_state[arm_idx] = apply_binary_gripper_logic(
            raw_val=raw_val,
            prev_binary_state=self.gripper_binary_state[arm_idx],
            open_threshold=self.gripper_open_threshold,
            close_threshold=self.gripper_close_threshold,
        )
        return bool(self.gripper_binary_state[arm_idx])

    def gripper_actions_to_closed(self, action: np.ndarray) -> list:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size < 14:
            raise ValueError(f"Expected 14-dim action, got shape {action.shape}")
        return [
            self._apply_gripper_action(float(action[6]), 0),
            self._apply_gripper_action(float(action[13]), 1),
        ]

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        clipped_pose = np.asarray(pose, dtype=np.float32).copy()
        clipped_pose[:, :3] = np.clip(
            clipped_pose[:, :3],
            self.xyz_bounding_box.low,
            self.xyz_bounding_box.high,
        )
        return clipped_pose

    def step(self, action: np.ndarray) -> tuple:
        now = time.monotonic()
        if self._next_step_deadline is None:
            step_deadline = now + self.control_dt
        else:
            step_deadline = self._next_step_deadline
            while step_deadline <= now:
                step_deadline += self.control_dt
        action = np.clip(
            np.asarray(action, dtype=np.float32),
            self.action_space.low,
            self.action_space.high,
        )

        target_pos = self.clip_safety_box(
            integrate_pose_velocity(
                self.target_pose_ref,
                action,
                self.action_velocity_scale,
                self.control_dt,
            )
        )
        self.target_pose_ref = target_pos
        gripper_closed = self.gripper_actions_to_closed(action)
        if self.is_virtual:
            self._apply_virtual_target(target_pos, gripper_closed=gripper_closed)
        else:
            self._send_pos_command(target_pos, gripper_closed=gripper_closed)

        self.curr_path_length += 1
        time.sleep(max(0.0, step_deadline - time.monotonic()))
        self._next_step_deadline = step_deadline + self.control_dt

        if not self.is_virtual:
            self._update_currpos()

        obs = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = bool(self.curr_path_length >= self.max_episode_length)
        info = {"state_stale": bool(self.state_stale)}
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> dict:
        images = {}
        for key, space in self.observation_space["images"].spaces.items():
            images[key] = np.asarray(
                self.latest_images.get(key, np.zeros(space.shape, dtype=np.uint8)),
                dtype=np.uint8,
            )

        gripper_obs = np.where(self.gripper_binary_state[:, None] == 1, 1.0, -1.0).astype(
            np.float32
        )
        state_observation = {
            "tcp_pose": self.currpos.astype(np.float32, copy=True),
            "tcp_vel": self.currvel.astype(np.float32, copy=True),
            "gripper_pose": gripper_obs,
        }
        return {"images": images, "state": state_observation}

    def render(self, mode="human"):
        return None

    def reset(self, **kwargs):
        self.cycle_count += 1
        self.curr_path_length = 0
        self.state_stale = False

        if self.is_virtual:
            self.currpos = self.resetpos.copy()
        else:
            self.stop_control()
            self.go_to_rest()

        self.currvel = np.zeros((2, 6), dtype=np.float32)
        self._last_tcp_pose_for_vel = self.currpos.copy()
        self._last_tcp_pose_time = time.time()
        self.target_pose_ref = self.currpos.copy()
        self._next_step_deadline = None
        self.gripper_binary_state = np.zeros((2,), dtype=int)
        if not self.is_virtual:
            self.start_control()

        self.initial_reset_pose = self.currpos.copy()
        return self._get_obs(), {}

    def go_to_rest(self):
        if self.is_virtual:
            self.currpos = self.resetpos.copy()
            self.currvel[:] = 0.0
            return
        try:
            self.session.post(self.url + "control/home", json={"duration": 3.0}, timeout=6)
            time.sleep(1)
        except requests.exceptions.RequestException as exc:
            print(f"[Env Warning] Joint reset failed: {exc}")
        self._update_currpos()

    def start_control(self):
        if self.is_virtual or self.servo_running:
            return
        payload = {
            "arr": np.asarray(self.currpos, dtype=np.float32).tolist(),
            "gripper_closed": self.gripper_binary_state.astype(bool).tolist(),
        }
        resp = self.session.post(self.url + "control/start", json=payload, timeout=3.0)
        resp.raise_for_status()
        self.control_backend = resp.json()["backend"]
        self.servo_running = True

    def stop_control(self):
        if self.is_virtual or not self.servo_running:
            return
        try:
            self.session.post(self.url + "control/stop", json={}, timeout=2.0).raise_for_status()
        finally:
            self.servo_running = False
            self.control_backend = None

    def _send_pos_command(self, pos: np.ndarray, gripper_closed: list = None):
        if self.is_virtual:
            raise RuntimeError("virtual OpenArmEnv must not send robot commands")
        data = {"arr": np.asarray(pos, dtype=np.float32).tolist()}
        if gripper_closed is not None:
            data["gripper_closed"] = [bool(x) for x in gripper_closed]
        try:
            resp = self.session.post(self.url + "control/target", json=data, timeout=2.0)
            if resp.status_code != 200:
                print(f"[Env Error] Control target failed: {resp.text}")
        except requests.exceptions.RequestException as exc:
            print(f"[Env Error] Control target request failed: {exc}")

    def refresh_obs(self) -> dict:
        if not self.is_virtual:
            self._update_currpos()
        return self._get_obs()

    def _update_tcp_velocity(self, new_pose):
        now = time.time()
        new_pose = np.asarray(new_pose, dtype=np.float32).reshape(2, 7)
        if self._last_tcp_pose_for_vel is None or self._last_tcp_pose_time is None:
            self.currvel[:] = 0.0
        else:
            dt = now - self._last_tcp_pose_time
            if dt > 1e-6:
                prev_pose = np.asarray(self._last_tcp_pose_for_vel, dtype=np.float32).reshape(2, 7)
                vel = np.zeros((2, 6), dtype=np.float32)
                vel[:, :3] = (new_pose[:, :3] - prev_pose[:, :3]) / dt
                for arm_idx in range(2):
                    prev_rot = Rotation.from_quat(prev_pose[arm_idx, 3:])
                    curr_rot = Rotation.from_quat(new_pose[arm_idx, 3:])
                    vel[arm_idx, 3:] = (curr_rot * prev_rot.inv()).as_rotvec() / dt
                self.currvel[:] = vel
        self._last_tcp_pose_for_vel = new_pose.copy()
        self._last_tcp_pose_time = now

    def _apply_virtual_target(self, pos: np.ndarray, gripper_closed: list = None):
        target = np.asarray(pos, dtype=np.float32).reshape(2, 7)
        self._update_tcp_velocity(target)
        self.currpos[:] = target
        if gripper_closed is not None:
            self.gripper_binary_state = np.asarray(gripper_closed, dtype=np.int32).reshape(2)
        self.state_stale = False

    def _update_currpos(self):
        if self.is_virtual:
            self.state_stale = False
            return
        try:
            resp = self.session.post(self.url + "state", timeout=3.0)
            ps = resp.json()

            def ensure_shape(arr_list, cols):
                arr = np.asarray(arr_list, dtype=np.float32)
                if arr.size == 2 * cols:
                    return arr.reshape(2, cols)
                return arr

            if "pose" in ps:
                new_pose = ensure_shape(ps["pose"], 7)
                self._update_tcp_velocity(new_pose)
                self.currpos[:] = new_pose

            if "gripper_closed" in ps:
                self.gripper_binary_state = np.asarray(
                    ps["gripper_closed"], dtype=np.int32
                ).reshape(2)
            self.state_stale = False
        except Exception as exc:
            self.state_stale = True
            print(f"[Env Error] Update state failed: {exc}")

    def init_cameras(self, cameras):
        raise NotImplementedError(
            "OpenArmEnv is state/control-only; use LocalOpenArmEnv for configured cameras"
        )

    def close(self):
        if self.session is not None:
            self.stop_control()
            self.session.close()
