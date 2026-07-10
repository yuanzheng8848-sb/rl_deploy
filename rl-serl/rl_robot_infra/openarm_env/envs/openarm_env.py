"""Gymnasium interface for the bimanual OpenArm environment."""

import base64
import time
from typing import Dict, Literal

import cv2
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
    REALSENSE_CAMERAS: Dict[str, str] = {}

    TARGET_POSE: np.ndarray = np.zeros((6,))
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))

    ACTION_SCALE: np.ndarray = np.array([0.01, 0.05, 1.0])
    RESET_POSE: np.ndarray = np.zeros((6,))

    ABS_POSE_LIMIT_HIGH: np.ndarray = np.array([0.5, 0.5, 0.8, 3.14, 3.14, 3.14])
    ABS_POSE_LIMIT_LOW: np.ndarray = np.array([-0.5, -0.5, 0.0, -3.14, -3.14, -3.14])

    GRIPPER_OPEN_THRESHOLD: float = -0.3
    GRIPPER_CLOSE_THRESHOLD: float = 0.3
    SAFE_GRIPPER_OPEN_CMD: float = -1.0
    SAFE_GRIPPER_CLOSE_CMD: float = 0.05


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


def binary_gripper_state_to_cmd(binary_state: int, open_cmd: float, close_cmd: float) -> float:
    return float(close_cmd) if int(binary_state) == 1 else float(open_cmd)


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
        hz=5,
        env_mode: EnvMode = "real",
        config: DefaultOpenArmConfig = None,
        max_episode_length=100,
    ):
        if env_mode not in ("real", "virtual"):
            raise ValueError(f"env_mode must be 'real' or 'virtual', got {env_mode!r}")

        self.hz = hz
        self.env_mode = env_mode
        self.is_virtual = env_mode == "virtual"
        self.config = config or DefaultOpenArmConfig()
        self.max_episode_length = int(max_episode_length)
        self.arm = "both"

        self.session = None if self.is_virtual else requests.Session()
        self.url = self.config.SERVER_URL

        self.action_scale = self.config.ACTION_SCALE

        self.xyz_bounding_box = gym.spaces.Box(
            self.config.ABS_POSE_LIMIT_LOW[:3],
            self.config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )

        single_arm_quat = euler_2_quat(self.config.RESET_POSE[3:])
        single_arm_reset = np.concatenate([self.config.RESET_POSE[:3], single_arm_quat])
        self.resetpos = np.vstack([single_arm_reset, single_arm_reset]).astype(np.float32)

        self.currpos = self.resetpos.copy()
        self.currvel = np.zeros((2, 6), dtype=np.float32)
        self._last_tcp_pose_for_vel = None
        self._last_tcp_pose_time = None
        self.state_stale = False

        self.q = np.zeros((2, 7), dtype=np.float32)
        self.dq = np.zeros((2, 7), dtype=np.float32)
        self.curr_gripper_pos = np.zeros((2,), dtype=np.float32)
        self.gripper_binary_state = np.zeros((2,), dtype=int)
        self.gripper_open_threshold, self.gripper_close_threshold = get_gripper_thresholds(
            self.config
        )
        self.safe_gripper_open_cmd = float(getattr(self.config, "SAFE_GRIPPER_OPEN_CMD", -1.0))
        self.safe_gripper_close_cmd = float(getattr(self.config, "SAFE_GRIPPER_CLOSE_CMD", 0.05))

        self.curr_path_length = 0
        self.cycle_count = 0
        self.latest_images = {}

        tcp_shape = (2, 7)
        gripper_shape = (2, 1)
        action_dim = 14
        img_spaces = {
            name: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)
            for name in self.config.REALSENSE_CAMERAS.keys()
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

        if self.config.REALSENSE_CAMERAS:
            self.init_cameras(self.config.REALSENSE_CAMERAS)

        if self.is_virtual:
            print(f"Initialized OpenArm Env (virtual, offline) - Arm: {self.arm}")
        else:
            print(f"Initialized OpenArm Env (real) connected to {self.url} - Arm: {self.arm}")

    def sync_binary_gripper_state_from_position(self):
        current = np.asarray(self.curr_gripper_pos, dtype=np.float32).reshape(-1)
        if current.size < 2:
            return
        updated = np.zeros((2,), dtype=np.int32)
        for arm_idx in range(2):
            pos = float(current[arm_idx])
            dist_open = abs(pos - self.safe_gripper_open_cmd)
            dist_close = abs(pos - self.safe_gripper_close_cmd)
            updated[arm_idx] = 1 if dist_close <= dist_open else 0
        self.gripper_binary_state = updated

    def _apply_gripper_action(self, raw_val: float, arm_idx: int) -> float:
        self.gripper_binary_state[arm_idx] = apply_binary_gripper_logic(
            raw_val=raw_val,
            prev_binary_state=self.gripper_binary_state[arm_idx],
            open_threshold=self.gripper_open_threshold,
            close_threshold=self.gripper_close_threshold,
        )
        return binary_gripper_state_to_cmd(
            self.gripper_binary_state[arm_idx],
            self.safe_gripper_open_cmd,
            self.safe_gripper_close_cmd,
        )

    def gripper_actions_to_commands(self, action: np.ndarray) -> list:
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
        start_time = time.time()
        action = np.clip(
            np.asarray(action, dtype=np.float32),
            self.action_space.low,
            self.action_space.high,
        )

        nextpos = self.currpos.copy()
        for arm_idx in (0, 1):
            idx_start = arm_idx * 7
            current_arm_action = action[idx_start : idx_start + 7]
            nextpos[arm_idx, :3] += current_arm_action[:3] * self.action_scale[0]
            rpy_delta = current_arm_action[3:6] * self.action_scale[1]
            rot_delta = Rotation.from_euler("xyz", rpy_delta)
            rot_curr = Rotation.from_quat(nextpos[arm_idx, 3:])
            nextpos[arm_idx, 3:] = (rot_delta * rot_curr).as_quat()

        target_pos = self.clip_safety_box(nextpos)
        final_gripper_cmds = self.gripper_actions_to_commands(action)
        if self.is_virtual:
            self._apply_virtual_target(target_pos, gripper_pos=final_gripper_cmds)
        else:
            self._send_pos_command(target_pos, gripper_pos=final_gripper_cmds)

        self.curr_path_length += 1
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        if not self.is_virtual:
            self._update_currpos()

        obs = self._get_obs()
        reward = self.compute_reward(obs, False)
        terminated = bool(reward == 1.0)
        truncated = bool(self.curr_path_length >= self.max_episode_length)
        info = {"state_stale": bool(self.state_stale)}
        return obs, reward, terminated, truncated, info

    def compute_reward(self, obs: Dict, gripper_effective: bool) -> float:
        return 0.0

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
            self.curr_gripper_pos = np.array(
                [self.safe_gripper_open_cmd, self.safe_gripper_open_cmd],
                dtype=np.float32,
            )
        else:
            self.go_to_rest()

        self.currvel = np.zeros((2, 6), dtype=np.float32)
        self._last_tcp_pose_for_vel = self.currpos.copy()
        self._last_tcp_pose_time = time.time()
        self.gripper_binary_state = np.zeros((2,), dtype=int)

        self.sync_binary_gripper_state_from_position()
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

    def _send_pos_command(self, pos: np.ndarray, gripper_pos: list = None):
        if self.is_virtual:
            raise RuntimeError("virtual OpenArmEnv must not send robot commands")
        data = {"arr": np.asarray(pos, dtype=np.float32).tolist()}
        if gripper_pos is not None:
            data["gripper"] = [float(x) for x in gripper_pos]
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

    def _apply_virtual_target(self, pos: np.ndarray, gripper_pos: list = None):
        target = np.asarray(pos, dtype=np.float32).reshape(2, 7)
        self._update_tcp_velocity(target)
        self.currpos[:] = target
        if gripper_pos is not None:
            self.curr_gripper_pos = np.asarray(gripper_pos, dtype=np.float32).reshape(2)
            self.sync_binary_gripper_state_from_position()
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

            if "q" in ps:
                self.q[:] = ensure_shape(ps["q"], 7)
            self.dq[:] = ensure_shape(ps.get("dq", [0] * 14), 7)
            if "gripper_pos" in ps:
                self.curr_gripper_pos = np.asarray(ps["gripper_pos"], dtype=np.float32)
                self.sync_binary_gripper_state_from_position()

            if "images" in ps:
                for name, b64_str in ps["images"].items():
                    try:
                        img_bytes = base64.b64decode(b64_str)
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        img = cv2.resize(img, (128, 128))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        key = None
                        if name == "head":
                            key = "image_primary"
                        elif name == "left":
                            key = "image_left"
                        elif name == "right":
                            key = "image_right"
                        if key:
                            self.latest_images[key] = img
                    except Exception as exc:
                        print(f"[Env Warning] Failed to decode image {name}: {exc}")
            self.state_stale = False
        except Exception as exc:
            self.state_stale = True
            print(f"[Env Error] Update state failed: {exc}")

    def init_cameras(self, cameras):
        pass

    def close(self):
        if self.session is not None:
            self.session.close()
