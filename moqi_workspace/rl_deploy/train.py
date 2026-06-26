#!/usr/bin/env python3

import ctypes
import copy
import fcntl
import glob
import os
import pickle as pkl
import site
import sys
import time
from collections import deque
from pathlib import Path
from threading import Event, Lock, Thread

import cv2
import gymnasium as gym
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
from gymnasium.wrappers import RecordEpisodeStatistics
from scipy.spatial.transform import Rotation as R


# Configure JAX CUDA library paths before JAX does any GPU work.
nvidia_base = os.path.join(site.getsitepackages()[0], "nvidia")
for lib in (
    "cublas/lib",
    "cudnn/lib",
    "cufft/lib",
    "cusolver/lib",
    "cusparse/lib",
    "nccl/lib",
    "nvjitlink/lib",
):
    path = os.path.join(nvidia_base, lib)
    if os.path.exists(path):
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{path}:{current_ld}"

os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={nvidia_base}"

try:
    nvjitlink_path = os.path.join(nvidia_base, "nvjitlink/lib/libnvJitLink.so.12")
    cusparse_path = os.path.join(nvidia_base, "cusparse/lib/libcusparse.so.12")
    if os.path.exists(nvjitlink_path):
        ctypes.CDLL(nvjitlink_path)
    if os.path.exists(cusparse_path):
        ctypes.CDLL(cusparse_path)
except Exception as exc:
    print(f"[DEBUG] Failed to preload CUDA libraries: {exc}")

import jax
import jax.numpy as jnp

# Compatibility shims for newer JAX releases used with older HIL-SERL code.
if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree_util.tree_map
if not hasattr(jax, "tree_leaves"):
    jax.tree_leaves = jax.tree_util.tree_leaves

_orig_shaped_array_update = jax.core.ShapedArray.update


def _compat_shaped_array_update(self, *args, **kwargs):
    # `named_shape` was removed from newer JAX ShapedArray constructors, but
    # some upstream Flax/JAX interactions may still pass it around.
    kwargs.pop("named_shape", None)
    return _orig_shaped_array_update(self, *args, **kwargs)


jax.core.ShapedArray.update = _compat_shaped_array_update


ROOT_DIR = Path(__file__).resolve().parents[2]
PYROKI_DIR = Path(__file__).resolve().parent.parent / "pyroki"
HIL_SERL_LAUNCHER_DIR = ROOT_DIR / "hil-serl" / "serl_launcher"
if str(HIL_SERL_LAUNCHER_DIR) not in sys.path:
    sys.path.insert(0, str(HIL_SERL_LAUNCHER_DIR))
if str(PYROKI_DIR) not in sys.path:
    sys.path.insert(0, str(PYROKI_DIR))

try:
    from evdev import InputDevice, ecodes
except Exception:
    InputDevice = None
    ecodes = None

try:
    from realsense_camera import OpenCVCamera, RealsenseCamera
except Exception as exc:
    print(f"[WARN] Failed to import local camera modules: {exc}")
    RealsenseCamera = None
    OpenCVCamera = None

from openarm_env import (  # noqa: E402
    DefaultOpenArmConfig,
    OpenArmEnv,
    apply_binary_gripper_logic,
    get_gripper_thresholds,
)

from agentlace.data.data_store import QueuedDataStore  # noqa: E402
from agentlace.trainer import TrainerClient, TrainerServer  # noqa: E402
from franka_env.utils.transformations import (  # noqa: E402
    construct_adjoint_matrix,
    construct_homogeneous_matrix,
)
from serl_launcher.data.data_store import (  # noqa: E402
    MemoryEfficientReplayBufferDataStore,
)
from serl_launcher.utils.launcher import (  # noqa: E402
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.utils.timer_utils import Timer  # noqa: E402
from serl_launcher.utils.train_utils import concat_batches  # noqa: E402


HEAD_CAMERA_DEVICE = "/dev/v4l/by-id/usb-Global_Shutter_Camera_Global_Shutter_Camera_01.00.00-video-index0"
HEAD_CAMERA_WIDTH = 640
HEAD_CAMERA_HEIGHT = 480
MODEL_IMAGE_SIZE = (128, 128)
APPLY_HEAD_CAMERA_CROP = False
NETWORK_IMAGE_CROP_VERSION = "primary_center_crop_v1"
ARM_FOCUS_VERSION = "single_arm_focus_v1"
TRAINING_IMAGE_KEYS = ["image_primary", "image_left", "image_right"]
PROPRIO_KEYS = ["tcp_pose", "tcp_vel", "gripper_pose"]
DEFAULT_SUCCESS_DEMO_DIR = (
    Path(__file__).resolve().parent / "demo" / "collected" / "success"
)

class TrainOpenArmConfig(DefaultOpenArmConfig):
    REALSENSE_CAMERAS = {
        "image_primary": "local-head",
        "image_left": "local-left",
        "image_right": "local-right",
    }


devices = jax.local_devices()
num_devices = len(devices)
if hasattr(jax.sharding, "PositionalSharding"):
    sharding = jax.sharding.PositionalSharding(devices).replicate()
else:
    sharding = jax.sharding.SingleDeviceSharding(devices[0])


def print_green(text):
    print(f"\033[92m {text}\033[00m")


def print_yellow(text):
    print(f"\033[93m {text}\033[00m")


def get_flag_value(name, default):
    try:
        if hasattr(FLAGS, "is_parsed") and not FLAGS.is_parsed():
            return default
        return getattr(FLAGS, name)
    except Exception:
        return default


def stack_obs(obs):
    dict_list = {key: [item[key] for item in obs] for key in obs[0]}
    return jax.tree_util.tree_map(
        lambda values: np.stack(values),
        dict_list,
        is_leaf=lambda value: isinstance(value, list),
    )


def space_stack(space, repeat):
    if isinstance(space, gym.spaces.Box):
        return gym.spaces.Box(
            low=np.repeat(space.low[None], repeat, axis=0),
            high=np.repeat(space.high[None], repeat, axis=0),
            dtype=space.dtype,
        )
    if isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict(
            {key: space_stack(value, repeat) for key, value in space.spaces.items()}
        )
    raise TypeError(f"Unsupported space type: {type(space)}")


class ChunkingWrapper(gym.Wrapper):
    def __init__(self, env, obs_horizon=1, act_exec_horizon=None):
        super().__init__(env)
        self.obs_horizon = int(obs_horizon)
        self.act_exec_horizon = act_exec_horizon
        self.current_obs = deque(maxlen=self.obs_horizon)
        self.observation_space = space_stack(self.env.observation_space, self.obs_horizon)
        self.action_space = self.env.action_space

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.current_obs.append(obs)
        return stack_obs(self.current_obs), reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_obs.extend([obs] * self.obs_horizon)
        return stack_obs(self.current_obs), info


class LocalOpenArmEnv(OpenArmEnv):
    """
    OpenArm env variant that always reads images locally, even in fake mode.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.fake_env:
            print("[LocalOpenArmEnv] forcing local camera init in fake mode.")
            self.init_cameras(None)

    def init_cameras(self, _config):
        self.cameras = []
        self.latest_images_raw = {}

        try:
            from mock_hardware import MockCamera
        except ImportError:
            MockCamera = None

        try:
            if self.fake_env and MockCamera is not None:
                cam_left = MockCamera(width=640, height=480, fps=30)
            else:
                cam_left = RealsenseCamera(
                    device_id="150622074105",
                    enable_depth=False,
                    width=640,
                    height=480,
                    fps=30,
                )
            self.cameras.append(("image_left", cam_left))
            print("Initialized Left Camera (150622074105)")
        except Exception as exc:
            print(f"Failed to init left camera: {exc}")

        try:
            if self.fake_env and MockCamera is not None:
                cam_right = MockCamera(width=640, height=480, fps=30)
            else:
                cam_right = RealsenseCamera(
                    device_id="236422072385",
                    enable_depth=False,
                    width=640,
                    height=480,
                    fps=30,
                )
            self.cameras.append(("image_right", cam_right))
            print("Initialized Right Camera (236422072385)")
        except Exception as exc:
            print(f"Failed to init right camera: {exc}")

        try:
            if self.fake_env and MockCamera is not None:
                cam_head = MockCamera(
                    width=HEAD_CAMERA_WIDTH,
                    height=HEAD_CAMERA_HEIGHT,
                    fps=30,
                )
            else:
                cam_head = OpenCVCamera(
                    HEAD_CAMERA_DEVICE,
                    width=HEAD_CAMERA_WIDTH,
                    height=HEAD_CAMERA_HEIGHT,
                    fps=30,
                    exposure=150,
                )
            self.cameras.append(("image_primary", cam_head))
            print(f"Initialized Head Camera ({HEAD_CAMERA_DEVICE})")
        except Exception as exc:
            print(f"Failed to init head camera: {exc}")

        self.stop_event = Event()
        self.capture_thread = Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _capture_loop(self):
        while not self.stop_event.is_set():
            for name, cam in self.cameras:
                try:
                    frame = cam.get_data(viz=False)
                    is_realsense = isinstance(frame, (list, tuple))
                    if is_realsense:
                        frame = frame[0]
                    if frame is None:
                        continue

                    if name == "image_primary" and APPLY_HEAD_CAMERA_CROP:
                        h, w = frame.shape[:2]
                        crop_h_ratio = 0.35
                        crop_w_ratio = 0.35
                        cy, cx = h // 2, w // 2
                        half_h = int(h * crop_h_ratio / 2)
                        half_w = int(w * crop_w_ratio / 2)
                        frame = frame[cy - half_h : cy + half_h, cx - half_w : cx + half_w]

                    if is_realsense:
                        full_rgb = frame
                    else:
                        full_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.latest_images_raw[name] = full_rgb

                    resized = cv2.resize(frame, MODEL_IMAGE_SIZE)
                    if is_realsense:
                        rgb = resized
                    else:
                        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    self.latest_images[name] = rgb
                except Exception as exc:
                    print(f"[Capture Error:{name}] {exc}")
            time.sleep(0.01)

    def close(self):
        if hasattr(self, "stop_event"):
            self.stop_event.set()
        if hasattr(self, "capture_thread"):
            self.capture_thread.join(timeout=1.0)
        super().close()


class DualRelativeFrame(gym.Wrapper):
    """
    Minimal dual-arm relative-frame wrapper for OpenArm's packed state schema.
    """

    def __init__(self, env, include_relative_pose=True):
        super().__init__(env)
        self.include_relative_pose = bool(include_relative_pose)
        self.left_transform = np.eye(6)
        self.right_transform = np.eye(6)
        self.left_reset_inv = np.eye(4)
        self.right_reset_inv = np.eye(4)

    def _update_from_obs(self, obs):
        tcp_pose = np.asarray(obs["state"]["tcp_pose"])
        self.left_transform = construct_adjoint_matrix(tcp_pose[0])
        self.right_transform = construct_adjoint_matrix(tcp_pose[1])
        if self.include_relative_pose:
            self.left_reset_inv = np.linalg.inv(construct_homogeneous_matrix(tcp_pose[0]))
            self.right_reset_inv = np.linalg.inv(construct_homogeneous_matrix(tcp_pose[1]))

    def _transform_pose(self, pose, reset_inv):
        transform = reset_inv @ construct_homogeneous_matrix(pose)
        pos = transform[:3, 3]
        quat = R.from_matrix(transform[:3, :3]).as_quat()
        return np.concatenate((pos, quat), axis=0)

    def transform_observation(self, obs):
        obs = {
            "images": obs["images"],
            "state": {
                key: np.array(value, copy=True)
                for key, value in obs["state"].items()
            },
        }
        tcp_vel = obs["state"]["tcp_vel"]
        tcp_vel[0] = np.linalg.inv(self.left_transform) @ tcp_vel[0]
        tcp_vel[1] = np.linalg.inv(self.right_transform) @ tcp_vel[1]
        if self.include_relative_pose:
            tcp_pose = obs["state"]["tcp_pose"]
            tcp_pose[0] = self._transform_pose(tcp_pose[0], self.left_reset_inv)
            tcp_pose[1] = self._transform_pose(tcp_pose[1], self.right_reset_inv)
        return obs

    def transform_action(self, action):
        action = np.asarray(action, dtype=np.float32).copy()
        if action.shape[0] >= 14:
            action[:6] = self.left_transform @ action[:6]
            action[7:13] = self.right_transform @ action[7:13]
        return action

    def transform_action_inv(self, action):
        action = np.asarray(action, dtype=np.float32).copy()
        if action.shape[0] >= 14:
            action[:6] = np.linalg.inv(self.left_transform) @ action[:6]
            action[7:13] = np.linalg.inv(self.right_transform) @ action[7:13]
        return action

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._update_from_obs(obs)
        return self.transform_observation(obs), info

    def step(self, action):
        transformed_action = self.transform_action(action)
        obs, reward, done, truncated, info = self.env.step(transformed_action)
        if "intervene_action" in info:
            info["intervene_action"] = self.transform_action_inv(info["intervene_action"])
        tcp_pose = np.asarray(obs["state"]["tcp_pose"])
        self.left_transform = construct_adjoint_matrix(tcp_pose[0])
        self.right_transform = construct_adjoint_matrix(tcp_pose[1])
        return self.transform_observation(obs), reward, done, truncated, info


class Quat2EulerWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        state_space = self.observation_space["state"]
        state_space.spaces["tcp_pose"] = gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=(2, 6),
            dtype=np.float32,
        )

    def observation(self, obs):
        obs = {
            "images": obs["images"],
            "state": {key: np.array(value, copy=True) for key, value in obs["state"].items()},
        }
        tcp_pose = obs["state"]["tcp_pose"]
        euler_pose = np.zeros((2, 6), dtype=np.float32)
        for arm_idx in range(2):
            euler_pose[arm_idx, :3] = tcp_pose[arm_idx, :3]
            euler_pose[arm_idx, 3:] = R.from_quat(tcp_pose[arm_idx, 3:]).as_euler("xyz")
        obs["state"]["tcp_pose"] = euler_pose
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


def crop_rgb_image(img, crop_ratio, y_offset_ratio=0.0):
    img = np.asarray(img)
    if img.ndim < 3 or img.shape[-1] != 3:
        return img
    if img.ndim > 3:
        leading_shape = img.shape[:-3]
        flat = img.reshape((-1, *img.shape[-3:]))
        cropped = [
            crop_rgb_image(frame, crop_ratio, y_offset_ratio=y_offset_ratio)
            for frame in flat
        ]
        return np.stack(cropped, axis=0).reshape((*leading_shape, *MODEL_IMAGE_SIZE[::-1], 3))

    ratio = float(np.clip(crop_ratio, 0.05, 1.0))
    y_offset_ratio = float(np.clip(y_offset_ratio, -0.5, 0.5))
    h, w = img.shape[:2]
    crop_h = max(1, int(round(h * ratio)))
    crop_w = max(1, int(round(w * ratio)))
    center_y = (h / 2.0) + (h * y_offset_ratio)
    y0 = int(round(center_y - crop_h / 2.0))
    y0 = int(np.clip(y0, 0, max(0, h - crop_h)))
    x0 = max(0, (w - crop_w) // 2)
    cropped = img[y0 : y0 + crop_h, x0 : x0 + crop_w]
    return cv2.resize(cropped, MODEL_IMAGE_SIZE)


def crop_primary_image_for_network(
    obs,
    raw_source=None,
    crop_ratio=0.3,
    y_offset_ratio=0.0,
):
    if not isinstance(obs, dict) or "images" not in obs:
        return obs
    images = obs.get("images")
    if not isinstance(images, dict) or "image_primary" not in images:
        return obs

    obs = {
        "images": {key: np.array(value, copy=True) for key, value in images.items()},
        "state": obs["state"],
    }
    source = raw_source if raw_source is not None else obs["images"]["image_primary"]
    obs["images"]["image_primary"] = crop_rgb_image(
        source,
        crop_ratio,
        y_offset_ratio=y_offset_ratio,
    ).astype(np.uint8)
    return obs


def get_network_image_crop_info():
    return {
        "enabled": bool(get_flag_value("network_crop_primary", True)),
        "version": NETWORK_IMAGE_CROP_VERSION,
        "image_key": "image_primary",
        "crop_ratio": float(get_flag_value("network_primary_crop_ratio", 0.3)),
        "y_offset_ratio": float(get_flag_value("network_primary_crop_y_offset", 0.0)),
    }


def transition_has_network_crop(transition):
    infos = transition.get("infos") if isinstance(transition, dict) else None
    if not isinstance(infos, dict):
        return False
    crop_info = infos.get("network_image_crop")
    return bool(
        isinstance(crop_info, dict)
        and crop_info.get("version") == NETWORK_IMAGE_CROP_VERSION
    )


def mark_transition_network_crop(transition):
    if not isinstance(transition, dict):
        return transition
    infos = transition.get("infos")
    if not isinstance(infos, dict):
        infos = {}
        transition["infos"] = infos
    infos["network_image_crop"] = get_network_image_crop_info()
    return transition


def maybe_apply_network_image_crop_to_obs_dict(obs_dict):
    if not get_flag_value("network_crop_primary", True):
        return obs_dict
    if not isinstance(obs_dict, dict) or "image_primary" not in obs_dict:
        return obs_dict
    obs_dict = {key: np.array(value, copy=True) for key, value in obs_dict.items()}
    obs_dict["image_primary"] = crop_rgb_image(
        obs_dict["image_primary"],
        get_flag_value("network_primary_crop_ratio", 0.3),
        y_offset_ratio=get_flag_value("network_primary_crop_y_offset", 0.0),
    ).astype(np.uint8)
    return obs_dict


def maybe_apply_network_image_crop_to_transition(transition):
    if (
        not get_flag_value("network_crop_primary", True)
        or not isinstance(transition, dict)
        or transition_has_network_crop(transition)
    ):
        return transition

    for obs_key in ("observations", "next_observations"):
        obs_dict = transition.get(obs_key)
        transition[obs_key] = maybe_apply_network_image_crop_to_obs_dict(obs_dict)
    return mark_transition_network_crop(transition)


class NetworkPrimaryImageCropWrapper(gym.ObservationWrapper):
    """
    Applies center crop only to the policy/replay observation stream.

    Raw camera buffers and classifier inputs stay untouched.
    """

    def __init__(self, env, crop_ratio=0.3, y_offset_ratio=0.0):
        super().__init__(env)
        self.crop_ratio = float(crop_ratio)
        self.y_offset_ratio = float(y_offset_ratio)

    def observation(self, obs):
        raw_source = None
        base_env = self.env.unwrapped
        latest_raw = getattr(base_env, "latest_images_raw", None)
        if isinstance(latest_raw, dict):
            raw_source = latest_raw.get("image_primary")
        return crop_primary_image_for_network(
            obs,
            raw_source=raw_source,
            crop_ratio=self.crop_ratio,
            y_offset_ratio=self.y_offset_ratio,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


def get_train_arm_mode():
    """Training is always bimanual."""
    return "both"


def arm_focus_enabled():
    """Arm focus is never used; training is always bimanual."""
    return False


def arm_focus_allows_arm(label):
    """All arms are always active in bimanual training."""
    return True


def get_inactive_camera_key_for_arm_focus():
    """No inactive camera in bimanual training."""
    return None


def _gripper_label_to_value(label):
    """Convert gripper label string to action value."""
    if label is None:
        return None
    label = str(label).strip().lower()
    if label == "open":
        return -1.0
    if label == "close":
        return 1.0
    if label in ("none", "free", ""):
        return None
    raise ValueError(f"gripper label must be open/close/none, got: {label!r}")


def _resolve_gripper_holds():
    """Grippers are always policy-controlled (no forced holds)."""
    return None, None


def mask_action_for_arm_focus(action):
    if not arm_focus_enabled():
        return action
    arr = np.asarray(action, dtype=np.float32).copy()
    if arr.shape[-1] < 14:
        return arr
    mode = get_train_arm_mode()
    active_val, inactive_val = _resolve_gripper_holds()
    if mode == "left":
        arr[..., 7:14] = 0.0
        if inactive_val is not None:
            arr[..., 13] = float(inactive_val)
        if active_val is not None:
            arr[..., 6] = float(active_val)
    elif mode == "right":
        arr[..., 0:7] = 0.0
        if inactive_val is not None:
            arr[..., 6] = float(inactive_val)
        if active_val is not None:
            arr[..., 13] = float(active_val)
    return arr


def black_inactive_camera_for_arm_focus(obs):
    inactive_key = get_inactive_camera_key_for_arm_focus()
    if inactive_key is None or not isinstance(obs, dict):
        return obs

    if "images" in obs and isinstance(obs["images"], dict):
        images = obs["images"]
        if inactive_key not in images:
            return obs
        obs = {
            "images": {key: np.array(value, copy=True) for key, value in images.items()},
            "state": obs["state"],
        }
        obs["images"][inactive_key] = np.zeros_like(obs["images"][inactive_key])
        return obs

    if inactive_key not in obs:
        return obs
    obs = {key: np.array(value, copy=True) for key, value in obs.items()}
    obs[inactive_key] = np.zeros_like(obs[inactive_key])
    return obs


def get_arm_focus_info():
    return {
        "enabled": arm_focus_enabled(),
        "version": ARM_FOCUS_VERSION,
        "train_arm": get_train_arm_mode(),
        "inactive_camera_key": get_inactive_camera_key_for_arm_focus(),
    }


def transition_has_arm_focus(transition):
    infos = transition.get("infos") if isinstance(transition, dict) else None
    if not isinstance(infos, dict):
        return False
    focus_info = infos.get("arm_focus")
    return bool(
        isinstance(focus_info, dict)
        and focus_info.get("version") == ARM_FOCUS_VERSION
        and focus_info.get("train_arm") == get_train_arm_mode()
    )


def mark_transition_arm_focus(transition):
    if not isinstance(transition, dict):
        return transition
    infos = transition.get("infos")
    if not isinstance(infos, dict):
        infos = {}
        transition["infos"] = infos
    infos["arm_focus"] = get_arm_focus_info()
    return transition


def maybe_apply_arm_focus_to_transition(transition):
    if (
        not arm_focus_enabled()
        or not isinstance(transition, dict)
        or transition_has_arm_focus(transition)
    ):
        return transition

    for obs_key in ("observations", "next_observations"):
        transition[obs_key] = black_inactive_camera_for_arm_focus(
            transition.get(obs_key)
        )
    if "actions" in transition:
        transition["actions"] = mask_action_for_arm_focus(transition["actions"])
    return mark_transition_arm_focus(transition)


class ArmFocusWrapper(gym.Wrapper):
    """
    Keeps the full bimanual interface but focuses learning/execution on one arm.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def _focus_obs(self, obs):
        return black_inactive_camera_for_arm_focus(obs)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._focus_obs(obs), info

    def step(self, action):
        focused_action = mask_action_for_arm_focus(action)
        obs, reward, done, truncated, info = self.env.step(focused_action)
        if not isinstance(info, dict):
            info = {}
        info["arm_focus"] = get_arm_focus_info()
        info["arm_focus_action"] = np.asarray(focused_action, dtype=np.float32)
        return self._focus_obs(obs), reward, done, truncated, info


class SERLObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, proprio_keys=None):
        super().__init__(env)
        self.proprio_keys = list(proprio_keys or PROPRIO_KEYS)
        self.proprio_space = gym.spaces.Dict(
            {key: self.env.observation_space["state"][key] for key in self.proprio_keys}
        )
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.flatten_space(self.proprio_space),
                **self.env.observation_space["images"].spaces,
            }
        )

    def observation(self, obs):
        proprio = {key: obs["state"][key] for key in self.proprio_keys}
        return {
            "state": gym.spaces.flatten(self.proprio_space, proprio),
            **obs["images"],
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


class KeyboardRewardWrapper(gym.Wrapper):
    """
    Keyboard-only reward wrapper.

    Reward is assigned purely by human keypress during teleop / training:
      <SPACE> marks the episode as a failure (reward 0, done).
      <ENTER> marks the episode as a success (reward 1, done).
      <H>     toggles handoff-focus tagging.

    No joint/EE target file is required.
    """

    def __init__(self, env, reward_image_key="image_primary"):
        super().__init__(env)
        self.reward_image_key = reward_image_key
        self.manual_fail_requested = False
        self.manual_success_requested = False
        self.handoff_focus_enabled = False
        self.render_enabled = bool(get_flag_value("render", False))
        self.render_window_name = "RL Deploy Monitor"

        self.listener_thread = Thread(target=self._keyboard_listener, daemon=True)
        self.listener_thread.start()
        controls = [
            "<SPACE> fail/reset",
            "<ENTER> success",
            "<H> toggle handoff focus tagging",
        ]
        print(f"[RewardWrapper] Keyboard controls: {', '.join(controls)}.")

    def _render_monitor(self, reward_value):
        if not self.render_enabled:
            return

        base_env = self.env.unwrapped
        latest_images = getattr(base_env, "latest_images", {})
        camera_keys = ["image_primary", "image_left", "image_right"]
        titles = {
            "image_primary": "primary",
            "image_left": "left",
            "image_right": "right",
        }

        panels = []
        for key in camera_keys:
            img = latest_images.get(key)
            if img is None:
                panel = np.zeros((MODEL_IMAGE_SIZE[1], MODEL_IMAGE_SIZE[0], 3), dtype=np.uint8)
            else:
                panel = np.array(img, copy=True)
                if panel.shape[:2] != (MODEL_IMAGE_SIZE[1], MODEL_IMAGE_SIZE[0]):
                    panel = cv2.resize(panel, MODEL_IMAGE_SIZE)
            panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
            cv2.putText(
                panel_bgr,
                titles[key],
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            panels.append(panel_bgr)

        row = np.hstack(panels)
        header = np.zeros((42, row.shape[1], 3), dtype=np.uint8)
        text = f"reward: {reward_value:.3f}"
        color = (0, 200, 0) if reward_value >= 0.8 else (0, 180, 255)
        cv2.putText(
            header,
            text,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        canvas = np.vstack([header, row])
        cv2.imshow(self.render_window_name, canvas)
        cv2.waitKey(1)

    def _evaluate_current_reward(self):
        reward = 0.0
        manual_done = False
        manual_success = False
        manual_failure = False

        if self.manual_fail_requested:
            self.manual_fail_requested = False
            self.manual_success_requested = False
            manual_done = True
            manual_failure = True
            print("[RewardWrapper] Episode marked as failure. Waiting actor loop to reset env.")
        elif self.manual_success_requested:
            self.manual_success_requested = False
            self.manual_fail_requested = False
            reward = 1.0
            manual_done = True
            manual_success = True
            print("[RewardWrapper] Episode marked as success. Waiting actor loop to reset env.")

        self._render_monitor(reward)

        success_flag = bool(manual_success)
        info = {
            "manual_termination": manual_done,
            "manual_success": manual_success,
            "manual_failure": manual_failure,
            "success": success_flag,
            "succeed": success_flag,
        }
        return np.asarray(reward, dtype=np.float32), manual_done, info

    def _keyboard_listener(self):
        from pynput import keyboard

        def on_press(key):
            try:
                if key == keyboard.Key.space:
                    self.manual_fail_requested = True
                    self.manual_success_requested = False
                    print("\n[RewardWrapper] SPACE pressed. Marking failure.")
                elif key == keyboard.Key.enter:
                    self.manual_success_requested = True
                    self.manual_fail_requested = False
                    print("\n[RewardWrapper] ENTER pressed. Marking success.")
                elif hasattr(key, "char") and key.char is not None:
                    key_char = key.char.lower()
                    if get_flag_value("handoff_keyboard_enabled", True) and key_char == "h":
                        self.handoff_focus_enabled = not self.handoff_focus_enabled
                        state = "enabled" if self.handoff_focus_enabled else "disabled"
                        print(f"\n[Handoff] focus {state}. New transitions will {'be' if self.handoff_focus_enabled else 'not be'} tagged as handoff.")
            except Exception:
                pass

        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

    def evaluate_transition(self, obs):
        reward, done, info = self._evaluate_current_reward()
        return obs, reward, done, False, info

    def reset(self, **kwargs):
        self.manual_fail_requested = False
        self.manual_success_requested = False
        obs, info = self.env.reset(**kwargs)
        print_green(f"[RewardWrapper] reset: keyboard-only reward, train_arm={get_train_arm_mode()}")
        return obs, info

    def is_handoff_focus_enabled(self):
        return bool(self.handoff_focus_enabled)

    def step(self, action):
        obs, _, done, truncated, info = self.env.step(action)
        reward, manual_done, reward_info = self._evaluate_current_reward()
        done = bool(done or manual_done)
        info.update(reward_info)
        return obs, reward, done, truncated, info

    def close(self):
        if self.render_enabled:
            try:
                cv2.destroyWindow(self.render_window_name)
            except Exception:
                pass
        return self.env.close()


class DualSpacemouseIntervention(gym.ActionWrapper):
    """
    Bimanual teleop wrapper that preserves the current OpenArm control semantics.
    """

    def __init__(
        self,
        env,
        left_event_path="auto",
        right_event_path="auto",
        trans_denom=420.0,
        rot_denom=380.0,
        deadzone=0.08,
        rot_deadzone=0.16,
        ee_x="x",
        ee_y="-y",
        ee_z="-z",
        control_hz=80.0,
        servo_backend="analytic",
        servo_hz=100.0,
        servo_trans_step=0.004,
        servo_rot_step=0.012,
        servo_gripper_step=0.05,
        print_raw=False,
    ):
        super().__init__(env)
        self.left_event_path = left_event_path
        self.right_event_path = right_event_path
        self.trans_denom = float(trans_denom)
        self.rot_denom = float(rot_denom)
        self.deadzone = float(deadzone)
        self.rot_deadzone = float(rot_deadzone)
        self.ee_x = ee_x
        self.ee_y = ee_y
        self.ee_z = ee_z
        self.control_hz = float(control_hz)
        self.servo_backend = str(servo_backend)
        self.servo_hz = float(servo_hz)
        self.servo_trans_step = float(servo_trans_step)
        self.servo_rot_step = float(servo_rot_step)
        self.servo_gripper_step = float(servo_gripper_step)
        self.print_raw = bool(print_raw)

        self.axis_codes = {}
        if ecodes is not None:
            self.axis_codes = {
                ecodes.ABS_X: "x",
                ecodes.ABS_Y: "y",
                ecodes.ABS_Z: "z",
                ecodes.ABS_RX: "rx",
                ecodes.ABS_RY: "ry",
                ecodes.ABS_RZ: "rz",
                ecodes.REL_X: "x",
                ecodes.REL_Y: "y",
                ecodes.REL_Z: "z",
                ecodes.REL_RX: "rx",
                ecodes.REL_RY: "ry",
                ecodes.REL_RZ: "rz",
            }

        self._left = self._make_device_state("left")
        self._right = self._make_device_state("right")
        self._servo_running = False
        self._last_obs = None
        self._prev_obs_for_transition = None
        self._target_pose_ref = None
        self._last_servo_mode = False
        self._intervention_mode = False
        self._idle_hold_sent = False
        self._init_devices()

    def _make_device_state(self, label):
        return {
            "label": label,
            "axes": {"x": 0.0, "y": 0.0, "z": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
            "button_state": {"gripper_close": False},
            "event_path": None,
            "dev": None,
            "enabled": False,
            "gripper_toggle_changed": False,
        }

    def _sync_button_state_from_hardware(self):
        binary = getattr(self.env.unwrapped, "gripper_binary_state", None)
        if binary is None:
            return
        binary = np.asarray(binary, dtype=np.int32).reshape(-1)
        if binary.size < 2:
            return
        self._left["button_state"]["gripper_close"] = bool(binary[0])
        self._right["button_state"]["gripper_close"] = bool(binary[1])
        self._left["gripper_toggle_changed"] = False
        self._right["gripper_toggle_changed"] = False

    def _has_3dx_axes(self, dev):
        caps = dev.capabilities(absinfo=False)
        abs_codes = set(caps.get(ecodes.EV_ABS, []))
        rel_codes = set(caps.get(ecodes.EV_REL, []))
        trans_abs = {ecodes.ABS_X, ecodes.ABS_Y, ecodes.ABS_Z}
        rot_abs = {ecodes.ABS_RX, ecodes.ABS_RY, ecodes.ABS_RZ}
        trans_rel = {ecodes.REL_X, ecodes.REL_Y, ecodes.REL_Z}
        rot_rel = {ecodes.REL_RX, ecodes.REL_RY, ecodes.REL_RZ}
        return (trans_abs.issubset(abs_codes) and rot_abs.issubset(abs_codes)) or (
            trans_rel.issubset(rel_codes) and rot_rel.issubset(rel_codes)
        )

    def _auto_detect_event_path(self, exclude_paths=None):
        exclude_paths = set(exclude_paths or [])
        try:
            from evdev import list_devices
        except Exception:
            return None
        candidates = []
        for path in list_devices():
            if path in exclude_paths:
                continue
            try:
                dev = InputDevice(path)
            except (PermissionError, OSError):
                continue
            name = (dev.name or "").lower()
            name_hit = any(
                key in name
                for key in (
                    "3dconnexion",
                    "spacemouse",
                    "space mouse",
                    "spacenavigator",
                    "space navigator",
                )
            )
            axes_hit = self._has_3dx_axes(dev)
            if name_hit or axes_hit:
                score = (100 if name_hit else 0) + (10 if axes_hit else 0)
                candidates.append((score, path, dev.name))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1], candidates[0][2]

    def _open_device(self, desired_path, exclude_paths=None):
        event_path = desired_path
        if str(desired_path).strip().lower() == "auto":
            detected = self._auto_detect_event_path(exclude_paths=exclude_paths)
            if detected is None:
                return None, None
            event_path, dev_name = detected
            print(f"[DualSpacemouse] auto-detected {event_path} ({dev_name})")
        try:
            dev = InputDevice(event_path)
            try:
                dev.grab()
            except OSError:
                pass
            flags_ = fcntl.fcntl(dev.fd, fcntl.F_GETFL)
            fcntl.fcntl(dev.fd, fcntl.F_SETFL, flags_ | os.O_NONBLOCK)
            return dev, event_path
        except Exception as exc:
            print(f"[WARN] Failed to open spacemouse device {event_path}: {exc}")
            return None, None

    def _init_devices(self):
        if InputDevice is None or ecodes is None:
            print("[WARN] evdev unavailable; disabling dual spacemouse intervention.")
            return
        left_dev, left_path = self._open_device(self.left_event_path)
        exclude = {left_path} if left_path else set()
        right_dev, right_path = self._open_device(self.right_event_path, exclude_paths=exclude)
        self._left["dev"] = left_dev
        self._left["event_path"] = left_path
        self._left["enabled"] = left_dev is not None
        self._right["dev"] = right_dev
        self._right["event_path"] = right_path
        self._right["enabled"] = right_dev is not None
        print(
            "[DualSpacemouse] left="
            f"{left_path or 'disabled'} right={right_path or 'disabled'} "
            f"mapping X<-{self.ee_x}, Y<-{self.ee_y}, Z<-{self.ee_z}"
        )

    @staticmethod
    def _apply_deadzone(value, deadzone):
        return 0.0 if abs(value) < deadzone else value

    @staticmethod
    def _parse_axis_spec(spec):
        spec = str(spec).strip().lower()
        if spec.startswith("-"):
            return spec[1:], -1.0
        return spec, 1.0

    def _device_to_ee_translation(self, axes):
        def get_norm(axis_spec):
            name, sign = self._parse_axis_spec(axis_spec)
            raw = axes.get(name, 0.0)
            return sign * np.clip(raw / self.trans_denom, -1.0, 1.0)

        return get_norm(self.ee_x), get_norm(self.ee_y), get_norm(self.ee_z)

    @staticmethod
    def _find_wrapper(wrapped_env, cls):
        cur = wrapped_env
        for _ in range(32):
            if isinstance(cur, cls):
                return cur
            if not hasattr(cur, "env"):
                break
            cur = cur.env
        return None

    def _get_reward_wrapper(self):
        return self._find_wrapper(self.env, KeyboardRewardWrapper)

    def _get_chunking_wrapper(self):
        return self._find_wrapper(self.env, ChunkingWrapper)

    def _transform_obs_to_policy_space(self, raw_obs):
        obs = raw_obs
        relative_wrapper = self._find_wrapper(self.env, DualRelativeFrame)
        if relative_wrapper is not None:
            tcp_pose = np.asarray(obs["state"]["tcp_pose"])
            relative_wrapper.left_transform = construct_adjoint_matrix(tcp_pose[0])
            relative_wrapper.right_transform = construct_adjoint_matrix(tcp_pose[1])
            obs = relative_wrapper.transform_observation(obs)

        quat_wrapper = self._find_wrapper(self.env, Quat2EulerWrapper)
        if quat_wrapper is not None:
            obs = quat_wrapper.observation(obs)

        crop_wrapper = self._find_wrapper(self.env, NetworkPrimaryImageCropWrapper)
        if crop_wrapper is not None:
            obs = crop_wrapper.observation(obs)

        serl_wrapper = self._find_wrapper(self.env, SERLObsWrapper)
        if serl_wrapper is not None:
            obs = serl_wrapper.observation(obs)

        chunking_wrapper = self._get_chunking_wrapper()
        if chunking_wrapper is not None:
            chunking_wrapper.current_obs.append(obs)
            obs = stack_obs(chunking_wrapper.current_obs)

        arm_focus_wrapper = self._find_wrapper(self.env, ArmFocusWrapper)
        if arm_focus_wrapper is not None:
            obs = arm_focus_wrapper._focus_obs(obs)
        return obs

    def _transform_action_to_base(self, action):
        transformed = np.array(action, copy=True)
        relative_wrapper = self._find_wrapper(self.env, DualRelativeFrame)
        if relative_wrapper is not None:
            transformed = relative_wrapper.transform_action(transformed)
        return transformed

    def _transform_action_to_policy(self, action):
        transformed = np.array(action, copy=True)
        relative_wrapper = self._find_wrapper(self.env, DualRelativeFrame)
        if relative_wrapper is not None:
            transformed = relative_wrapper.transform_action_inv(transformed)
        return transformed

    def _post_server_json(self, route, payload, timeout):
        base_env = self.env.unwrapped
        return base_env.session.post(base_env.url.rstrip("/") + route, json=payload, timeout=timeout)

    def _start_servo(self):
        if self._servo_running:
            return
        base_env = self.env.unwrapped
        if self._target_pose_ref is None:
            base_env.refresh_obs()
            self._target_pose_ref = np.array(base_env.currpos, copy=True)
        payload = {
            "arr": np.asarray(self._target_pose_ref, dtype=np.float32).tolist(),
            "gripper": [float(x) for x in base_env.curr_gripper_pos],
            "servo_hz": self.servo_hz,
            "trans_step": self.servo_trans_step,
            "rot_step": self.servo_rot_step,
            "gripper_step": self.servo_gripper_step,
            "backend": self.servo_backend,
        }
        resp = self._post_server_json("/servo/start", payload, timeout=3.0)
        resp.raise_for_status()
        self._servo_running = True

    def _stop_servo(self):
        if not self._servo_running:
            return
        try:
            self._post_server_json("/servo/stop", {}, timeout=2.0)
        except Exception:
            pass
        self._servo_running = False

    def _build_hold_action(self):
        return np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0 if self._left["button_state"]["gripper_close"] else -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0 if self._right["button_state"]["gripper_close"] else -1.0,
            ],
            dtype=np.float32,
        )

    def _update_target_pose_ref(self, action):
        base_env = self.env.unwrapped
        base_action = self._transform_action_to_base(action)
        updated = np.array(self._target_pose_ref, copy=True)
        for arm_idx in range(2):
            arm_action = base_action[arm_idx * 7 : (arm_idx + 1) * 7]
            updated[arm_idx, :3] += arm_action[:3] * base_env.action_scale[0]
            rot_curr = R.from_quat(updated[arm_idx, 3:])
            rot_delta = R.from_euler("xyz", arm_action[3:6] * base_env.action_scale[1])
            updated[arm_idx, 3:] = (rot_delta * rot_curr).as_quat()
        self._target_pose_ref = updated

    def _update_servo_target(self, action):
        base_env = self.env.unwrapped
        payload = {
            "arr": np.asarray(self._target_pose_ref, dtype=np.float32).tolist(),
            "gripper": base_env.gripper_actions_to_commands(action),
        }
        resp = self._post_server_json("/servo/target", payload, timeout=2.0)
        resp.raise_for_status()

    def _sample_env_like_step(self, action):
        base_env = self.env.unwrapped
        raw_next_obs = base_env.refresh_obs()
        next_obs = self._transform_obs_to_policy_space(raw_next_obs)
        reward_wrapper = self._get_reward_wrapper()
        if reward_wrapper is not None:
            next_obs, reward, done, truncated, reward_info = reward_wrapper.evaluate_transition(next_obs)
        else:
            reward = np.asarray(0.0, dtype=np.float32)
            done = False
            truncated = False
            reward_info = {}
        transition = {
            "observations": self._prev_obs_for_transition,
            "actions": np.array(action, copy=True),
            "next_observations": next_obs,
            "rewards": reward,
            "dones": done,
            "truncated": truncated,
        }
        info = dict(reward_info)
        info.update(
            {
                "intervene_action": np.array(action, copy=True),
                "intervention_mode": True,
                "intervened": True,
                "control_backend": self.servo_backend,
                "arm_focus": get_arm_focus_info(),
                "sampled_transition": transition,
            }
        )
        self._prev_obs_for_transition = next_obs
        self._last_obs = next_obs
        return next_obs, reward, done, truncated, info

    def _poll_one_device(self, state):
        if not state["enabled"] or state["dev"] is None:
            return False
        got_any = False
        state["gripper_toggle_changed"] = False
        while True:
            try:
                event = state["dev"].read_one()
            except (BlockingIOError, OSError):
                return got_any
            if event is None:
                return got_any
            got_any = True
            if event.type in (ecodes.EV_ABS, ecodes.EV_REL):
                axis = self.axis_codes.get(event.code)
                if axis is not None:
                    if event.type == ecodes.EV_REL:
                        state["axes"][axis] += float(event.value)
                    else:
                        state["axes"][axis] = float(event.value)
            elif event.type == ecodes.EV_KEY and int(event.value) == 1:
                if event.code in (ecodes.BTN_0, ecodes.BTN_LEFT):
                    self._intervention_mode = not self._intervention_mode
                    print(
                        f"[DualSpacemouse:{state['label']}] intervention mode: "
                        f"{'ON' if self._intervention_mode else 'OFF'}"
                    )
                elif event.code in (ecodes.BTN_1, ecodes.BTN_RIGHT):
                    state["button_state"]["gripper_close"] = not state["button_state"]["gripper_close"]
                    state["gripper_toggle_changed"] = True
                    print(
                        f"[DualSpacemouse:{state['label']}] gripper mode: "
                        f"{'CLOSE' if state['button_state']['gripper_close'] else 'OPEN'}"
                    )

    def _has_motion_input(self, axes):
        dx, dy, dz = self._device_to_ee_translation(axes)
        if abs(dx) > self.deadzone or abs(dy) > self.deadzone or abs(dz) > self.deadzone:
            return True
        rx = abs(np.clip(axes["rx"] / self.rot_denom, -1.0, 1.0))
        ry = abs(np.clip(-axes["ry"] / self.rot_denom, -1.0, 1.0))
        rz = abs(np.clip(-axes["rz"] / self.rot_denom, -1.0, 1.0))
        return rx > self.rot_deadzone or ry > self.rot_deadzone or rz > self.rot_deadzone

    def _build_arm_action(self, state):
        dx, dy, dz = self._device_to_ee_translation(state["axes"])
        dx = self._apply_deadzone(dx, self.deadzone)
        dy = self._apply_deadzone(dy, self.deadzone)
        dz = self._apply_deadzone(dz, self.deadzone)
        rx = self._apply_deadzone(np.clip(state["axes"]["rx"] / self.rot_denom, -1.0, 1.0), self.rot_deadzone)
        ry = self._apply_deadzone(np.clip(-state["axes"]["ry"] / self.rot_denom, -1.0, 1.0), self.rot_deadzone)
        rz = self._apply_deadzone(np.clip(-state["axes"]["rz"] / self.rot_denom, -1.0, 1.0), self.rot_deadzone)
        return np.array(
            [dx, dy, dz, rx, ry, rz, 1.0 if state["button_state"]["gripper_close"] else -1.0],
            dtype=np.float32,
        )

    def _build_intervene_action(self, policy_action):
        action = np.asarray(policy_action, dtype=np.float32).copy()
        if action.shape[0] != 14:
            action = np.zeros((14,), dtype=np.float32)
        base_action = self._transform_action_to_base(action)
        left_motion = arm_focus_allows_arm("left") and (
            self._has_motion_input(self._left["axes"]) or self._left["gripper_toggle_changed"]
        )
        right_motion = arm_focus_allows_arm("right") and (
            self._has_motion_input(self._right["axes"]) or self._right["gripper_toggle_changed"]
        )
        if left_motion:
            base_action[:7] = self._build_arm_action(self._left)
        else:
            base_action[:6] = 0.0
            base_action[6] = 1.0 if self._left["button_state"]["gripper_close"] else -1.0
        if right_motion:
            base_action[7:14] = self._build_arm_action(self._right)
        else:
            base_action[7:13] = 0.0
            base_action[13] = 1.0 if self._right["button_state"]["gripper_close"] else -1.0
        return mask_action_for_arm_focus(self._transform_action_to_policy(base_action))

    def reset(self, **kwargs):
        for state in (self._left, self._right):
            for key in state["axes"]:
                state["axes"][key] = 0.0
            state["gripper_toggle_changed"] = False
        self._stop_servo()
        self._intervention_mode = False
        self._idle_hold_sent = False
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self._prev_obs_for_transition = obs
        self._target_pose_ref = np.array(self.env.unwrapped.currpos, copy=True)
        return obs, info

    def step(self, action):
        loop_start = time.time()
        left_got = self._poll_one_device(self._left)
        right_got = self._poll_one_device(self._right)
        if not left_got:
            for key in self._left["axes"]:
                self._left["axes"][key] = 0.0
        if not right_got:
            for key in self._right["axes"]:
                self._right["axes"][key] = 0.0

        if self._intervention_mode and (not self._last_servo_mode):
            self.env.unwrapped.refresh_obs()
            self._target_pose_ref = np.array(self.env.unwrapped.currpos, copy=True)
            self._sync_button_state_from_hardware()
            self._idle_hold_sent = False
            if self._prev_obs_for_transition is None:
                self._prev_obs_for_transition = self._last_obs

        if self._intervention_mode:
            left_active = arm_focus_allows_arm("left") and (
                self._has_motion_input(self._left["axes"]) or self._left["gripper_toggle_changed"]
            )
            right_active = arm_focus_allows_arm("right") and (
                self._has_motion_input(self._right["axes"]) or self._right["gripper_toggle_changed"]
            )
            if (not left_active) and (not right_active):
                if not self._servo_running:
                    self.env.unwrapped.refresh_obs()
                    self._target_pose_ref = np.array(self.env.unwrapped.currpos, copy=True)
                    self._start_servo()
                hold_action = mask_action_for_arm_focus(self._build_hold_action())
                if (
                    (not self._idle_hold_sent)
                    or (
                        arm_focus_allows_arm("left")
                        and self._left["gripper_toggle_changed"]
                    )
                    or (
                        arm_focus_allows_arm("right")
                        and self._right["gripper_toggle_changed"]
                    )
                ):
                    self.env.unwrapped.refresh_obs()
                    self._target_pose_ref = np.array(self.env.unwrapped.currpos, copy=True)
                    self._update_servo_target(hold_action)
                    self._idle_hold_sent = True
                idle_info = {
                    "intervention_idle": True,
                    "intervention_mode": True,
                    "control_backend": self.servo_backend,
                }
                if self._last_obs is None:
                    self._last_obs, _ = self.env.reset()
                dt = time.time() - loop_start
                time.sleep(max(0.0, (1.0 / self.control_hz) - dt))
                self._last_servo_mode = True
                return self._last_obs, 0.0, False, False, idle_info

            chosen_action = self._build_intervene_action(action)
            self._idle_hold_sent = False
            if self.print_raw:
                print(f"[DualSpacemouse] action={np.round(chosen_action, 4).tolist()}")
            if not self._servo_running:
                self.env.unwrapped.refresh_obs()
                self._target_pose_ref = np.array(self.env.unwrapped.currpos, copy=True)
                self._start_servo()
            self._update_target_pose_ref(chosen_action)
            self._update_servo_target(chosen_action)
            obs, reward, done, truncated, info = self._sample_env_like_step(chosen_action)
            dt = time.time() - loop_start
            time.sleep(max(0.0, (1.0 / self.control_hz) - dt))
            self._last_servo_mode = True
            return obs, reward, done, truncated, info

        if self._servo_running:
            self.env.unwrapped.refresh_obs()
            self._target_pose_ref = np.array(self.env.unwrapped.currpos, copy=True)
            self._stop_servo()
            self._idle_hold_sent = False

        obs, reward, done, truncated, info = self.env.step(
            mask_action_for_arm_focus(np.asarray(action, dtype=np.float32))
        )
        self._last_obs = obs
        self._prev_obs_for_transition = obs
        self._last_servo_mode = False
        return obs, reward, done, truncated, info

    def close(self):
        self._stop_servo()
        for state in (self._left, self._right):
            dev = state["dev"]
            if dev is not None:
                try:
                    dev.ungrab()
                except OSError:
                    pass
        return self.env.close()


FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", "openarm_hilserl_bimanual", "Experiment name.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Run learner loop.")
flags.DEFINE_boolean("actor", False, "Run actor loop.")
flags.DEFINE_boolean("eval", False, "Run evaluation loop.")
flags.DEFINE_boolean("debug", False, "Disable wandb when true.")
flags.DEFINE_boolean("mock", False, "Use fake env / mock hardware observations.")
flags.DEFINE_boolean("render", False, "Render reward monitor windows.")
flags.DEFINE_boolean("save_video", False, "Keep compatibility flag for env creation.")
flags.DEFINE_string("ip", "localhost", "Learner IP.")
flags.DEFINE_multi_string(
    "demo_path",
    [str(DEFAULT_SUCCESS_DEMO_DIR)],
    "Offline demo buffer files or directories loaded into the intervention/demo buffer.",
)
flags.DEFINE_string(
    "checkpoint_path",
    "/home/sj/Desktop/zy/moqi_workspace/rl_deploy/checkpoints_hilserl",
    "Checkpoint directory.",
)
flags.DEFINE_integer("eval_checkpoint_step", 0, "Checkpoint step used for eval, 0 means latest.")
flags.DEFINE_integer("eval_n_trajs", 5, "Number of evaluation trajectories.")
flags.DEFINE_integer("hz", 5, "OpenArm env control frequency.")
flags.DEFINE_integer("max_traj_length", 400, "Max episode length.")
flags.DEFINE_integer("max_steps", 100000, "Max training steps.")
flags.DEFINE_integer("random_steps", 50, "Initial random exploration steps.")
flags.DEFINE_integer("training_starts", 300, "Replay warmup before learner updates.")
flags.DEFINE_integer("replay_buffer_capacity", 50000, "Replay buffer capacity.")
flags.DEFINE_integer("batch_size", 256, "Learner batch size.")
flags.DEFINE_boolean(
    "handoff_keyboard_enabled",
    True,
    "Enable keyboard toggling of handoff-focus tagging during actor data collection.",
)
flags.DEFINE_integer(
    "handoff_demo_repeat",
    4,
    "Number of demo-buffer insertions for each new handoff-focused intervention transition.",
)
flags.DEFINE_integer("steps_per_update", 50, "Network publish period.")
flags.DEFINE_integer("critic_actor_ratio", 4, "Critic-to-actor update ratio.")
flags.DEFINE_integer("log_period", 10, "Logging period.")
flags.DEFINE_integer("checkpoint_period", 200, "Checkpoint save period.")
flags.DEFINE_float("discount", 0.97, "Discount factor.")
flags.DEFINE_string("encoder_type", "resnet-pretrained", "Pixel encoder type.")
flags.DEFINE_boolean(
    "network_crop_primary",
    True,
    "Center-crop image_primary only for policy/replay observations.",
)
flags.DEFINE_float(
    "network_primary_crop_ratio",
    0.3,
    "Centered crop side ratio for image_primary before resizing to the network input size.",
)
flags.DEFINE_float(
    "network_primary_crop_y_offset",
    0.0,
    "Vertical crop center offset as a fraction of image height; positive moves the crop down.",
)
flags.DEFINE_string("left_spacemouse_event_path", "auto", "Left device event path.")
flags.DEFINE_string("right_spacemouse_event_path", "auto", "Right device event path.")
flags.DEFINE_float("spacemouse_trans_denom", 420.0, "Translation denominator.")
flags.DEFINE_float("spacemouse_rot_denom", 380.0, "Rotation denominator.")
flags.DEFINE_float("spacemouse_deadzone", 0.08, "Translation deadzone.")
flags.DEFINE_float("spacemouse_rot_deadzone", 0.16, "Rotation deadzone.")
flags.DEFINE_string("spacemouse_ee_x", "x", "Device axis mapped to EE X.")
flags.DEFINE_string("spacemouse_ee_y", "-y", "Device axis mapped to EE Y.")
flags.DEFINE_string("spacemouse_ee_z", "-z", "Device axis mapped to EE Z.")
flags.DEFINE_float("spacemouse_control_hz", 80.0, "Teleop control loop Hz.")
flags.DEFINE_string("spacemouse_servo_backend", "analytic", "Servo backend.")
flags.DEFINE_float("spacemouse_servo_hz", 100.0, "Servo loop Hz.")
flags.DEFINE_float("spacemouse_servo_trans_step", 0.004, "Servo translation step.")
flags.DEFINE_float("spacemouse_servo_rot_step", 0.012, "Servo rotation step.")
flags.DEFINE_float("spacemouse_servo_gripper_step", 0.05, "Servo gripper step.")
flags.DEFINE_boolean("spacemouse_print_raw", False, "Print raw teleop actions.")
flags.DEFINE_float(
    "grasp_penalty",
    0.0,
    "Gripper-penalty compatibility value. Keep at 0.0 to disable the penalty while preserving old checkpoint compatibility.",
)


def verify_camera_alignment():
    if APPLY_HEAD_CAMERA_CROP:
        raise ValueError(
            "APPLY_HEAD_CAMERA_CROP=True would mismatch the recorded teleop distribution."
        )
    print("[Consistency] camera alignment: image_primary -> RGB -> resize 128")


def find_wrapper(env, cls):
    cur = env
    for _ in range(32):
        if isinstance(cur, cls):
            return cur
        if not hasattr(cur, "env"):
            break
        cur = cur.env
    return None


def is_intervention_mode_active(env):
    wrapper = find_wrapper(env, DualSpacemouseIntervention)
    return bool(wrapper is not None and wrapper._intervention_mode)


def is_handoff_focus_active(env):
    wrapper = find_wrapper(env, KeyboardRewardWrapper)
    return bool(wrapper is not None and wrapper.is_handoff_focus_enabled())


def extract_gripper_binary_state_from_obs(obs):
    if not isinstance(obs, dict):
        return None
    state = obs.get("state")
    if state is None:
        return None
    # gripper_pose is emitted by the env as binary semantics in observation space:
    # open -> -1, close -> +1.
    if isinstance(state, dict) and "gripper_pose" in state:
        gripper_pose = np.asarray(state["gripper_pose"], dtype=np.float32).reshape(-1)
    else:
        gripper_pose = np.asarray(state, dtype=np.float32).reshape(-1)
        if gripper_pose.size < 2:
            return None
        gripper_pose = gripper_pose[-2:]
    return (gripper_pose >= 0.5).astype(np.int32)


def compute_grasp_penalty(
    action,
    last_binary_state,
    penalty,
    open_threshold,
    close_threshold,
):
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.size < 14:
        return np.asarray(0.0, dtype=np.float32)

    penalty_value = 0.0
    for arm_idx, gripper_idx in enumerate((6, 13)):
        last_state = int(last_binary_state[arm_idx])
        cmd = float(action[gripper_idx])
        next_state = apply_binary_gripper_logic(
            raw_val=cmd,
            prev_binary_state=last_state,
            open_threshold=open_threshold,
            close_threshold=close_threshold,
        )
        if next_state != last_state:
            penalty_value += float(penalty)
    return np.asarray(penalty_value, dtype=np.float32)


def maybe_add_grasp_penalty(transition):
    transition["grasp_penalty"] = np.asarray(0.0, dtype=np.float32)
    return transition


class GripperPenaltyWrapper(gym.Wrapper):
    """
    HIL-SERL-style learned-gripper penalty: prefer no-op over redundant open/close.
    """

    def __init__(self, env, penalty=-0.02):
        super().__init__(env)
        self.penalty = float(penalty)
        self.last_gripper_binary_state = np.zeros((2,), dtype=np.int32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        base_env = self.env.unwrapped
        self.last_gripper_binary_state = np.asarray(
            getattr(base_env, "gripper_binary_state", np.zeros((2,), dtype=np.int32)),
            dtype=np.int32,
        ).copy()
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        effective_action = info.get("intervene_action", action) if isinstance(info, dict) else action
        if not isinstance(info, dict):
            info = {}
        base_env = self.env.unwrapped
        open_threshold, close_threshold = get_gripper_thresholds(base_env)
        info["grasp_penalty"] = compute_grasp_penalty(
            effective_action,
            self.last_gripper_binary_state,
            self.penalty,
            open_threshold=open_threshold,
            close_threshold=close_threshold,
        )
        self.last_gripper_binary_state = np.asarray(
            getattr(base_env, "gripper_binary_state", self.last_gripper_binary_state),
            dtype=np.int32,
        ).copy()
        return obs, reward, done, truncated, info


def create_env(
    fake_env=False,
    enable_reward=False,
    enable_intervention=False,
    enable_network_crop=False,
):
    """Create the training environment with the configured wrappers."""
    max_traj = get_flag_value("max_traj_length", 400)

    env = LocalOpenArmEnv(
        fake_env=bool(fake_env or get_flag_value("mock", False)),
        save_video=get_flag_value("save_video", False),
        hz=get_flag_value("hz", 10),
        config=TrainOpenArmConfig(),
        max_episode_length=max_traj,
    )

    env = DualRelativeFrame(env)
    env = Quat2EulerWrapper(env)
    if enable_network_crop and get_flag_value("network_crop_primary", True):
        env = NetworkPrimaryImageCropWrapper(
            env,
            crop_ratio=get_flag_value("network_primary_crop_ratio", 0.3),
            y_offset_ratio=get_flag_value("network_primary_crop_y_offset", 0.0),
        )
    env = SERLObsWrapper(env, proprio_keys=PROPRIO_KEYS)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    if enable_reward:
        env = KeyboardRewardWrapper(env, reward_image_key="image_primary")

    env = GripperPenaltyWrapper(env, penalty=get_flag_value("grasp_penalty", -0.02))

    if enable_intervention:
        env = DualSpacemouseIntervention(
            env,
            left_event_path=get_flag_value("left_spacemouse_event_path", "auto"),
            right_event_path=get_flag_value("right_spacemouse_event_path", "auto"),
            trans_denom=get_flag_value("spacemouse_trans_denom", 420.0),
            rot_denom=get_flag_value("spacemouse_rot_denom", 380.0),
            deadzone=get_flag_value("spacemouse_deadzone", 0.08),
            rot_deadzone=get_flag_value("spacemouse_rot_deadzone", 0.16),
            ee_x=get_flag_value("spacemouse_ee_x", "x"),
            ee_y=get_flag_value("spacemouse_ee_y", "-y"),
            ee_z=get_flag_value("spacemouse_ee_z", "-z"),
            control_hz=get_flag_value("spacemouse_control_hz", 80.0),
            servo_backend=get_flag_value("spacemouse_servo_backend", "analytic"),
            servo_hz=get_flag_value("spacemouse_servo_hz", 100.0),
            servo_trans_step=get_flag_value("spacemouse_servo_trans_step", 0.004),
            servo_rot_step=get_flag_value("spacemouse_servo_rot_step", 0.012),
            servo_gripper_step=get_flag_value("spacemouse_servo_gripper_step", 0.05),
            print_raw=get_flag_value("spacemouse_print_raw", False),
        )

    env = RecordEpisodeStatistics(env)
    return env


def create_agent(env):
    agent = make_sac_pixel_agent_hybrid_dual_arm(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=TRAINING_IMAGE_KEYS,
        encoder_type=FLAGS.encoder_type,
        discount=FLAGS.discount,
    )
    return jax.device_put(
        jax.tree_util.tree_map(jnp.array, agent),
        sharding,
    )


def load_transition_files(paths, data_store):
    if not paths:
        return
    seen_paths = set()
    for path in paths:
        if path is None:
            continue
        norm_path = os.path.abspath(os.fspath(path))
        if norm_path in seen_paths:
            continue
        seen_paths.add(norm_path)
        if os.path.isdir(path):
            load_transition_dir(path, data_store)
            continue
        with open(path, "rb") as handle:
            transitions = pkl.load(handle)
        for transition in transitions:
            maybe_apply_network_image_crop_to_transition(transition)
            maybe_apply_arm_focus_to_transition(transition)
            maybe_add_grasp_penalty(transition)
            data_store.insert(transition)


def load_transition_dir(dir_path, data_store):
    if not dir_path or not os.path.exists(dir_path):
        return
    for path in sorted(glob.glob(os.path.join(dir_path, "*.pkl"))):
        with open(path, "rb") as handle:
            transitions = pkl.load(handle)
        for transition in transitions:
            maybe_apply_network_image_crop_to_transition(transition)
            maybe_apply_arm_focus_to_transition(transition)
            maybe_add_grasp_penalty(transition)
            data_store.insert(transition)


def _collect_transition_files(paths):
    collected = []
    seen = set()
    for path in paths:
        if path is None:
            continue
        norm_path = os.path.abspath(os.fspath(path))
        if norm_path in seen:
            continue
        seen.add(norm_path)
        if os.path.isdir(norm_path):
            for file_path in glob.glob(os.path.join(norm_path, "*.pkl")):
                collected.append(os.path.abspath(file_path))
        elif os.path.isfile(norm_path):
            collected.append(norm_path)
    unique_files = []
    seen_files = set()
    for file_path in collected:
        if file_path in seen_files:
            continue
        seen_files.add(file_path)
        unique_files.append(file_path)
    unique_files.sort(key=os.path.getmtime, reverse=True)
    return unique_files


def load_recent_transition_files(paths, data_store, limit):
    if limit is None or limit <= 0:
        return 0
    files = _collect_transition_files(paths)
    if not files:
        return 0

    selected = deque()
    remaining = int(limit)
    for file_path in files:
        if remaining <= 0:
            break
        with open(file_path, "rb") as handle:
            transitions = pkl.load(handle)
        for transition in reversed(transitions):
            if remaining <= 0:
                break
            selected.appendleft(transition)
            remaining -= 1

    loaded = 0
    for transition in selected:
        maybe_apply_network_image_crop_to_transition(transition)
        maybe_apply_arm_focus_to_transition(transition)
        maybe_add_grasp_penalty(transition)
        data_store.insert(transition)
        loaded += 1
    return loaded


def save_transition_dump(base_dir, subdir, step, transitions):
    if not transitions:
        return
    target_dir = os.path.join(base_dir, subdir)
    os.makedirs(target_dir, exist_ok=True)
    with open(os.path.join(target_dir, f"transitions_{step}.pkl"), "wb") as handle:
        pkl.dump(transitions, handle)


def get_demo_load_paths():
    configured_paths = list(FLAGS.demo_path or [])
    success_dir = os.path.abspath(os.fspath(DEFAULT_SUCCESS_DEMO_DIR))
    normalized = {os.path.abspath(os.fspath(path)) for path in configured_paths if path is not None}
    if success_dir not in normalized:
        configured_paths.append(success_dir)
    return configured_paths


def is_handoff_focus_transition(transition):
    if not isinstance(transition, dict):
        return False
    infos = transition.get("infos")
    return bool(isinstance(infos, dict) and infos.get("handoff_focus", False))


def load_prioritized_demo_transitions(paths, data_store, limit):
    files = _collect_transition_files(paths)
    if not files:
        return {
            "handoff_loaded": 0,
            "other_loaded": 0,
            "total_loaded": 0,
        }

    handoff_selected = deque()
    other_selected = deque()
    for file_path in files:
        with open(file_path, "rb") as handle:
            transitions = pkl.load(handle)
        for transition in reversed(transitions):
            if is_handoff_focus_transition(transition):
                handoff_selected.appendleft(transition)
            else:
                other_selected.appendleft(transition)

    selected = list(handoff_selected)
    other_loaded = 0
    if limit is None or limit <= 0:
        limit = 0
    if len(selected) < limit:
        needed = int(limit) - len(selected)
        selected.extend(list(other_selected)[-needed:] if needed > 0 else [])
        other_loaded = min(needed, len(other_selected))
    handoff_loaded = len(handoff_selected)

    loaded = 0
    for transition in selected:
        maybe_apply_network_image_crop_to_transition(transition)
        maybe_apply_arm_focus_to_transition(transition)
        maybe_add_grasp_penalty(transition)
        data_store.insert(transition)
        loaded += 1
    return {
        "handoff_loaded": handoff_loaded,
        "other_loaded": other_loaded,
        "total_loaded": loaded,
    }


def get_demo_load_paths():
    configured_paths = list(FLAGS.demo_path or [])
    success_dir = os.path.abspath(os.fspath(DEFAULT_SUCCESS_DEMO_DIR))
    normalized = {os.path.abspath(os.fspath(path)) for path in configured_paths if path is not None}
    if success_dir not in normalized:
        configured_paths.append(success_dir)
    return configured_paths


def is_handoff_focus_transition(transition):
    size = len(data_store)
    if size <= 0:
        return 0

    correct_index = getattr(data_store, "_is_correct_index", None)
    if correct_index is None:
        return size

    lock = getattr(data_store, "_lock", None)
    if lock is None:
        return int(np.count_nonzero(correct_index))

    with lock:
        return int(np.count_nonzero(correct_index))


def restore_agent_from_checkpoint(agent, checkpoint_path):
    if not checkpoint_path:
        raise ValueError("empty checkpoint path")
    checkpoint_path = os.path.abspath(os.fspath(checkpoint_path))
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint path does not exist: {checkpoint_path}")
    latest = checkpoints.latest_checkpoint(checkpoint_path)
    if latest is None:
        raise FileNotFoundError(f"no checkpoint found under: {checkpoint_path}")
    ckpt = checkpoints.restore_checkpoint(checkpoint_path, agent.state)
    print_green(f"[Eval] loaded checkpoint: {latest}")
    return agent.replace(state=ckpt)


def evaluate(agent, env, sampling_rng):
    """Simple evaluation loop."""
    if (
        FLAGS.checkpoint_path
        and os.path.exists(FLAGS.checkpoint_path)
    ):
        ckpt_target = agent.state
        if FLAGS.eval_checkpoint_step:
            ckpt = checkpoints.restore_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path),
                ckpt_target,
                step=FLAGS.eval_checkpoint_step,
            )
        else:
            ckpt = checkpoints.restore_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path),
                ckpt_target,
            )
        agent = agent.replace(state=ckpt)

    successes = 0
    times = []
    for episode in range(FLAGS.eval_n_trajs):
        obs, _ = env.reset()
        done = False
        start_time = time.time()
        while not done:
            sampling_rng, key = jax.random.split(sampling_rng)
            actions = agent.sample_actions(
                observations=jax.device_put(obs),
                seed=key,
                argmax=False,
            )
            actions = np.asarray(jax.device_get(actions))
            obs, reward, done, truncated, info = env.step(actions)
            done = bool(done or truncated)
        if reward:
            successes += 1
            times.append(time.time() - start_time)
        print(f"[Eval] episode={episode} reward={float(np.asarray(reward))}")

    print(f"success rate: {successes / max(FLAGS.eval_n_trajs, 1):.3f}")
    if times:
        print(f"average success time: {np.mean(times):.3f}s")


def actor(agent, data_store, intvn_data_store, env, sampling_rng, checkpoint_path=None):
    checkpoint_path = checkpoint_path or FLAGS.checkpoint_path
    if FLAGS.eval:
        evaluate(agent, env, sampling_rng)
        return

    start_step = 0
    buffer_dir = os.path.join(checkpoint_path, "buffer")
    if checkpoint_path and os.path.exists(buffer_dir):
        existing = sorted(glob.glob(os.path.join(buffer_dir, "transitions_*.pkl")))
        if existing:
            last_step = int(os.path.basename(existing[-1])[12:-4])
            start_step = last_step + 1

    datastore_dict = {"actor_env": data_store, "actor_env_intvn": intvn_data_store}
    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=3000,
    )

    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    reward_wrapper = find_wrapper(env, KeyboardRewardWrapper)
    obs, _ = env.reset()

    timer = Timer()
    running_return = 0.0
    intervention_count = 0
    intervention_steps = 0
    already_intervened = False
    handoff_focus_insertions = 0
    handoff_focus_transitions = 0
    last_timer_stats_step = None
    transitions = []
    demo_transitions = []

    # Episode-level monitoring: rolling success rate / counter / latest joint distance.
    success_history = deque(maxlen=100)
    episode_idx = 0
    last_joint_distance = 0.0
    episode_initial_distance = 0.0
    min_joint_distance = float("inf")
    distance_traj = []

    pbar = tqdm.tqdm(
        total=max(FLAGS.max_steps - start_step, 0),
        initial=0,
        dynamic_ncols=True,
        desc="actor",
    )
    step = start_step
    while step < FLAGS.max_steps:
        timer.tick("total")

        with timer.context("sample_actions"):
            if is_intervention_mode_active(env):
                actions = np.zeros(env.action_space.shape, dtype=np.float32)
            elif step < FLAGS.random_steps:
                actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    argmax=False,
                )
                actions = np.asarray(jax.device_get(actions))

        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(actions)
            sampled_transition = info.get("sampled_transition") if isinstance(info, dict) else None

            if isinstance(info, dict) and info.get("intervention_idle", False):
                timer.tock("total")
                if step % FLAGS.log_period == 0 and last_timer_stats_step != step:
                    client.request("send-stats", {"timer": timer.get_average_times(reset=False)})
                    last_timer_stats_step = step
                continue

            intervened = "intervene_action" in info
            if intervened:
                actions = np.asarray(info["intervene_action"], dtype=np.float32)
                intervention_steps += 1
                if not already_intervened:
                    intervention_count += 1
                already_intervened = True
            else:
                if isinstance(info, dict) and "arm_focus_action" in info:
                    actions = np.asarray(info["arm_focus_action"], dtype=np.float32)
                already_intervened = False

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
                reward = transition["rewards"]
                done = transition["dones"]
                truncated = bool(sampled_transition.get("truncated", False))
                intervened = True
            else:
                transition = {
                    "observations": obs,
                    "actions": np.asarray(actions, dtype=np.float32),
                    "next_observations": next_obs,
                    "rewards": np.asarray(reward, dtype=np.float32),
                    "masks": np.asarray(1.0 - float(done), dtype=np.float32),
                    "dones": bool(done),
                    "infos": copy.deepcopy(info),
                }

            maybe_add_grasp_penalty(transition)
            infos = transition.get("infos")
            if not isinstance(infos, dict):
                infos = {}
                transition["infos"] = infos
            if is_handoff_focus_active(env):
                infos["handoff_focus"] = True
                infos["handoff_mark_source"] = "keyboard"
            if get_flag_value("network_crop_primary", True):
                mark_transition_network_crop(transition)
            if arm_focus_enabled():
                maybe_apply_arm_focus_to_transition(transition)
            data_store.insert(transition)
            transitions.append(transition.copy())

            if intervened:
                demo_repeat = 1
                if infos.get("handoff_focus", False):
                    handoff_focus_transitions += 1
                    demo_repeat = max(1, int(get_flag_value("handoff_demo_repeat", 4)))
                    handoff_focus_insertions += demo_repeat
                for _ in range(demo_repeat):
                    intvn_data_store.insert(transition)
                demo_transitions.append(transition.copy())

            obs = next_obs
            running_return += float(np.asarray(reward))
            if isinstance(info, dict) and "joint_distance" in info:
                d = float(info["joint_distance"])
                last_joint_distance = d
                if d < min_joint_distance:
                    min_joint_distance = d
                distance_traj.append(d)
            if isinstance(info, dict) and "joint_initial_distance" in info and episode_initial_distance == 0.0:
                episode_initial_distance = float(info["joint_initial_distance"])

            if done or truncated:
                episode_success = bool(info.get("success", False)) if isinstance(info, dict) else False
                success_history.append(int(episode_success))
                episode_idx += 1
                episode_length = (
                    int(info["episode"].get("l", 0))
                    if isinstance(info, dict) and isinstance(info.get("episode"), dict)
                    else 0
                )
                intervention_ratio = (
                    float(intervention_steps) / max(episode_length, 1)
                    if episode_length > 0
                    else 0.0
                )

                final_distance = (
                    float(distance_traj[-1]) if distance_traj else float(last_joint_distance)
                )
                min_distance_seen = (
                    float(min_joint_distance) if min_joint_distance != float("inf") else final_distance
                )
                distance_reduction_ratio = (
                    (episode_initial_distance - final_distance) / episode_initial_distance
                    if episode_initial_distance > 1e-6
                    else 0.0
                )
                if len(distance_traj) >= 2:
                    decreasing = sum(
                        1 for a, b in zip(distance_traj, distance_traj[1:]) if a > b
                    )
                    distance_monotonicity = float(decreasing) / float(len(distance_traj) - 1)
                else:
                    distance_monotonicity = 0.0
                mean_step_reward = float(running_return) / max(episode_length, 1)
                autonomy_rate = 1.0 - float(intervention_ratio)

                if not isinstance(info, dict):
                    info = {}
                if "episode" in info and isinstance(info["episode"], dict):
                    info["episode"]["intervention_count"] = intervention_count
                    info["episode"]["intervention_steps"] = intervention_steps
                info["train_arm"] = get_train_arm_mode()
                info["episode_idx"] = episode_idx
                info["episode_success"] = int(episode_success)
                info["success_rate_100"] = float(np.mean(success_history))
                info["intervention_ratio"] = float(intervention_ratio)
                info["autonomy_rate"] = float(autonomy_rate)
                info["final_joint_distance"] = final_distance
                info["min_joint_distance"] = min_distance_seen
                info["distance_reduction_ratio"] = float(distance_reduction_ratio)
                info["distance_monotonicity"] = float(distance_monotonicity)
                info["mean_step_reward"] = float(mean_step_reward)
                info["episode_return"] = float(running_return)
                info["episode_initial_distance"] = float(episode_initial_distance)
                if episode_success and episode_length > 0:
                    info["time_to_success"] = int(episode_length)
                stats = {"environment": info}
                client.request("send-stats", stats)
                pbar.set_description(
                    f"ep={episode_idx} "
                    f"sr100={info['success_rate_100']:.2f} "
                    f"return={running_return:.3f}"
                )
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                last_joint_distance = 0.0
                episode_initial_distance = 0.0
                min_joint_distance = float("inf")
                distance_traj.clear()
                already_intervened = False
                client.update()

                obs, _ = env.reset()

        if step > 0 and step % FLAGS.steps_per_update == 0:
            client.update()

        if step > 0 and step % FLAGS.checkpoint_period == 0:
            save_transition_dump(checkpoint_path, "buffer", step, transitions)
            save_transition_dump(checkpoint_path, "demo_buffer", step, demo_transitions)
            transitions = []
            demo_transitions = []

        timer.tock("total")
        if step % FLAGS.log_period == 0 and last_timer_stats_step != step:
            client.request("send-stats", {"timer": timer.get_average_times()})
            if handoff_focus_transitions > 0:
                print_green(
                    "[HandoffStats] "
                    f"handoff_transitions={handoff_focus_transitions}, "
                    f"demo_buffer_insertions={handoff_focus_insertions}"
                )
            last_timer_stats_step = step
        step += 1
        pbar.update(1)
    pbar.close()


def learner(rng, agent, replay_buffer, demo_buffer, wandb_logger=None, checkpoint_path=None):
    checkpoint_path = checkpoint_path or FLAGS.checkpoint_path
    start_step = 0
    latest = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        latest = checkpoints.latest_checkpoint(os.path.abspath(checkpoint_path))
    if latest:
        start_step = int(os.path.basename(latest)[11:]) + 1
    step_state = {"value": start_step}

    def stats_callback(req_type, payload):
        assert req_type == "send-stats"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step_state["value"])
        return {}

    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    pbar = tqdm.tqdm(
        total=FLAGS.training_starts,
        initial=len(replay_buffer),
        desc="Filling replay buffer",
        leave=True,
    )
    while len(replay_buffer) < FLAGS.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)
    pbar.close()

    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    def make_replay_iterator(batch_size):
        return replay_buffer.get_iterator(
            sample_args={
                "batch_size": batch_size,
                "pack_obs_and_next_obs": True,
            },
            device=sharding,
        )

    def make_demo_iterator(batch_size):
        return demo_buffer.get_iterator(
            sample_args={
                "batch_size": batch_size,
                "pack_obs_and_next_obs": True,
            },
            device=sharding,
        )

    demo_batch_size = max(1, FLAGS.batch_size // 2)
    demo_sampleable = count_sampleable_transitions(demo_buffer)
    use_demo = demo_sampleable > 0
    replay_batch_size = FLAGS.batch_size // 2 if use_demo else FLAGS.batch_size
    replay_iterator = make_replay_iterator(replay_batch_size)
    demo_iterator = make_demo_iterator(demo_batch_size) if use_demo else None
    if use_demo:
        print_green(
            "demo buffer available at startup; learner begins in mixed mode "
            f"(demo_sampleable={demo_sampleable}, replay_batch={replay_batch_size}, demo_batch={demo_batch_size})."
        )
    else:
        print_yellow(
            "no sampleable demo data at startup; learner begins in replay-only mode "
            "and will switch to mixed mode once demo data becomes available."
        )

    train_critic_networks = frozenset({"critic", "grasp_critic"})
    train_networks = frozenset({"critic", "grasp_critic", "actor", "temperature"})
    timer = Timer()

    for step in tqdm.tqdm(range(start_step, FLAGS.max_steps), dynamic_ncols=True, desc="learner"):
        step_state["value"] = step
        if not use_demo:
            demo_sampleable = count_sampleable_transitions(demo_buffer)
            if demo_sampleable > 0:
                use_demo = True
                replay_batch_size = FLAGS.batch_size // 2
                replay_iterator = make_replay_iterator(replay_batch_size)
                demo_iterator = make_demo_iterator(demo_batch_size)
                print_green(
                    "demo buffer became available during training; switching learner to mixed mode "
                    f"at step {step} (demo_sampleable={demo_sampleable}, replay_batch={replay_batch_size}, demo_batch={demo_batch_size})."
                )
        for _ in range(max(FLAGS.critic_actor_ratio - 1, 0)):
            with timer.context("sample_replay"):
                batch = next(replay_iterator)
                if use_demo:
                    demo_batch = next(demo_iterator)
                    batch = concat_batches(batch, demo_batch, axis=0)
            with timer.context("train_critics"):
                agent, _ = agent.update(batch, networks_to_update=train_critic_networks)

        with timer.context("train"):
            batch = next(replay_iterator)
            if use_demo:
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update(batch, networks_to_update=train_networks)

        if step > 0 and step % FLAGS.steps_per_update == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if wandb_logger and step % FLAGS.log_period == 0:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log(
                {
                    "timer": timer.get_average_times(),
                    "replay_size": len(replay_buffer),
                    "intervention_buffer_size": len(demo_buffer),
                    "demo_batch_ratio": float(demo_batch_size) / FLAGS.batch_size if use_demo else 0.0,
                },
                step=step,
            )

        if step > 0 and step % FLAGS.checkpoint_period == 0:
            checkpoints.save_checkpoint(
                os.path.abspath(checkpoint_path),
                agent.state,
                step=step,
                keep=100,
            )


def main(_):
    if sum([FLAGS.actor, FLAGS.learner, FLAGS.eval]) != 1:
        raise ValueError("Exactly one of --actor, --learner, --eval must be true.")
    if FLAGS.batch_size % max(num_devices, 1) != 0:
        raise ValueError("batch_size must be divisible by the number of local JAX devices.")

    checkpoint_path = FLAGS.checkpoint_path

    verify_camera_alignment()

    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    env = create_env(
        fake_env=FLAGS.learner,
        enable_reward=(FLAGS.actor or FLAGS.eval),
        enable_intervention=FLAGS.actor,
        enable_network_crop=True,
    )

    print(f"Training image keys: {TRAINING_IMAGE_KEYS}")
    print(f"Train arm mode: {get_train_arm_mode()} (bimanual)")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Reward: keyboard-only (SPACE=fail, ENTER=success)")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    agent = create_agent(env)

    if checkpoint_path and os.path.exists(checkpoint_path):
        latest = checkpoints.latest_checkpoint(os.path.abspath(checkpoint_path))
        if latest:
            ckpt = checkpoints.restore_checkpoint(
                os.path.abspath(checkpoint_path),
                agent.state,
            )
            agent = agent.replace(state=ckpt)
            print_green(f"Loaded checkpoint: {latest}")

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding)
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=FLAGS.replay_buffer_capacity,
            image_keys=TRAINING_IMAGE_KEYS,
            include_grasp_penalty=True,
        )
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=FLAGS.replay_buffer_capacity,
            image_keys=TRAINING_IMAGE_KEYS,
            include_grasp_penalty=True,
        )
        # Always load full buffer history
        load_transition_files(get_demo_load_paths(), demo_buffer)
        load_transition_dir(os.path.join(checkpoint_path, "buffer"), replay_buffer)
        load_transition_dir(os.path.join(checkpoint_path, "demo_buffer"), demo_buffer)
        print_green(f"replay buffer size: {len(replay_buffer)}")
        print_green(f"demo buffer size: {len(demo_buffer)}")

        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=f"{FLAGS.exp_name}_{get_train_arm_mode()}",
            debug=FLAGS.debug,
        )
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer,
            wandb_logger=wandb_logger,
            checkpoint_path=checkpoint_path,
        )
    else:
        sampling_rng = jax.device_put(sampling_rng, sharding)
        data_store = QueuedDataStore(50000)
        intvn_data_store = QueuedDataStore(50000)
        actor(agent, data_store, intvn_data_store, env, sampling_rng, checkpoint_path=checkpoint_path)


if __name__ == "__main__":
    app.run(main)
