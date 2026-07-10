"""OpenArm-specific Gymnasium wrappers."""
import fcntl
import os
import time

import cv2
import numpy as np
import gymnasium as gym
from scipy.spatial.transform import Rotation as R

from openarm_env.utils.transformations import (
    construct_adjoint_matrix,
    construct_homogeneous_matrix,
)
from rl_launcher.wrappers import ChunkingWrapper, SERLObsWrapper

from openarm_env.envs.openarm_env import (
    apply_binary_gripper_logic,
    get_gripper_thresholds,
)
from openarm_env.camera.local_camera import MODEL_IMAGE_SIZE

try:
    from evdev import InputDevice, ecodes
except Exception:  # pragma: no cover - evdev/hardware dependent
    InputDevice = None
    ecodes = None


# ---------------------------------------------------------------------------
# Observation stacking helper.
# ---------------------------------------------------------------------------
def stack_obs(obs):
    import jax

    dict_list = {key: [item[key] for item in obs] for key in obs[0]}
    return jax.tree_util.tree_map(
        lambda values: np.stack(values),
        dict_list,
        is_leaf=lambda value: isinstance(value, list),
    )


# ---------------------------------------------------------------------------
# Image crop helpers.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Grasp penalty helpers.
# ---------------------------------------------------------------------------
def compute_grasp_penalty(action, last_binary_state, penalty, open_threshold, close_threshold):
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


# ===========================================================================
# Policy obs adapter
# ===========================================================================
class OpenArmPolicyObsAdapter:
    """Convert base OpenArm observations into the policy observation space."""

    def __init__(
        self,
        relative_wrapper=None,
        quat_wrapper=None,
        crop_wrapper=None,
        serl_wrapper=None,
        chunking_wrapper=None,
    ):
        self.relative_wrapper = relative_wrapper
        self.quat_wrapper = quat_wrapper
        self.crop_wrapper = crop_wrapper
        self.serl_wrapper = serl_wrapper
        self.chunking_wrapper = chunking_wrapper

    def __call__(self, raw_obs):
        obs = raw_obs
        if self.relative_wrapper is not None:
            tcp_pose = np.asarray(obs["state"]["tcp_pose"])
            self.relative_wrapper.left_transform = construct_adjoint_matrix(tcp_pose[0])
            self.relative_wrapper.right_transform = construct_adjoint_matrix(tcp_pose[1])
            obs = self.relative_wrapper.transform_observation(obs)
        if self.quat_wrapper is not None:
            obs = self.quat_wrapper.observation(obs)
        if self.crop_wrapper is not None:
            obs = self.crop_wrapper.observation(obs)
        if self.serl_wrapper is not None:
            obs = self.serl_wrapper.observation(obs)
        if self.chunking_wrapper is not None:
            self.chunking_wrapper.current_obs.append(obs)
            obs = stack_obs(self.chunking_wrapper.current_obs)
        return obs


# ===========================================================================
# DualRelativeFrame
# ===========================================================================
class DualRelativeFrame(gym.Wrapper):
    """Minimal dual-arm relative-frame wrapper for OpenArm's packed state schema."""

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


# ===========================================================================
# Quat2EulerWrapper
# ===========================================================================
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


# ===========================================================================
# NetworkPrimaryImageCropWrapper
# ===========================================================================
class NetworkPrimaryImageCropWrapper(gym.ObservationWrapper):
    """Applies center crop only to the policy/replay observation stream.

    Raw camera buffers stay untouched. When the base env exposes
    latest_images_raw (LocalOpenArmEnv), the crop is computed from the
    full-resolution head frame for better quality. Downstream policy, critic,
    demo pkl, and classifier inputs all consume this same cropped
    image_primary.
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


# ===========================================================================
# GripperPenaltyWrapper
# ===========================================================================
class GripperPenaltyWrapper(gym.Wrapper):
    """HIL-SERL-style learned-gripper penalty: prefer no-op over redundant open/close."""

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


# ===========================================================================
# DualSpacemouseIntervention
# ===========================================================================
class DualSpacemouseIntervention(gym.ActionWrapper):
    """Bimanual teleop wrapper preserving OpenArm control semantics.

    Every arm is always active, actions are never masked, and intervention
    transitions use the outer MultiCameraBinaryRewardClassifierWrapper reward.
    Requires the base env to expose
    refresh_obs() / currpos / gripper_binary_state / session / url and the flask
    server's /control/* routes.
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
        transition_sample_delay=0.0,
        print_raw=False,
        policy_obs_adapter=None,
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
        self.transition_sample_delay = float(transition_sample_delay)
        self.print_raw = bool(print_raw)
        self.policy_obs_adapter = policy_obs_adapter

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

    def _get_chunking_wrapper(self):
        return self._find_wrapper(self.env, ChunkingWrapper)

    def _transform_obs_to_policy_space(self, raw_obs):
        if self.policy_obs_adapter is None:
            raise RuntimeError("DualSpacemouseIntervention requires a policy_obs_adapter")
        return self.policy_obs_adapter(raw_obs)

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
        resp = self._post_server_json("/control/start", payload, timeout=3.0)
        resp.raise_for_status()
        self._servo_running = True

    def _stop_servo(self):
        if not self._servo_running:
            return
        try:
            self._post_server_json("/control/stop", {}, timeout=2.0)
        except Exception:
            pass
        self._servo_running = False

    def _build_hold_action(self):
        return np.array(
            [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1.0 if self._left["button_state"]["gripper_close"] else -1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
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
        resp = self._post_server_json("/control/target", payload, timeout=2.0)
        resp.raise_for_status()

    def _sample_env_like_step(self, action):
        base_env = self.env.unwrapped
        if self.transition_sample_delay > 0:
            time.sleep(self.transition_sample_delay)
        raw_next_obs = base_env.refresh_obs()
        next_obs = self._transform_obs_to_policy_space(raw_next_obs)
        # Reward comes from the outer classifier wrapper, not from teleop.
        reward = np.asarray(0.0, dtype=np.float32)
        done = False
        truncated = False
        transition = {
            "observations": self._prev_obs_for_transition,
            "actions": np.array(action, copy=True),
            "next_observations": next_obs,
            "rewards": reward,
            "dones": done,
            "truncated": truncated,
        }
        info = {
            "intervene_action": np.array(action, copy=True),
            "intervention_mode": True,
            "intervened": True,
            "control_backend": self.servo_backend,
            "transition_sample_delay": self.transition_sample_delay,
            "sampled_transition": transition,
        }
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
        left_motion = self._has_motion_input(self._left["axes"]) or self._left["gripper_toggle_changed"]
        right_motion = self._has_motion_input(self._right["axes"]) or self._right["gripper_toggle_changed"]
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
        return self._transform_action_to_policy(base_action)

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
            left_active = self._has_motion_input(self._left["axes"]) or self._left["gripper_toggle_changed"]
            right_active = self._has_motion_input(self._right["axes"]) or self._right["gripper_toggle_changed"]
            if (not left_active) and (not right_active):
                if not self._servo_running:
                    self.env.unwrapped.refresh_obs()
                    self._target_pose_ref = np.array(self.env.unwrapped.currpos, copy=True)
                    self._start_servo()
                hold_action = self._build_hold_action()
                if (
                    (not self._idle_hold_sent)
                    or self._left["gripper_toggle_changed"]
                    or self._right["gripper_toggle_changed"]
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
            np.asarray(action, dtype=np.float32)
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
