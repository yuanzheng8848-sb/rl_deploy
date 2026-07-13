"""OpenArm-specific Gymnasium wrappers."""
import fcntl
import os
import time

import cv2
import numpy as np
import gymnasium as gym
from scipy.spatial.transform import Rotation as R

from openarm_env.utils.transformations import (
    construct_homogeneous_matrix,
    construct_twist_rotation_matrix,
)
from rl_launcher.wrappers import ChunkingWrapper
from rl_launcher.wrappers.chunking import stack_obs

from openarm_env.envs.openarm_env import integrate_pose_velocity
from openarm_env.camera.local_camera import MODEL_IMAGE_SIZE

try:
    from evdev import InputDevice, ecodes
except Exception:  # pragma: no cover - evdev/hardware dependent
    InputDevice = None
    ecodes = None


def consume_intervention_toggle_requests(device_states):
    """Merge one polling round's toggle requests and clear every device flag."""
    requests = [
        bool(state.pop("intervention_toggle_requested", False))
        for state in device_states
    ]
    return any(requests)


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
        stacked = np.stack(cropped, axis=0)
        return stacked.reshape((*leading_shape, *stacked.shape[1:]))

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


# ===========================================================================
# Policy obs adapter
# ===========================================================================
class OpenArmPolicyObsAdapter:
    """Replay the installed observation wrappers for externally sampled frames."""

    def __init__(self, wrapped_env):
        wrappers = []
        current = wrapped_env
        while hasattr(current, "env"):
            wrappers.append(current)
            current = current.env
        self.wrappers = list(reversed(wrappers))

    def __call__(self, raw_obs):
        obs = raw_obs
        for wrapper in self.wrappers:
            external_transform = getattr(type(wrapper), "transform_external_observation", None)
            if callable(external_transform):
                obs = external_transform(wrapper, obs)
            elif isinstance(wrapper, ChunkingWrapper):
                wrapper.current_obs.append(obs)
                obs = stack_obs(wrapper.current_obs)
            elif isinstance(wrapper, gym.ObservationWrapper):
                obs = wrapper.observation(obs)
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

    def _update_twist_transforms(self, tcp_pose):
        tcp_pose = np.asarray(tcp_pose)
        self.left_transform = construct_twist_rotation_matrix(tcp_pose[0])
        self.right_transform = construct_twist_rotation_matrix(tcp_pose[1])

    def _update_from_obs(self, obs):
        tcp_pose = np.asarray(obs["state"]["tcp_pose"])
        self._update_twist_transforms(tcp_pose)
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
        # These matrices contain rotations only, so transpose is the exact
        # inverse and avoids a generic matrix inversion on every observation.
        tcp_vel[0] = self.left_transform.T @ tcp_vel[0]
        tcp_vel[1] = self.right_transform.T @ tcp_vel[1]
        if self.include_relative_pose:
            tcp_pose = obs["state"]["tcp_pose"]
            tcp_pose[0] = self._transform_pose(tcp_pose[0], self.left_reset_inv)
            tcp_pose[1] = self._transform_pose(tcp_pose[1], self.right_reset_inv)
        return obs

    def transform_external_observation(self, obs):
        tcp_pose = np.asarray(obs["state"]["tcp_pose"])
        self._update_twist_transforms(tcp_pose)
        return self.transform_observation(obs)

    def transform_action(self, action):
        action = np.asarray(action, dtype=np.float32).copy()
        if action.shape[0] >= 14:
            action[:6] = self.left_transform @ action[:6]
            action[7:13] = self.right_transform @ action[7:13]
        return action

    def transform_action_inv(self, action):
        action = np.asarray(action, dtype=np.float32).copy()
        if action.shape[0] >= 14:
            action[:6] = self.left_transform.T @ action[:6]
            action[7:13] = self.right_transform.T @ action[7:13]
        return action

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._update_from_obs(obs)
        return self.transform_observation(obs), info

    def step(self, action):
        transformed_action = self.transform_action(action)
        obs, reward, done, truncated, info = self.env.step(transformed_action)
        tcp_pose = np.asarray(obs["state"]["tcp_pose"])
        self._update_twist_transforms(tcp_pose)
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
        if not np.isfinite(self.control_hz) or self.control_hz <= 0.0:
            raise ValueError(f"control_hz must be finite and positive, got {control_hz!r}")
        self.control_dt = 1.0 / self.control_hz
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
        self._last_obs = None
        self._prev_obs_for_transition = None
        self._target_pose_ref = None
        self._last_intervention_mode = False
        self._intervention_mode = False
        self._idle_hold_sent = False
        self._init_devices()

    @property
    def intervention_active(self):
        return bool(self._intervention_mode)

    def _make_device_state(self, label):
        return {
            "label": label,
            "axes": {"x": 0.0, "y": 0.0, "z": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
            "button_state": {"gripper_close": False},
            "event_path": None,
            "dev": None,
            "enabled": False,
            "gripper_toggle_changed": False,
            "relative_axes": set(),
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

    def _transform_action_to_policy(self, action):
        transformed = np.array(action, copy=True)
        relative_wrapper = self._find_wrapper(self.env, DualRelativeFrame)
        if relative_wrapper is not None:
            transformed = relative_wrapper.transform_action_inv(transformed)
        return transformed

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

    def _update_target_pose_ref(self, base_action, dt):
        base_env = self.env.unwrapped
        self._target_pose_ref = base_env.clip_safety_box(
            integrate_pose_velocity(
                self._target_pose_ref,
                base_action,
                base_env.action_velocity_scale,
                dt,
            )
        )
        base_env.target_pose_ref = np.array(self._target_pose_ref, copy=True)

    def _update_servo_target(self, action):
        base_env = self.env.unwrapped
        base_env._send_pos_command(
            self._target_pose_ref,
            gripper_closed=base_env.gripper_actions_to_closed(action),
        )

    def _sample_env_like_step(self, action):
        base_env = self.env.unwrapped
        raw_next_obs = base_env.refresh_obs()
        next_obs = self._transform_obs_to_policy_space(raw_next_obs)
        base_env.curr_path_length += 1
        truncated = bool(base_env.curr_path_length >= base_env.max_episode_length)
        # Reward comes from the outer classifier wrapper, not from teleop.
        transition = {
            "observations": self._prev_obs_for_transition,
            "actions": np.array(action, copy=True),
            "next_observations": next_obs,
        }
        info = {
            "intervene_action": np.array(action, copy=True),
            "intervention_transition": transition,
        }
        self._prev_obs_for_transition = next_obs
        self._last_obs = next_obs
        return next_obs, 0.0, False, truncated, info

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
                        state["relative_axes"].add(axis)
                    else:
                        state["axes"][axis] = float(event.value)
                        state["relative_axes"].discard(axis)
            elif event.type == ecodes.EV_KEY and int(event.value) == 1:
                if event.code in (ecodes.BTN_0, ecodes.BTN_LEFT):
                    state["intervention_toggle_requested"] = True
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

    def _build_intervene_actions(self):
        """Return the same human command in policy-local and world frames."""
        base_action = np.zeros((14,), dtype=np.float32)
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
        return self._transform_action_to_policy(base_action), base_action

    def _poll_devices_once(self):
        self._poll_one_device(self._left)
        self._poll_one_device(self._right)
        if consume_intervention_toggle_requests((self._left, self._right)):
            self._intervention_mode = not self._intervention_mode
            print(f"[DualSpacemouse] intervention mode: {'ON' if self._intervention_mode else 'OFF'}")

    def _consume_relative_axes(self):
        """Clear per-tick EV_REL deltas while preserving EV_ABS positions."""
        for state in (self._left, self._right):
            for axis in state["relative_axes"]:
                state["axes"][axis] = 0.0

    def _run_intervention_window(self, window_start):
        """Update the servo at high rate and emit at most one policy transition."""
        base_env = self.env.unwrapped
        transition_dt = base_env.control_dt
        window_end = window_start + transition_dt
        integrated_policy_action = np.zeros((14,), dtype=np.float64)
        final_grippers = self._build_hold_action()[[6, 13]]
        had_input = False

        while self._intervention_mode:
            tick_start = time.monotonic()
            remaining = window_end - tick_start
            if remaining <= 0.0:
                break
            tick_dt = min(self.control_dt, remaining)
            left_active = self._has_motion_input(self._left["axes"]) or self._left["gripper_toggle_changed"]
            right_active = self._has_motion_input(self._right["axes"]) or self._right["gripper_toggle_changed"]

            if left_active or right_active:
                policy_action, base_action = self._build_intervene_actions()
                self._update_target_pose_ref(base_action, tick_dt)
                self._update_servo_target(policy_action)
                self._idle_hold_sent = False
                had_input = True
            else:
                base_action = self._build_hold_action()
                policy_action = self._transform_action_to_policy(base_action)
                if not self._idle_hold_sent:
                    base_env.refresh_obs()
                    self._target_pose_ref = np.array(base_env.currpos, copy=True)
                    base_env.target_pose_ref = np.array(self._target_pose_ref, copy=True)
                    self._update_servo_target(policy_action)
                    self._idle_hold_sent = True

            integrated_policy_action[:6] += policy_action[:6] * tick_dt
            integrated_policy_action[7:13] += policy_action[7:13] * tick_dt
            final_grippers = policy_action[[6, 13]]
            self._consume_relative_axes()

            time.sleep(max(0.0, tick_start + tick_dt - time.monotonic()))
            if time.monotonic() >= window_end:
                break
            self._poll_devices_once()

        time.sleep(max(0.0, window_end - time.monotonic()))
        if not had_input:
            raw_obs = base_env.refresh_obs()
            self._last_obs = self._transform_obs_to_policy_space(raw_obs)
            self._prev_obs_for_transition = self._last_obs
            return self._last_obs, 0.0, False, False, {"intervention_idle": True}

        aggregated_action = np.zeros((14,), dtype=np.float32)
        aggregated_action[:6] = integrated_policy_action[:6] / transition_dt
        aggregated_action[7:13] = integrated_policy_action[7:13] / transition_dt
        aggregated_action[[6, 13]] = final_grippers
        aggregated_action = np.clip(
            aggregated_action,
            self.action_space.low,
            self.action_space.high,
        )
        if self.print_raw:
            print(f"[DualSpacemouse] aggregated_action={np.round(aggregated_action, 4).tolist()}")
        return self._sample_env_like_step(aggregated_action)

    def reset(self, **kwargs):
        for state in (self._left, self._right):
            for key in state["axes"]:
                state["axes"][key] = 0.0
            state["gripper_toggle_changed"] = False
        self._intervention_mode = False
        self._idle_hold_sent = False
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self._prev_obs_for_transition = obs
        self._target_pose_ref = np.array(self.env.unwrapped.currpos, copy=True)
        return obs, info

    def step(self, action):
        loop_start = time.monotonic()
        self._poll_devices_once()

        if self._intervention_mode and (not self._last_intervention_mode):
            self.env.unwrapped.refresh_obs()
            self._target_pose_ref = np.array(self.env.unwrapped.currpos, copy=True)
            self.env.unwrapped.target_pose_ref = np.array(self._target_pose_ref, copy=True)
            self._sync_button_state_from_hardware()
            self._idle_hold_sent = False
            if self._prev_obs_for_transition is None:
                self._prev_obs_for_transition = self._last_obs

        if self._intervention_mode:
            self._last_intervention_mode = True
            return self._run_intervention_window(loop_start)

        obs, reward, done, truncated, info = self.env.step(
            np.asarray(action, dtype=np.float32)
        )
        self._consume_relative_axes()
        self._last_obs = obs
        self._prev_obs_for_transition = obs
        self._last_intervention_mode = False
        return obs, reward, done, truncated, info

    def close(self):
        for state in (self._left, self._right):
            dev = state["dev"]
            if dev is not None:
                try:
                    dev.ungrab()
                except OSError:
                    pass
        return self.env.close()
