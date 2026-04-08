#!/usr/bin/env python3

import os
import sys
import ctypes
import fcntl

# --- 强制指定 JAX 使用的 NVIDIA 库路径 ---
# 必须在导入 jax 或其他可能使用它的库之前完成此操作
# 这是为了解决在某些 Conda 环境下 JAX 无法找到 CUDA 库的问题
import site
nvidia_base = os.path.join(site.getsitepackages()[0], "nvidia")
libs = [
    "cublas/lib", "cudnn/lib", "cufft/lib", "cusolver/lib", 
    "cusparse/lib", "nccl/lib", "nvjitlink/lib"
]
for lib in libs:
    path = os.path.join(nvidia_base, lib)
    if os.path.exists(path):
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        # 将 NVIDIA 库路径添加到 LD_LIBRARY_PATH 环境变量中
        os.environ["LD_LIBRARY_PATH"] = f"{path}:{current_ld}"

# 设置 XLA_FLAGS 以帮助 JAX 找到 CUDA 数据目录
os.environ['XLA_FLAGS'] = f"--xla_gpu_cuda_data_dir={nvidia_base}"

# 显式按依赖顺序预加载库
try:
    # 首先预加载 nvJitLink (因为 cusparse 依赖于它)
    nvjitlink_path = os.path.join(nvidia_base, "nvjitlink/lib/libnvJitLink.so.12")
    if os.path.exists(nvjitlink_path):
        ctypes.CDLL(nvjitlink_path)
        print(f"[DEBUG] Successfully preloaded {nvjitlink_path}")
    
    # 然后预加载 cuSPARSE
    cusparse_path = os.path.join(nvidia_base, "cusparse/lib/libcusparse.so.12")
    if os.path.exists(cusparse_path):
        ctypes.CDLL(cusparse_path)
        print(f"[DEBUG] Successfully preloaded {cusparse_path}")
        
    sys.stdout.flush()
except Exception as e:
    print(f"[DEBUG] Failed to preload libraries: {e}")
    sys.stdout.flush()
# ------------------------------------------

import time
from functools import partial

import jax
print(f"DEBUG: JAX Devices: {jax.devices()}")
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import flax.linen as nn
import pickle as pkl
import os
import json
import cv2
import datetime
import shutil

# --- Local Camera Support Imports ---
from pathlib import Path
import threading
import queue
try:
    from evdev import InputDevice, ecodes
except Exception:
    InputDevice = None
    ecodes = None

# Add pyroki to path for RealsenseCamera
sys.path.append(str(Path(__file__).parent.parent / "pyroki"))
try:
    from realsense_camera import RealsenseCamera, OpenCVCamera
except ImportError as e:
    print(f"Failed to import camera modules: {e}")

from openarm_env import OpenArmEnv, DefaultOpenArmConfig

# Keep train-time head camera preprocessing aligned with classifier training data.
# classifier/train_classifier.py trains on cam_2_rgb images and resizes directly to 128x128.
CLASSIFIER_CAM_SUBDIR = "cam_2_rgb"
HEAD_CAMERA_DEVICE = "/dev/video12"
HEAD_CAMERA_WIDTH = 640
HEAD_CAMERA_HEIGHT = 480
MODEL_IMAGE_SIZE = (128, 128)
APPLY_HEAD_CAMERA_CROP = False
EXTRA_FAILURE_IMAGE_DIR = "/home/sj/Desktop/zy/moqi_workspace/rl_deploy/classifier/extra_failure_images"
CLASSIFIER_JPEG_QUALITY = 90

class LocalOpenArmEnv(OpenArmEnv):
    """
    Subclass of OpenArmEnv that initializes and reads from local cameras
    instead of relying on the server to send images.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # OpenArmEnv.__init__ returns early if fake_env=True, skipping camera init.
        # We enforce camera initialization here for LocalOpenArmEnv to support Mock Cameras.
        if self.fake_env:
             print("[LocalOpenArmEnv] Enforcing local camera init for FAKE/MOCK mode.")
             self.init_cameras(None)

    def reset(self, **kwargs):
        return super().reset(**kwargs)

    def init_cameras(self, config):
        self.cameras = []
        self.latest_images_raw = {}
        # Configuration from main_v5_record_velocity.py
        # Left: 150622074105, Right: 236422072385
        # Head: /dev/video18
        
        # Import MockCamera if needed
        try:
             from mock_hardware import MockCamera
        except ImportError:
             MockCamera = None

        # 1. Left Camera
        # 1. Left Camera
        try:
            if self.fake_env:
                cam_left = MockCamera(width=640, height=480, fps=30)
            else:
                cam_left = RealsenseCamera(
                    device_id="150622074105",
                    enable_depth=False,
                    width=640,
                    height=480,
                    fps=30
                )
            self.cameras.append(("image_left", cam_left))
            print("Initialized Left Camera (150622074105)")
        except Exception as e:
            print(f"Failed to init Left Camera: {e}")

        # 2. Right Camera
        try:
            if self.fake_env:
                cam_right = MockCamera(width=640, height=480, fps=30)
            else:
                cam_right = RealsenseCamera(
                    device_id="236422072385",
                    enable_depth=False,
                    width=640,
                    height=480,
                    fps=30
                )
            self.cameras.append(("image_right", cam_right))
            print("Initialized Right Camera (236422072385)")
        except Exception as e:
            print(f"Failed to init Right Camera: {e}")

        # 3. Head Camera (Primary)
        try:
            # Note: Exposure 150 as in reference
            if self.fake_env:
                cam_head = MockCamera(width=HEAD_CAMERA_WIDTH, height=HEAD_CAMERA_HEIGHT, fps=30)
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
        except Exception as e:
            print(f"Failed to init Head Camera: {e}")

        # Start Capture Thread
        self.stop_event = threading.Event()
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _capture_loop(self):
        while not self.stop_event.is_set():
            for name, cam in self.cameras:
                try:
                    # get_data returns (color, depth) or just color depending on implementation
                    # RealsenseCamera.get_data returns [color, depth] (LIST)
                    # OpenCVCamera.get_data returns frame (ndarray) or None
                    
                    img = cam.get_data(viz=False)
                    
                    # Check if Realsense (returns list [color, depth])
                    is_realsense = isinstance(img, (list, tuple))
                    
                    # If list or tuple (color, depth), take color
                    if is_realsense:
                        img = img[0]
                    
                    if img is not None:
                        if name == "image_primary" and APPLY_HEAD_CAMERA_CROP:
                            # Keep disabled for classifier alignment.
                            h, w = img.shape[:2]
                            crop_h_ratio = 0.35
                            crop_w_ratio = 0.35
                            cy, cx = h // 2, w // 2
                            half_h = int(h * crop_h_ratio / 2)
                            half_w = int(w * crop_w_ratio / 2)
                            img = img[cy - half_h : cy + half_h, cx - half_w : cx + half_w]

                        # Keep a full-resolution RGB copy for classifier-pipeline alignment.
                        if is_realsense:
                            img_rgb_full = img
                        else:
                            img_rgb_full = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        self.latest_images_raw[name] = img_rgb_full

                        # Resize to 128x128 for RL
                        img_resized = cv2.resize(img, MODEL_IMAGE_SIZE)
                        
                        if is_realsense:
                            # Realsense returns RGB, use as is
                            img_rgb = img_resized
                        else:
                            # OpenCV returns BGR, convert to RGB
                            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                        
                        self.latest_images[name] = img_rgb
                    else:
                        # Debug print for missing frame
                        if self.cycle_count % 100 == 0: # Print occasionally
                             print(f"[Warn] Camera {name} returned None frame")
                        
                except Exception as e:
                    print(f"[Error] Capture error {name}: {e}")
                    pass
            

                    
            time.sleep(0.01) # Low latency capture (Perception > Control Hz)

    def step(self, action):
        # Enforce 10Hz Control Frequency
        # We rely on the parent class OpenArmEnv.step() to handle the timing
        # via the 'hz' parameter (set to 10). It calculates dt and sleeps accordingly.
        # We DO NOT sleep here to avoid double-sleeping (which would result in 5Hz).
        return super().step(action)

    def close(self):
        if hasattr(self, "stop_event"):
            self.stop_event.set()
        if hasattr(self, "capture_thread"):
            self.capture_thread.join(timeout=1.0)
        super().close()

import gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics

# 导入 SERL 框架组件
from serl_launcher.agents.continuous.drq import DrQAgent
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.wrappers.chunking import ChunkingWrapper, stack_obs
from serl_launcher.utils.train_utils import concat_batches

# 导入 AgentLace 用于分布式训练通信
from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_drq_agent,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

class ClassifierRewardWrapper(gym.Wrapper):
    def __init__(self, env, classifier_ckpt_path=None, reward_image_key="image_primary", enable_animation=False):
        super().__init__(env)
        self.reward_image_key = reward_image_key
        self.enable_animation = enable_animation
        self.current_stage = "stage1"
        
        # Determine sampling rng
        rng = jax.random.PRNGKey(0)
        rng, key = jax.random.split(rng)
        
        # Create a dummy sample to initialize the classifier
        # The classifier implies input key "image_0" from training
        # We must map our reward_image_key to "image_0"
        dummy_img = jnp.zeros((1, 128, 128, 3), dtype=jnp.float32)
        sample = {"image_0": dummy_img, "state": jnp.zeros((1, 14))}
        
        if classifier_ckpt_path and os.path.exists(classifier_ckpt_path):
            print(f"[RewardWrapper] Loading classifier from {classifier_ckpt_path}...")
            self.classifier_func = load_classifier_func(
                key=key,
                sample=sample,
                image_keys=["image_0"],
                checkpoint_path=classifier_ckpt_path,
            )
            print("[RewardWrapper] Classifier loaded.")
        else:
            print("[RewardWrapper] No classifier checkpoint provided or found. Classifier reference will be unavailable.")
            self.classifier_func = None

        # Manual supervision flags.
        # Reward for learning is driven only by keyboard input:
        # SPACE => failure/reset, ENTER => success.
        self.manual_fail_requested = False
        self.manual_success_requested = False
        
        import threading
        self.listener_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self.listener_thread.start()
        print("[RewardWrapper] Keyboard controls: <SPACE> fail/reset, <ENTER> success.")

    def set_stage(self, stage_name):
        self.current_stage = str(stage_name)
        print(f"[RewardWrapper] Active training stage: {self.current_stage}")

    def _get_reward_images(self):
        raw_img = None
        raw_img_full = None
        try:
            base_env = self.env.unwrapped
            if hasattr(base_env, "latest_images"):
                raw_img = base_env.latest_images.get(self.reward_image_key)
            if hasattr(base_env, "latest_images_raw"):
                raw_img_full = base_env.latest_images_raw.get(self.reward_image_key)
        except Exception as e:
            print(f"[RewardWrapper] Error accessing base env: {e}")
        return raw_img, raw_img_full

    def _evaluate_current_reward(self, raw_img, raw_img_full):
        reward = 0.0
        classifier_prob = 0.0
        classifier_reward = 0.0
        aligned_img = None

        source_img = raw_img_full if raw_img_full is not None else raw_img
        if source_img is not None:
            try:
                decoded_full_rgb, _ = self._encode_decode_jpeg90_rgb(source_img)
                aligned_img = cv2.resize(decoded_full_rgb, MODEL_IMAGE_SIZE)
            except Exception as e:
                print(f"[RewardWrapper] Error aligning image pipeline: {e}")
                aligned_img = cv2.resize(source_img, MODEL_IMAGE_SIZE) if source_img is not None else None

        if self.classifier_func is not None and aligned_img is not None:
            try:
                img_jax = jnp.asarray(aligned_img, dtype=jnp.float32)[None, ...]
                logits = self.classifier_func({"image_0": img_jax})
                classifier_prob = float(nn.sigmoid(logits).item())
                classifier_reward = classifier_prob
            except Exception as e:
                print(f"[RewardWrapper] Error running classifier: {e}")

        manual_termination = False
        manual_success = False
        manual_failure = False

        if self.manual_fail_requested:
            self.manual_fail_requested = False
            self.manual_success_requested = False
            reward = 0.0
            manual_termination = True
            manual_failure = True
            print("[RewardWrapper] Episode marked as failure. Waiting actor loop to reset env.")
        elif self.manual_success_requested:
            self.manual_success_requested = False
            self.manual_fail_requested = False
            reward = 1.0
            manual_termination = True
            manual_success = True
            print("[RewardWrapper] Episode marked as success. Waiting actor loop to reset env.")

        info = {
            "classifier_prob": classifier_prob,
            "classifier_reward": classifier_reward,
            "keyboard_reward": reward,
            "manual_termination": manual_termination,
            "manual_success": manual_success,
            "manual_failure": manual_failure,
            "success": manual_success,
            "active_stage": self.current_stage,
        }
        return np.asarray(reward, dtype=np.float32), manual_termination, info

    def evaluate_transition(self, obs):
        raw_img, raw_img_full = self._get_reward_images()
        reward, done, info = self._evaluate_current_reward(raw_img, raw_img_full)

        if FLAGS.render and raw_img is not None:
            cv2.imshow("Right Arm Camera (Raw)", cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR))
            show_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
            text = (
                f"P: {info['classifier_prob']:.3f} "
                f"C: {info['classifier_reward']:.3f} "
                f"K: {float(reward):.3f}"
            )
            if info["manual_success"]:
                status_text = "MANUAL SUCCESS"
                color = (0, 255, 0)
            elif info["manual_failure"]:
                status_text = "MANUAL FAILURE"
                color = (0, 0, 255)
            else:
                status_text = "RUNNING"
                color = (0, 255, 255)
            cv2.putText(show_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(show_img, status_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(show_img, f"Stage: {self.current_stage}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("Reward Monitor", show_img)
            cv2.waitKey(1)

        return obs, reward, bool(done), False, info

    def _encode_decode_jpeg90_rgb(self, img_rgb: np.ndarray):
        """
        Align image statistics with DataRecorder pipeline:
        RGB -> BGR -> JPEG(quality=90) -> decode BGR -> RGB
        Returns (decoded_rgb, encoded_jpeg_bytes).
        """
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        ok, enc = cv2.imencode(
            ".jpg",
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, CLASSIFIER_JPEG_QUALITY],
        )
        if not ok:
            return img_rgb, None
        dec_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if dec_bgr is None:
            return img_rgb, None
        dec_rgb = cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2RGB)
        return dec_rgb, enc

    def _keyboard_listener(self):
        """Background thread: SPACE marks failure, ENTER marks success."""
        from pynput import keyboard
        
        def on_press(key):
            try:
                if key == keyboard.Key.space:
                    self.manual_fail_requested = True
                    self.manual_success_requested = False
                    print(f"\n[RewardWrapper] SPACE pressed during {self.current_stage}. Marking failure.")
                elif key == keyboard.Key.enter:
                    self.manual_success_requested = True
                    self.manual_fail_requested = False
                    print(f"\n[RewardWrapper] ENTER pressed during {self.current_stage}. Marking success.")
            except Exception:
                pass
        
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

    def reset(self, **kwargs):
        self._current_step = 0
        self.manual_fail_requested = False
        self.manual_success_requested = False
        return self.env.reset(**kwargs)

    def step(self, action):
        self._current_step += 1
        obs, rew, done, truncated, info = self.env.step(action)
        raw_img, raw_img_full = self._get_reward_images()
        reward, manual_done, reward_info = self._evaluate_current_reward(raw_img, raw_img_full)
        done = bool(done or manual_done)
        info.update(reward_info)

        # Visualization
        if FLAGS.render and raw_img is not None:
            # 1. Show raw image in a separate window
            cv2.imshow("Right Arm Camera (Raw)", cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR))
            
            # 2. Show annotated image in Reward Monitor
            show_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
            text = (
                f"P: {info['classifier_prob']:.3f} "
                f"C: {info['classifier_reward']:.3f} "
                f"K: {float(reward):.3f}"
            )
            if info["manual_success"]:
                status_text = "MANUAL SUCCESS"
                color = (0, 255, 0)
            elif info["manual_failure"]:
                status_text = "MANUAL FAILURE"
                color = (0, 0, 255)
            else:
                status_text = "RUNNING"
                color = (0, 255, 255)
            cv2.putText(show_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(show_img, status_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(show_img, f"Stage: {self.current_stage}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("Reward Monitor", show_img)
            cv2.waitKey(1)
        
        return obs, reward, done, truncated, info

class EvdevSpacemouseIntervention(gym.ActionWrapper):
    """
    Evdev-based 3D mouse intervention wrapper.
    Behavior is aligned with rl_deploy/test/3dx/test_3dx_operation.py.
    Arm pose and gripper are both overridden by the 3D mouse only while
    intervention mode is active. Outside intervention mode, the policy controls
    all action dimensions and the environment applies thresholded gripper
    execution.
    """

    def __init__(
        self,
        env,
        event_path="/dev/input/event7",
        trans_denom=420.0,
        rot_denom=380.0,
        deadzone=0.08,
        rot_deadzone=0.16,
        ee_x="x",
        ee_y="-y",
        ee_z="-z",
        realtime_servo=True,
        control_hz=80.0,
        servo_backend="analytic",
        servo_hz=100.0,
        servo_trans_step=0.004,
        servo_rot_step=0.012,
        servo_gripper_step=0.05,
        gripper_open_cmd=-0.95,
        gripper_close_cmd=0.0,
        print_raw=False,
    ):
        super().__init__(env)
        self.event_path = event_path
        self.trans_denom = float(trans_denom)
        self.rot_denom = float(rot_denom)
        self.deadzone = float(deadzone)
        self.rot_deadzone = float(rot_deadzone)
        self.ee_x = ee_x
        self.ee_y = ee_y
        self.ee_z = ee_z
        self.realtime_servo = bool(realtime_servo)
        self.control_hz = float(control_hz)
        self.servo_backend = str(servo_backend)
        self.servo_hz = float(servo_hz)
        self.servo_trans_step = float(servo_trans_step)
        self.servo_rot_step = float(servo_rot_step)
        self.servo_gripper_step = float(servo_gripper_step)
        self.gripper_open_cmd = float(gripper_open_cmd)
        self.gripper_close_cmd = float(gripper_close_cmd)
        self.print_raw = print_raw

        self.axes = {"x": 0.0, "y": 0.0, "z": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}
        # Button mode:
        # - BTN_0 / BTN_LEFT: toggle intervention mode on each press
        # - BTN_1 / BTN_RIGHT: toggle gripper open/close on each press
        self.button_state = {"gripper_close": False}
        self._enabled = False
        self._dev = None
        self._intervention_mode = False
        self._gripper_closed = False
        self._gripper_toggle_changed = False
        self._last_obs = None
        self._last_servo_mode = False
        self._servo_running = False
        self._idle_hold_sent = False
        self._target_pose_ref = None
        self._prev_obs_for_transition = None
        self._last_sample_ts = 0.0

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

        self._init_device()

    def _init_device(self):
        if InputDevice is None or ecodes is None:
            print("[WARN] evdev unavailable; disabling spacemouse intervention.")
            return
        
        def has_3dx_axes(dev):
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

        def auto_detect_event_path():
            try:
                from evdev import list_devices
            except Exception:
                return None

            candidates = []
            for path in list_devices():
                try:
                    dev = InputDevice(path)
                except (PermissionError, OSError):
                    continue

                name = (dev.name or "").lower()
                name_hit = any(
                    k in name
                    for k in (
                        "3dconnexion",
                        "spacemouse",
                        "space mouse",
                        "spacenavigator",
                        "space navigator",
                    )
                )
                axes_hit = has_3dx_axes(dev)
                if name_hit or axes_hit:
                    score = (100 if name_hit else 0) + (10 if axes_hit else 0)
                    candidates.append((score, path, dev.name))

            if not candidates:
                return None
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1], candidates[0][2]

        event_path = self.event_path
        if str(self.event_path).strip().lower() == "auto":
            detected = auto_detect_event_path()
            if detected is None:
                print("[WARN] spacemouse auto-detect failed; disabling spacemouse intervention.")
                return
            event_path, dev_name = detected
            print(f"[Spacemouse] auto-detected device: {event_path} ({dev_name})")

        try:
            self._dev = InputDevice(event_path)
            try:
                self._dev.grab()
            except OSError:
                print("[WARN] Could not grab spacemouse device; continuing without exclusive access.")
            flags = fcntl.fcntl(self._dev.fd, fcntl.F_GETFL)
            fcntl.fcntl(self._dev.fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            self._enabled = True
            self.event_path = event_path
            print(
                f"[Spacemouse] evdev enabled: {self._dev.name} ({self.event_path}), "
                f"mapping X<-{self.ee_x}, Y<-{self.ee_y}, Z<-{self.ee_z}"
            )
        except Exception as e:
            self._enabled = False
            print(f"[WARN] Failed to open spacemouse device {self.event_path}: {e}")

    @staticmethod
    def _apply_deadzone(v, dz):
        return 0.0 if abs(v) < dz else v

    @staticmethod
    def _parse_axis_spec(spec):
        s = str(spec).strip().lower()
        if s.startswith("-"):
            return s[1:], -1.0
        return s, 1.0

    def _device_to_ee_translation(self):
        def get_norm(axis_spec):
            name, sign = self._parse_axis_spec(axis_spec)
            raw = self.axes.get(name, 0.0)
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

    def _post_server_json(self, route, payload, timeout):
        base_env = self.env.unwrapped
        session = getattr(base_env, "session", None)
        url = getattr(base_env, "url", None)
        if session is None or url is None:
            raise RuntimeError("Base env session/url unavailable for servo control")
        return session.post(url.rstrip("/") + route, json=payload, timeout=timeout)

    def _get_reward_wrapper(self):
        return self._find_wrapper(self.env, ClassifierRewardWrapper)

    def _get_chunking_wrapper(self):
        return self._find_wrapper(self.env, ChunkingWrapper)

    def _transform_obs_to_policy_space(self, raw_obs):
        obs = raw_obs
        relative_wrapper = self._find_wrapper(self.env, RelativeFrame)
        if relative_wrapper is not None:
            relative_wrapper.adjoint_matrix = construct_adjoint_matrix(obs["state"]["tcp_pose"])
            obs = relative_wrapper.transform_observation(obs)

        quat_wrapper = self._find_wrapper(self.env, Quat2EulerWrapper)
        if quat_wrapper is not None:
            obs = quat_wrapper.observation(obs)

        serl_wrapper = self._find_wrapper(self.env, SERLObsWrapper)
        if serl_wrapper is not None:
            obs = serl_wrapper.observation(obs)

        chunking_wrapper = self._get_chunking_wrapper()
        if chunking_wrapper is not None:
            chunking_wrapper.current_obs.append(obs)
            obs = stack_obs(chunking_wrapper.current_obs)

        return obs

    def _transform_action_to_base(self, action):
        transformed = np.array(action, copy=True)
        relative_wrapper = self._find_wrapper(self.env, RelativeFrame)
        if relative_wrapper is not None:
            transformed = relative_wrapper.transform_action(transformed)
        return transformed

    def _transform_action_to_policy(self, action):
        transformed = np.array(action, copy=True)
        relative_wrapper = self._find_wrapper(self.env, RelativeFrame)
        if relative_wrapper is not None:
            transformed = relative_wrapper.transform_action_inv(transformed)
        return transformed

    def _get_active_arm_indices(self):
        arm = getattr(self.env.unwrapped, "arm", "both")
        if arm == "both":
            return [0, 1]
        if arm == "left":
            return [0]
        return [1]

    def _desired_gripper_cmds(self, action):
        base_env = self.env.unwrapped
        gripper_cmds = [float(x) for x in base_env.curr_gripper_pos]
        for arm_idx in self._get_active_arm_indices():
            gripper_cmds[arm_idx] = (
                self.gripper_close_cmd if self.button_state["gripper_close"] else self.gripper_open_cmd
            )
        return gripper_cmds

    def _is_analytic_backend(self):
        return self.realtime_servo and self.servo_backend == "analytic"

    def _gripper_needs_keepalive(self, action):
        base_env = self.env.unwrapped
        desired = self._desired_gripper_cmds(action)
        threshold = max(self.servo_gripper_step, 0.02)
        for arm_idx in self._get_active_arm_indices():
            if abs(float(base_env.curr_gripper_pos[arm_idx]) - float(desired[arm_idx])) > threshold:
                return True
        return False

    def _start_servo(self):
        if (not self.realtime_servo) or self._servo_running:
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
            "arm": getattr(base_env, "arm", "both"),
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
        self._idle_hold_sent = False

    def _update_target_pose_ref(self, action):
        base_env = self.env.unwrapped
        arm = getattr(base_env, "arm", "both")
        base_action = self._transform_action_to_base(action)
        updated = np.array(self._target_pose_ref, copy=True)
        for arm_idx in self._get_active_arm_indices():
            arm_action = base_action if arm != "both" else base_action[arm_idx * 7 : (arm_idx + 1) * 7]
            updated[arm_idx, :3] += arm_action[:3] * base_env.action_scale[0]
            rot_curr = R.from_quat(updated[arm_idx, 3:])
            rot_delta = R.from_euler("xyz", arm_action[3:6] * base_env.action_scale[1])
            updated[arm_idx, 3:] = (rot_delta * rot_curr).as_quat()
        self._target_pose_ref = updated

    def _update_servo_target(self, action):
        payload = {
            "arr": np.asarray(self._target_pose_ref, dtype=np.float32).tolist(),
            "gripper": self._desired_gripper_cmds(action),
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
                "control_backend": self.servo_backend if self.realtime_servo else "env_step",
                "sampled_transition": transition,
            }
        )
        self._prev_obs_for_transition = next_obs
        self._last_obs = next_obs
        return next_obs, reward, done, truncated, info

    def _poll_events(self):
        if not self._enabled or self._dev is None:
            return False
        got_any = False
        self._gripper_toggle_changed = False
        while True:
            try:
                ev = self._dev.read_one()
            except BlockingIOError:
                return got_any
            except OSError:
                return got_any
            if ev is None:
                return got_any

            got_any = True
            if ev.type in (ecodes.EV_ABS, ecodes.EV_REL):
                axis = self.axis_codes.get(ev.code)
                if axis is not None:
                    if ev.type == ecodes.EV_REL:
                        self.axes[axis] += float(ev.value)
                    else:
                        self.axes[axis] = float(ev.value)
            elif ev.type == ecodes.EV_KEY:
                # BTN_0 / BTN_LEFT: toggle intervention on key-down only.
                if ev.code in (ecodes.BTN_0, ecodes.BTN_LEFT):
                    if int(ev.value) == 1:
                        self._intervention_mode = not self._intervention_mode
                        mode = "ON" if self._intervention_mode else "OFF"
                        print(f"[Spacemouse] intervention mode: {mode}")
                # BTN_1 / BTN_RIGHT: toggle gripper open/close on key-down only.
                elif ev.code in (ecodes.BTN_1, ecodes.BTN_RIGHT):
                    if int(ev.value) == 1:
                        self._gripper_closed = not self._gripper_closed
                        self.button_state["gripper_close"] = self._gripper_closed
                        self._gripper_toggle_changed = True
                        g = "CLOSE" if self._gripper_closed else "OPEN"
                        print(f"[Spacemouse] gripper mode: {g}")
        return got_any

    def _has_motion_input(self):
        dx, dy, dz = self._device_to_ee_translation()
        if abs(dx) > self.deadzone or abs(dy) > self.deadzone or abs(dz) > self.deadzone:
            return True
        rx = abs(np.clip(self.axes["rx"] / self.rot_denom, -1.0, 1.0))
        ry = abs(np.clip(-self.axes["ry"] / self.rot_denom, -1.0, 1.0))
        rz = abs(np.clip(-self.axes["rz"] / self.rot_denom, -1.0, 1.0))
        if rx > self.rot_deadzone or ry > self.rot_deadzone or rz > self.rot_deadzone:
            return True
        return False

    def _build_intervene_action(self, policy_action):
        base_action = np.asarray(policy_action, dtype=np.float32).copy()
        if base_action.shape[0] < 6:
            return base_action

        dx, dy, dz = self._device_to_ee_translation()
        dx = self._apply_deadzone(dx, self.deadzone)
        dy = self._apply_deadzone(dy, self.deadzone)
        dz = self._apply_deadzone(dz, self.deadzone)
        rx = self._apply_deadzone(np.clip(self.axes["rx"] / self.rot_denom, -1.0, 1.0), self.rot_deadzone)
        ry = self._apply_deadzone(np.clip(-self.axes["ry"] / self.rot_denom, -1.0, 1.0), self.rot_deadzone)
        rz = self._apply_deadzone(np.clip(-self.axes["rz"] / self.rot_denom, -1.0, 1.0), self.rot_deadzone)
        base_action[:6] = np.array([dx, dy, dz, rx, ry, rz], dtype=np.float32)
        manual_gripper = 1.0 if self.button_state["gripper_close"] else -1.0
        arm = getattr(self.env.unwrapped, "arm", "both")
        if arm == "both":
            for arm_idx in self._get_active_arm_indices():
                start = arm_idx * 7
                if base_action.shape[0] >= start + 7:
                    base_action[start + 6] = manual_gripper
        elif base_action.shape[0] >= 7:
            base_action[6] = manual_gripper

        return self._transform_action_to_policy(base_action)

    def reset(self, **kwargs):
        # Reset intervention state at episode boundaries.
        for k in self.axes:
            self.axes[k] = 0.0
        self.button_state["gripper_close"] = self._gripper_closed
        self._stop_servo()
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self._prev_obs_for_transition = obs
        self._last_sample_ts = time.time()
        self._idle_hold_sent = False
        self._target_pose_ref = np.array(self.env.unwrapped.currpos, copy=True)
        return obs, info

    def step(self, action):
        loop_start = time.time()
        got_event = self._poll_events()
        if not got_event:
            # REL 设备在无事件时清零，避免残留输入
            for k in self.axes:
                self.axes[k] = 0.0

        if self._intervention_mode and (not self._last_servo_mode):
            self.env.unwrapped.refresh_obs()
            self._target_pose_ref = np.array(self.env.unwrapped.currpos, copy=True)
            self._last_sample_ts = time.time()
            if self._prev_obs_for_transition is None:
                self._prev_obs_for_transition = self._last_obs

        if self._intervention_mode:
            has_motion = self._has_motion_input()
            if (not has_motion) and (not self._gripper_toggle_changed):
                if self.realtime_servo:
                    if self._is_analytic_backend():
                        if not self._servo_running:
                            self.env.unwrapped.refresh_obs()
                            self._target_pose_ref = np.array(self.env.unwrapped.currpos, copy=True)
                            self._start_servo()
                        hold_action = np.array(
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 if self.button_state["gripper_close"] else -1.0],
                            dtype=np.float32,
                        )
                        if not self._idle_hold_sent:
                            self.env.unwrapped.refresh_obs()
                            self._target_pose_ref = np.array(self.env.unwrapped.currpos, copy=True)
                            self._update_servo_target(hold_action)
                            self._idle_hold_sent = True
                        elif self._gripper_needs_keepalive(hold_action):
                            self.env.unwrapped.refresh_obs()
                            self._update_servo_target(hold_action)
                    elif self._servo_running:
                        self.env.unwrapped.refresh_obs()
                        self._target_pose_ref = np.array(self.env.unwrapped.currpos, copy=True)
                        self._stop_servo()
                        self.env.unwrapped.refresh_obs()
                        self._target_pose_ref = np.array(self.env.unwrapped.currpos, copy=True)
                idle_info = {
                    "intervention_idle": True,
                    "intervention_mode": True,
                    "control_backend": self.servo_backend if self.realtime_servo else "env_step",
                }
                if self._last_obs is None:
                    self._last_obs, _ = self.env.reset()
                if self.realtime_servo:
                    dt = time.time() - loop_start
                    time.sleep(max(0.0, (1.0 / self.control_hz) - dt))
                self._last_servo_mode = self._intervention_mode
                return self._last_obs, 0.0, False, False, idle_info

            chosen_action = self._build_intervene_action(action)
            self._idle_hold_sent = False
            if self.print_raw:
                print(
                    f"[Spacemouse] mode={'ON' if self._intervention_mode else 'OFF'} "
                    f"[Spacemouse] raw xyz=({self.axes['x']:+.0f},{self.axes['y']:+.0f},{self.axes['z']:+.0f}) "
                    f"rpy=({self.axes['rx']:+.0f},{self.axes['ry']:+.0f},{self.axes['rz']:+.0f}) "
                    f"btn2_close={int(self.button_state['gripper_close'])} "
                    f"-> action={np.round(np.asarray(chosen_action), 4).tolist()}"
                )
            if self.realtime_servo:
                if not self._servo_running:
                    self.env.unwrapped.refresh_obs()
                    self._target_pose_ref = np.array(self.env.unwrapped.currpos, copy=True)
                    self._start_servo()
                    self.env.unwrapped.refresh_obs()
                    self._target_pose_ref = np.array(self.env.unwrapped.currpos, copy=True)
                    self._update_servo_target(
                        np.array(
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 if self.button_state["gripper_close"] else -1.0],
                            dtype=np.float32,
                        )
                    )
                self._update_target_pose_ref(chosen_action)
                self._update_servo_target(chosen_action)
                self._last_sample_ts = time.time()
                obs, rew, done, truncated, info = self._sample_env_like_step(chosen_action)
                dt = time.time() - loop_start
                time.sleep(max(0.0, (1.0 / self.control_hz) - dt))
                self._last_servo_mode = self._intervention_mode
                return obs, rew, done, truncated, info

            obs, rew, done, truncated, info = self.env.step(chosen_action)
            self._last_obs = obs
            self._prev_obs_for_transition = obs
            info["intervene_action"] = chosen_action
            info["intervened"] = True
            info["intervention_mode"] = True
            info["control_backend"] = "env_step"
            self._last_servo_mode = self._intervention_mode
            return obs, rew, done, truncated, info

        if self._servo_running:
            self.env.unwrapped.refresh_obs()
            self._target_pose_ref = np.array(self.env.unwrapped.currpos, copy=True)
            self._stop_servo()
            self.env.unwrapped.refresh_obs()
            self._target_pose_ref = np.array(self.env.unwrapped.currpos, copy=True)

        chosen_action = np.asarray(action, dtype=np.float32).copy()
        obs, rew, done, truncated, info = self.env.step(chosen_action)
        self._last_obs = obs
        self._prev_obs_for_transition = obs
        self._last_servo_mode = self._intervention_mode
        return obs, rew, done, truncated, info

    def close(self):
        self._stop_servo()
        if self._dev is not None:
            try:
                self._dev.ungrab()
            except OSError:
                pass
        return self.env.close()

# 本地导入 (自定义环境和 Wrappers)
from openarm_env import OpenArmEnv, DefaultOpenArmConfig
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.wrappers import Quat2EulerWrapper
from franka_env.utils.transformations import construct_homogeneous_matrix, construct_adjoint_matrix
from scipy.spatial.transform import Rotation as R

FLAGS = flags.FLAGS

# 定义命令行参数
flags.DEFINE_string("env", "OpenArmEnv", "环境名称")
flags.DEFINE_string("arm", "right", "控制哪个机械臂: 'left', 'right', 或 'both'")
flags.DEFINE_integer("hz", 5, "环境控制频率（Hz）")
flags.DEFINE_string("agent", "drq", "Agent 名称")
flags.DEFINE_string("exp_name", "forward_reach_10cm", "实验名称，用于 wandb 日志")
flags.DEFINE_integer("max_traj_length", 300, "最大轨迹长度")
flags.DEFINE_integer("seed", 42, "随机种子")
flags.DEFINE_bool("save_model", False, "是否保存模型")
flags.DEFINE_integer("critic_actor_ratio", 4, "Critic 与 Actor 的更新比例 (Critic 更新次数 / Actor 更新次数)")

flags.DEFINE_integer("max_steps", 10000, "最大训练步数")
flags.DEFINE_integer("replay_buffer_capacity", 50000, "Replay buffer 容量")
flags.DEFINE_integer("batch_size", 256, "Batch 大小")

flags.DEFINE_integer("random_steps", 50, "随机动作采样步数 (Warmup)")
flags.DEFINE_integer("training_starts", 300, "开始训练的步数")
flags.DEFINE_integer("steps_per_update", 30, "每隔多少步更新一次服务器 (Actor -> Learner)")

flags.DEFINE_integer("log_period", 10, "日志记录周期")
flags.DEFINE_integer("eval_period", 2000, "评估周期")

# 标志位：指示当前进程是 Learner 还是 Actor
flags.DEFINE_boolean("learner", False, "是 Learner 还是 Trainer")
flags.DEFINE_boolean("render", False, "Enable visualization of camera feed and reward")
flags.DEFINE_boolean("actor", False, "是 Learner 还是 Trainer")
flags.DEFINE_boolean("enable_success_animation", False, "Enable robot celebration animation on success")

flags.DEFINE_string("ip", "localhost", "Learner 的 IP 地址")
# "small" 是 4 层卷积网络，"resnet" 和 "mobilenet" 是冻结权重的预训练网络
flags.DEFINE_string("encoder_type", "resnet-pretrained", "编码器类型")
flags.DEFINE_string("demo_path", "/home/sj/Desktop/zy/moqi_workspace/rl_deploy/demo/merged_demos.pkl", "自定义演示数据路径（当 demo_pkl_variant=custom 时使用）")
flags.DEFINE_enum(
    "demo_pkl_variant",
    "v1",
    ["v1", "v2", "custom"],
    "选择加载的 demo pkl：v1=merged_demos.pkl，v2=merged_demos_2.pkl，custom=使用 demo_path。",
)
flags.DEFINE_boolean(
    "demo_drop_over_limit_transitions",
    False,
    "加载 demo 时丢弃超出 OpenArmEnv 单步动作范围的 transition。",
)
flags.DEFINE_integer("checkpoint_period", 200, "保存 Checkpoint 的周期")
flags.DEFINE_string("checkpoint_path", "/home/sj/Desktop/zy/moqi_workspace/rl_deploy/checkpoints", "保存 Checkpoint 的路径")
flags.DEFINE_boolean("enable_two_stage_training", True, "启用两阶段训练：stage1 成功后切到 stage2。")
flags.DEFINE_enum(
    "training_stage",
    "stage1",
    ["stage1", "stage2"],
    "Learner 模式下当前进程负责训练哪个阶段。",
)
flags.DEFINE_string(
    "stage1_checkpoint_path",
    "/home/sj/Desktop/zy/moqi_workspace/rl_deploy/checkpoints",
    "第一阶段 checkpoint 路径。",
)
flags.DEFINE_string(
    "stage2_checkpoint_path",
    "/home/sj/Desktop/zy/moqi_workspace/rl_deploy/checkpoints_stage2",
    "第二阶段 checkpoint 路径。",
)
flags.DEFINE_integer("stage1_port_number", 6678, "第一阶段 learner 端口。")
flags.DEFINE_integer("stage1_broadcast_port", 6679, "第一阶段 learner 广播端口。")
flags.DEFINE_integer("stage2_port_number", 6690, "第二阶段 learner 端口。")
flags.DEFINE_integer("stage2_broadcast_port", 6691, "第二阶段 learner 广播端口。")
flags.DEFINE_string(
    "reward_classifier_ckpt_path", "/home/sj/Desktop/zy/moqi_workspace/rl_deploy/classifier/classifier_ckpt", "Path to reward classifier ckpt for cam1."
)

flags.DEFINE_integer(
    "eval_checkpoint_step", 0, "从该步数的 ckpt 评估策略"
)
flags.DEFINE_integer("eval_n_trajs", 5, "评估时的轨迹数量")

flags.DEFINE_boolean(
    "debug", False, "调试模式"
)  # 调试模式将禁用 wandb 日志

flags.DEFINE_boolean(
    "mock", False, "使用模拟环境 (fake_env=True)，即使对于 Actor"
)
flags.DEFINE_boolean(
    "eval", False, "纯评估模式：加载最新检查点，不参与训练，无限轨迹渲染"
)
flags.DEFINE_boolean(
    "enable_spacemouse_intervention", True, "Actor 训练时启用 3D 鼠标干涉 (intervene_action 写入 buffer)"
)
flags.DEFINE_string(
    "spacemouse_event_path",
    "/dev/input/event7",
    "3D 鼠标 evdev 设备路径（如 /dev/input/event7）或 'auto' 自动检测",
)
flags.DEFINE_float(
    "spacemouse_trans_denom",
    420.0,
    "3D 鼠标 xyz 归一化系数（与测试脚本一致）",
)
flags.DEFINE_float(
    "spacemouse_rot_denom",
    380.0,
    "3D 鼠标 rx/ry/rz 归一化系数（与测试脚本一致）",
)
flags.DEFINE_float(
    "spacemouse_deadzone",
    0.08,
    "3D 鼠标死区（与测试脚本一致）",
)
flags.DEFINE_float(
    "spacemouse_rot_deadzone",
    0.16,
    "3D 鼠标旋转死区（与测试脚本一致）",
)
flags.DEFINE_string("spacemouse_ee_x", "x", "设备轴映射到 EE X（如 x / -y）")
flags.DEFINE_string("spacemouse_ee_y", "-y", "设备轴映射到 EE Y（如 y / -x）")
flags.DEFINE_string("spacemouse_ee_z", "-z", "设备轴映射到 EE Z（如 z / -z）")
flags.DEFINE_boolean(
    "spacemouse_realtime_servo",
    True,
    "3D 鼠标介入时启用 realtime servo 高频跟随，并按 env.hz 采样 transition。",
)
flags.DEFINE_float(
    "spacemouse_control_hz",
    80.0,
    "3D 鼠标实时控制更新频率。",
)
flags.DEFINE_string(
    "spacemouse_servo_backend",
    "analytic",
    "Realtime servo backend，可选 baseik 或 analytic。",
)
flags.DEFINE_float(
    "spacemouse_servo_hz",
    100.0,
    "Server realtime servo 内部循环频率。",
)
flags.DEFINE_float(
    "spacemouse_servo_trans_step",
    0.004,
    "Server realtime servo 每周期最大平移步长（米）。",
)
flags.DEFINE_float(
    "spacemouse_servo_rot_step",
    0.012,
    "Server realtime servo 每周期最大旋转步长（弧度）。",
)
flags.DEFINE_float(
    "spacemouse_servo_gripper_step",
    0.05,
    "Server realtime servo 每周期最大夹爪步长。",
)
flags.DEFINE_float(
    "spacemouse_gripper_open_cmd",
    -0.95,
    "3D 鼠标 realtime servo 开爪时发送的硬件 gripper 命令。",
)
flags.DEFINE_float(
    "spacemouse_gripper_close_cmd",
    0.0,
    "3D 鼠标 realtime servo 闭爪时发送的硬件 gripper 命令。",
)
flags.DEFINE_boolean(
    "spacemouse_print_raw",
    False,
    "是否打印 3D 鼠标原始值与覆盖动作（调试）",
)
flags.DEFINE_boolean(
    "debug_transition_shapes",
    False,
    "打印普通 transition 与 sampled transition 的结构摘要，检查 replay 样本是否同构。",
)
flags.DEFINE_integer(
    "debug_transition_every",
    1,
    "每隔多少条 transition 打印一次结构摘要。",
)

# 初始化 JAX 设备
devices = jax.local_devices()
num_devices = len(devices)
if len(devices) == 1:
    sharding = jax.sharding.SingleDeviceSharding(devices[0])
else:
    # Fallback/Polyfill for PositionalSharding on new JAX if needed, 
    # or just use NamedSharding for multi-device.
    # For now, simplistic approach for 0.8.x:
    mesh = jax.sharding.Mesh(devices, ('devices',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('devices'))


STAGE_TASKS = ("stage1", "stage2")


def get_stage_checkpoint_path(stage_name):
    return FLAGS.stage1_checkpoint_path if stage_name == "stage1" else FLAGS.stage2_checkpoint_path


def get_stage_trainer_config(stage_name):
    if stage_name == "stage1":
        return make_trainer_config(
            port_number=FLAGS.stage1_port_number,
            broadcast_port=FLAGS.stage1_broadcast_port,
        )
    return make_trainer_config(
        port_number=FLAGS.stage2_port_number,
        broadcast_port=FLAGS.stage2_broadcast_port,
    )


def print_green(x):
    """打印绿色文本"""
    return print("\033[92m {}\033[00m".format(x))


def resolve_demo_path():
    if FLAGS.demo_pkl_variant == "v1":
        return "/home/sj/Desktop/zy/moqi_workspace/rl_deploy/demo/merged_demos.pkl"
    if FLAGS.demo_pkl_variant == "v2":
        return "/home/sj/Desktop/zy/moqi_workspace/rl_deploy/demo/merged_demos_2.pkl"
    return FLAGS.demo_path


def verify_classifier_camera_alignment():
    """
    Ensure online head camera preprocessing matches classifier training assumptions.
    """
    if APPLY_HEAD_CAMERA_CROP:
        raise ValueError(
            "APPLY_HEAD_CAMERA_CROP=True will mismatch classifier training distribution. "
            "Set it to False for cam_2_rgb-trained classifier."
        )
    print(
        "[Consistency] classifier camera alignment: "
        f"cam_subdir={CLASSIFIER_CAM_SUBDIR}, device={HEAD_CAMERA_DEVICE}, "
        f"capture={HEAD_CAMERA_WIDTH}x{HEAD_CAMERA_HEIGHT}, model_input={MODEL_IMAGE_SIZE}, "
        f"crop={APPLY_HEAD_CAMERA_CROP}"
    )


def find_wrapper(wrapped_env, cls):
    cur = wrapped_env
    for _ in range(32):
        if isinstance(cur, cls):
            return cur
        if not hasattr(cur, "env"):
            break
        cur = cur.env
    return None


def set_reward_wrapper_stage(env, stage_name):
    reward_wrapper = find_wrapper(env, ClassifierRewardWrapper)
    if reward_wrapper is not None:
        reward_wrapper.set_stage(stage_name)


def restore_agent_from_checkpoint(
    agent: DrQAgent,
    checkpoint_path: str,
    label: str,
    fallback_checkpoint_path: str = None,
    reset_step_on_fallback: bool = False,
):
    candidate_paths = [checkpoint_path]
    if fallback_checkpoint_path and fallback_checkpoint_path not in candidate_paths:
        candidate_paths.append(fallback_checkpoint_path)

    for idx, ckpt_path in enumerate(candidate_paths):
        if not ckpt_path or not os.path.exists(ckpt_path):
            continue
        latest_ckpt = checkpoints.latest_checkpoint(ckpt_path)
        if not latest_ckpt:
            continue

        source_label = label if idx == 0 else f"{label} (fallback)"
        print(f"Restoring {source_label} checkpoint from {latest_ckpt}")
        restored_state = checkpoints.restore_checkpoint(ckpt_path, agent.state)
        if idx > 0 and reset_step_on_fallback:
            step_dtype = getattr(restored_state.step, "dtype", jnp.int32)
            restored_state = restored_state.replace(step=jnp.asarray(0, dtype=step_dtype))
            print(f"{label}: fallback checkpoint loaded; resetting internal step to 0 for new stage training.")
        agent = agent.replace(state=restored_state)
        print(f"{label} resumed from internal step {int(agent.state.step)}")
        return agent, ckpt_path

    print(f"No checkpoint found for {label}; starting from scratch.")
    return agent, None


##############################################################################


def actor(agent_or_agents, data_store_or_stores, env, sampling_rng):
    """
    Actor 循环，当 "--actor" 设置为 True 时运行。
    负责与环境交互、收集数据并发送给 Learner。
    """
    # 如果是纯评估模式 (FLAGS.eval)，则无限循环评估最新的 Checkpoint
    if FLAGS.eval:
        print_green("Starting Pure Two-Stage Evaluation Mode (Infinite Episodes)")
        FLAGS.render = True
        agents = agent_or_agents
        while True:
            for stage_name in STAGE_TASKS:
                ckpt_path = get_stage_checkpoint_path(stage_name)
                latest_ckpt = checkpoints.latest_checkpoint(ckpt_path)
                if latest_ckpt:
                    print(f"[Eval] Loading {stage_name} checkpoint: {latest_ckpt}")
                    restored_state = checkpoints.restore_checkpoint(ckpt_path, agents[stage_name].state)
                    agents[stage_name] = agents[stage_name].replace(state=restored_state)
                else:
                    print(f"[Eval] WARNING: No checkpoint found for {stage_name}.")

            obs, _ = env.reset()
            current_stage = "stage1"
            set_reward_wrapper_stage(env, current_stage)
            stage1_success = False
            stage2_success = False
            total_steps = 0

            while total_steps < FLAGS.max_traj_length:
                actions = agents[current_stage].sample_actions(
                    observations=jax.device_put(obs),
                    argmax=True,
                )
                actions = np.asarray(jax.device_get(actions))

                next_obs, reward, done, truncated, info = env.step(actions)
                obs = next_obs
                total_steps += 1

                if done or truncated:
                    if current_stage == "stage1" and info.get("manual_success", False):
                        stage1_success = True
                        current_stage = "stage2"
                        set_reward_wrapper_stage(env, current_stage)
                        continue
                    if current_stage == "stage2" and info.get("manual_success", False):
                        stage2_success = True
                    break

            print(
                f"[Eval] Episode Finished. steps={total_steps}, "
                f"stage1_success={stage1_success}, "
                f"stage2_success={stage2_success}, "
                f"overall_success={stage1_success and stage2_success}"
            )
            time.sleep(1.0)

    # 如果指定了评估步数 (Original Actor Eval logic)
    elif FLAGS.eval_checkpoint_step:
        success_counter = 0
        time_list = []

        # 加载指定步数的 Checkpoint
        ckpt = checkpoints.restore_checkpoint(
            get_stage_checkpoint_path("stage1"),
            agent_or_agents.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent_or_agents = agent_or_agents.replace(state=ckpt)

        # 运行评估循环
        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset()
            done = False
            start_time = time.time()
            while not done:
                # 采样动作 (评估时使用确定性策略 argmax=True)
                actions = agent_or_agents.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=True,
                )
                actions = np.asarray(jax.device_get(actions))

                next_obs, reward, done, truncated, info = env.step(actions)
                obs = next_obs

                if done:
                    if reward:
                        dt = time.time() - start_time
                        time_list.append(dt)
                        print(dt)

                    success_counter += reward
                    print(reward)
                    print(f"{success_counter}/{episode + 1}")

        print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        print(f"average time: {np.mean(time_list)}")
        return  # 评估完成后退出

    two_stage_enabled = FLAGS.enable_two_stage_training
    if two_stage_enabled:
        agents = agent_or_agents
        data_store = data_store_or_stores
        client = TrainerClient(
            "actor_env",
            FLAGS.ip,
            get_stage_trainer_config("stage2"),
            data_store,
            wait_for_server=True,
        )

        def update_params_stage2(params):
            nonlocal agents
            agents["stage2"] = agents["stage2"].replace(
                state=agents["stage2"].state.replace(params=params)
            )

        client.recv_network_callback(update_params_stage2)
    else:
        agent = agent_or_agents
        data_store = data_store_or_stores
        client = TrainerClient(
            "actor_env",
            FLAGS.ip,
            get_stage_trainer_config("stage1"),
            data_store,
            wait_for_server=True,
        )

        def update_params(params):
            nonlocal agent
            agent = agent.replace(state=agent.state.replace(params=params))

        client.recv_network_callback(update_params)

    print("[Actor] Resetting environment...")
    obs, _ = env.reset()
    current_stage = "stage1"
    set_reward_wrapper_stage(env, current_stage)
    print("[Actor] Environment reset done.")
    done = False
    transition_debug_counter = 0
    transition_reference_schema = None

    def _is_intervention_mode_active(wrapped_env):
        cur = wrapped_env
        for _ in range(32):
            if hasattr(cur, "_intervention_mode"):
                return bool(getattr(cur, "_intervention_mode"))
            if not hasattr(cur, "env"):
                break
            cur = cur.env
        return False

    def _summarize_leaf(value):
        arr = np.asarray(value)
        summary = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
            flat = arr.reshape(-1)
            summary["sample"] = np.round(flat[: min(6, flat.size)], 4).tolist()
        return summary

    def _summarize_obs_dict(obs_dict):
        return {k: _summarize_leaf(v) for k, v in obs_dict.items()}

    def _extract_transition_schema(transition):
        return {
            "observations": {k: _summarize_leaf(v) for k, v in transition["observations"].items()},
            "next_observations": {k: _summarize_leaf(v) for k, v in transition["next_observations"].items()},
            "actions": _summarize_leaf(transition["actions"]),
            "rewards": _summarize_leaf(transition["rewards"]),
            "masks": _summarize_leaf(transition["masks"]),
            "dones": {"type": type(transition["dones"]).__name__},
        }

    def _compare_schema(path, ref, cur, errors):
        if isinstance(ref, dict) and isinstance(cur, dict):
            ref_keys = set(ref.keys())
            cur_keys = set(cur.keys())
            missing = sorted(ref_keys - cur_keys)
            extra = sorted(cur_keys - ref_keys)
            if missing:
                errors.append(f"{path}: missing keys {missing}")
            if extra:
                errors.append(f"{path}: extra keys {extra}")
            for key in sorted(ref_keys & cur_keys):
                child_path = f"{path}.{key}" if path else key
                _compare_schema(child_path, ref[key], cur[key], errors)
            return
        if ref != cur:
            errors.append(f"{path}: expected {ref}, got {cur}")

    def _validate_transition_schema(kind, transition):
        nonlocal transition_reference_schema
        schema = _extract_transition_schema(transition)
        if kind == "normal":
            transition_reference_schema = schema
            return
        if transition_reference_schema is None:
            return
        errors = []
        _compare_schema("", transition_reference_schema, schema, errors)
        if errors:
            joined = " | ".join(errors)
            print(f"[TransitionError] sampled transition schema mismatch: {joined}")

    def _log_transition_debug(kind, transition, reward_value, done_value, truncated_value):
        nonlocal transition_debug_counter
        if not FLAGS.debug_transition_shapes:
            return
        transition_debug_counter += 1
        if FLAGS.debug_transition_every > 1 and (transition_debug_counter % FLAGS.debug_transition_every) != 0:
            return
        summary = {
            "kind": kind,
            "observations": _summarize_obs_dict(transition["observations"]),
            "next_observations": _summarize_obs_dict(transition["next_observations"]),
            "actions": _summarize_leaf(transition["actions"]),
            "reward": _summarize_leaf(reward_value),
            "done": bool(done_value),
            "truncated": bool(truncated_value),
            "mask": _summarize_leaf(transition["masks"]),
        }
        print("[TransitionDebug] " + json.dumps(summary, ensure_ascii=True))

    # 训练循环
    timer = Timer()
    running_return = 0.0

    print("[Actor] Starting training loop...")
    if two_stage_enabled:
        actor_step = 0
        pbar = tqdm.tqdm(total=FLAGS.max_steps, dynamic_ncols=True, desc="stage2 actor")
    else:
        actor_step = 0
        pbar = tqdm.tqdm(total=FLAGS.max_steps, dynamic_ncols=True)

    def get_stage_step(stage_name):
        return actor_step

    while True:
        if two_stage_enabled:
            if actor_step >= FLAGS.max_steps:
                break
        else:
            if actor_step >= FLAGS.max_steps:
                break

        timer.tick("total")
        skip_transition = False
        inserted_sampled_transition = False
        intervention_mode_active = _is_intervention_mode_active(env)
        transition_stage = current_stage

        with timer.context("sample_actions"):
            # 接管模式开启时，不做策略输出；动作由 3D 鼠标 wrapper 决定
            if intervention_mode_active:
                actions = np.zeros(env.action_space.shape, dtype=np.float32)
            else:
                # 在初始随机步数内，使用随机动作进行探索
                if two_stage_enabled and current_stage == "stage1":
                    actions = agents["stage1"].sample_actions(
                        observations=jax.device_put(obs),
                        argmax=True,
                    )
                    actions = np.asarray(jax.device_get(actions))
                elif get_stage_step(current_stage) < FLAGS.random_steps:
                    actions = env.action_space.sample()
                else:
                    # 之后使用 Agent 策略采样动作 (随机性用于探索)
                    sampling_rng, key = jax.random.split(sampling_rng)
                    active_agent = agents[current_stage] if two_stage_enabled else agent
                    actions = active_agent.sample_actions(
                        observations=jax.device_put(obs),
                        seed=key,
                        deterministic=False,
                    )
                    actions = np.asarray(jax.device_get(actions))

        # 执行环境步骤
        with timer.context("step_env"):
            # print(f"[Actor] Step {step} start...")
            next_obs, reward, done, truncated, info = env.step(actions)
            # print(f"[Actor] Step {step} done.")

            sampled_transition = info.get("sampled_transition") if isinstance(info, dict) else None

            # In spacemouse intervention idle mode, env.step is intentionally bypassed.
            # Skip transition insertion/logics to avoid learning useless no-op samples.
            if isinstance(info, dict) and info.get("intervention_idle", False):
                skip_transition = True

            if not skip_transition:
                # 如果存在人工干预 (例如通过 3D 鼠标)，覆盖 Agent 的动作并写入 Replay Buffer
                if "intervene_action" in info:
                    actions = info.pop("intervene_action")

                reward = np.asarray(reward, dtype=np.float32)
                running_return += reward
                
                # 构建 Transition 数据字典
                
                # --- Filter/Mask Transition for Training ---
                # Environment produces image_left. We want to KEEP the key for compatibility
                # but MASK the content to black/zeros so the network ignores it.
                
                # Local definition matches main
                training_image_keys = ["image_primary", "image_left", "image_right"] 
                
                # Helper: Copy dict -> mask 'image_left' -> return
                def mask_obs(original_obs):
                    new_obs = {}
                    for k, v in original_obs.items():
                        if k == "image_left":
                            new_obs[k] = np.zeros_like(v)
                        else:
                            new_obs[k] = v
                    return new_obs

                if sampled_transition is not None:
                    transition = dict(
                        observations=mask_obs(sampled_transition["observations"]),
                        actions=np.asarray(sampled_transition["actions"], dtype=np.float32),
                        next_observations=mask_obs(sampled_transition["next_observations"]),
                        rewards=np.asarray(sampled_transition["rewards"], dtype=np.float32),
                        masks=np.asarray(1.0 - float(sampled_transition["dones"]), dtype=np.float32),
                        dones=bool(sampled_transition["dones"]),
                    )
                    reward = transition["rewards"]
                    done = transition["dones"]
                    truncated = bool(sampled_transition.get("truncated", False))
                    actions = transition["actions"]
                    inserted_sampled_transition = True
                else:
                    obs_masked = mask_obs(obs)
                    next_obs_masked = mask_obs(next_obs)
                    transition = dict(
                        observations=obs_masked,
                        actions=actions,
                        next_observations=next_obs_masked,
                        rewards=reward,
                        masks=1.0 - done,
                        dones=done,
                    )
                _log_transition_debug(
                    kind="sampled" if sampled_transition is not None else "normal",
                    transition=transition,
                    reward_value=reward,
                    done_value=done,
                    truncated_value=truncated,
                )
                _validate_transition_schema(
                    kind="sampled" if sampled_transition is not None else "normal",
                    transition=transition,
                )
                # 将数据插入本地队列，准备发送给 Learner
                if two_stage_enabled:
                    if current_stage == "stage2":
                        data_store.insert(transition)
                else:
                    data_store.insert(transition)

                obs = next_obs

                # --- Visualization ---
                try:
                    if FLAGS.render:
                        import cv2
                        
                        # Helper to process image
                        def process_img(img_key):
                            if img_key in next_obs:
                                img = next_obs[img_key]
                                if img.ndim == 4:
                                    img = img[0]
                                # Check if image is valid
                                if img.size == 0:
                                    return np.zeros((128, 128, 3), dtype=np.uint8)
                                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                            return np.zeros((128, 128, 3), dtype=np.uint8)

                        # Get all 3 images
                        img_primary = process_img("image_primary")
                        img_left = process_img("image_left")
                        img_right = process_img("image_right")

                        # Resize to common height for stacking (e.g., 256)
                        h, w = 256, 256
                        img_primary = cv2.resize(img_primary, (w, h))
                        img_left = cv2.resize(img_left, (w, h))
                        img_right = cv2.resize(img_right, (w, h))

                        # Add labels
                        def add_label(img, text):
                            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            return img

                        img_primary = add_label(img_primary, "Primary (Head)")
                        img_left = add_label(img_left, "Left")
                        img_right = add_label(img_right, "Right (Wrist)")

                        # Stack horizontally
                        vis_img = np.hstack([img_left, img_primary, img_right])

                        # Draw Reward & Probability on the combined image
                        if isinstance(info, dict):
                            prob = info.get("classifier_prob", 0.0)
                            rew = float(np.asarray(reward))
                            color = (0, 255, 0) if prob > 0.5 else (0, 0, 255)
                            text = f"Prob: {prob:.4f}  Reward: {rew:.3f}"
                            cv2.putText(vis_img, text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                        cv2.imshow("Actor View (Left | Primary | Right)", vis_img)
                        cv2.waitKey(1)
                except Exception as e:
                    print(f"Vis Error: {e}")
                # ---------------------

                if done or truncated:
                    if two_stage_enabled:
                        if transition_stage == "stage1" and info.get("manual_success", False):
                            print("[Actor] Stage1 success received. Switching to stage2 policy without global reset.")
                            current_stage = "stage2"
                            set_reward_wrapper_stage(env, current_stage)
                        else:
                            if transition_stage == "stage2":
                                stats = {
                                    "stage2_train": info,
                                    "env_steps": actor_step + 1,
                                }
                                client.request("send-stats", stats)
                            running_return = 0.0
                            current_stage = "stage1"
                            set_reward_wrapper_stage(env, current_stage)
                            obs, _ = env.reset()
                    else:
                        stats = {"train": info}  # 发送统计数据给 Learner 记录日志
                        client.request("send-stats", stats)
                        running_return = 0.0
                        obs, _ = env.reset()

        timer.tock("total")

        # 只要这一轮真正写入了 transition，就推进 actor 计数。
        # 这样人工接管的有效控制步也会计入训练步数；只有 intervention idle/no-op 不计数。
        count_actor_step = (not skip_transition) and ((not two_stage_enabled) or transition_stage == "stage2")

        if count_actor_step:
            if two_stage_enabled:
                actor_step += 1
                pbar.update(1)

                if actor_step % FLAGS.steps_per_update == 0:
                    client.update()

                if actor_step % FLAGS.log_period == 0:
                    stats = {
                        "stage2_timer": timer.get_average_times(),
                        "env_steps": actor_step,
                    }
                    client.request("send-stats", stats)
            else:
                actor_step += 1
                pbar.update(1)

                # 定期从 Learner 更新网络参数
                if actor_step % FLAGS.steps_per_update == 0:
                    client.update()

                # 定期发送计时器统计数据
                if actor_step % FLAGS.log_period == 0:
                    stats = {"timer": timer.get_average_times()}
                    client.request("send-stats", stats)

    pbar.close()


##############################################################################


def learner(rng, agent: DrQAgent, replay_buffer, demo_buffer, stage_name="stage1"):
    """
    Learner 循环，当 "--learner" 设置为 True 时运行。
    负责从 Buffer 采样数据并更新 Agent 网络。
    """
    # 设置 wandb 和日志记录
    wandb_logger = make_wandb_logger(
        project="serl_dev",
        description=f"{FLAGS.exp_name or FLAGS.env}_{stage_name}",
        debug=FLAGS.debug,
    )

    # 跟踪训练步数
    update_steps = int(agent.state.step)
    
    # 如果从 Checkpoint 恢复，调整 update_steps 以匹配 Checkpoint 的步数
    # 这样 WandB 日志可以连续记录
    checkpoint_path = get_stage_checkpoint_path(stage_name)
    if checkpoint_path and os.path.exists(checkpoint_path):
        latest_ckpt = checkpoints.latest_checkpoint(checkpoint_path)
        if latest_ckpt:
            try:
                ckpt_step = int(latest_ckpt.split('_')[-1])
                if ckpt_step < update_steps:
                    print(f"Adjusting logging step from {update_steps} to {ckpt_step} to match checkpoint.")
                    update_steps = ckpt_step
            except ValueError:
                pass

    def stats_callback(type: str, payload: dict) -> dict:
        """服务器接收到统计请求时的回调函数"""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=update_steps)
        return {}  # 不期望返回值

    # 创建 TrainerServer
    server = TrainerServer(get_stage_trainer_config(stage_name), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.start(threaded=True)

    # 等待 Replay Buffer 填充到 training_starts 数量
    pbar = tqdm.tqdm(
        total=FLAGS.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < FLAGS.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # 更新进度条
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # 更新进度条
    pbar.close()

    # 发送初始网络参数给 Actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # 创建数据迭代器 (RLPD 核心逻辑)
    # 50% 数据来自 Replay Buffer (在线数据)，50% 来自 Demo Buffer (演示数据)
    use_demo_buffer = len(demo_buffer) > 0
    replay_batch_size = FLAGS.batch_size // 2 if use_demo_buffer else FLAGS.batch_size
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": replay_batch_size,
            "pack_obs_and_next_obs": True,
        },
        device=sharding,
    )
    demo_iterator = None
    if use_demo_buffer:
        demo_iterator = demo_buffer.get_iterator(
            sample_args={
                "batch_size": FLAGS.batch_size // 2,
                "pack_obs_and_next_obs": True,
            },
            device=sharding,
        )

    # Learner 主循环
    timer = Timer()
    learner_pbar = tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True, desc="learner")
    for step in learner_pbar:
        # 运行 n-1 次 Critic 更新和 1 次 Critic + Actor 更新
        # 这通过减少 CPU 到 GPU 的大 Batch 传输次数来加速训练
        for critic_step in range(FLAGS.critic_actor_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                if use_demo_buffer:
                    demo_batch = next(demo_iterator)
                    # 拼接在线数据和演示数据
                    batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                # 仅更新 Critic
                agent, critics_info = agent.update_critics(
                    batch,
                )

        with timer.context("train"):
            batch = next(replay_iterator)
            if use_demo_buffer:
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)
            # 更新 Actor 和 Critic (High UTD)
            agent, update_info = agent.update_high_utd(batch, utd_ratio=1)

        # 发布更新后的网络参数
        if step > 0 and step % (FLAGS.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        # 记录日志
        if update_steps % FLAGS.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=update_steps)
            wandb_logger.log({"timer": timer.get_average_times()}, step=update_steps)

        learner_pbar.set_postfix(
            update_step=int(update_steps),
            replay_size=len(replay_buffer),
            demo_size=len(demo_buffer),
            refresh=False,
        )

        # 保存 Checkpoint
        if FLAGS.checkpoint_period and update_steps % FLAGS.checkpoint_period == 0:
            assert checkpoint_path is not None
            checkpoints.save_checkpoint(
                checkpoint_path, agent.state, step=update_steps, keep=100, overwrite=True
            )

        update_steps += 1

    learner_pbar.close()


##############################################################################


def learner_two_stage(rng, agents, replay_buffers, demo_buffers):
    """
    单个 learner 进程只训练 stage2。
    stage1 仅作为已完成子任务的固定策略存在，不参与 replay/learner 更新。
    """
    wandb_logger = make_wandb_logger(
        project="serl_dev",
        description=FLAGS.exp_name or FLAGS.env,
        debug=FLAGS.debug,
    )

    update_steps = int(agents["stage2"].state.step)

    def stats_callback(type: str, payload: dict) -> dict:
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=update_steps)
        return {}

    server = TrainerServer(
        get_stage_trainer_config("stage2"),
        request_callback=stats_callback,
    )
    server.register_data_store("actor_env", replay_buffers["stage2"])
    server.start(threaded=True)
    server.publish_network(agents["stage2"].state.params)
    print_green("sent initial stage2 network to actor")

    pbar = tqdm.tqdm(
        total=FLAGS.training_starts,
        initial=len(replay_buffers["stage2"]),
        desc="Filling up stage2 replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffers["stage2"]) < FLAGS.training_starts:
        pbar.update(len(replay_buffers["stage2"]) - pbar.n)
        time.sleep(1)
    pbar.update(len(replay_buffers["stage2"]) - pbar.n)
    pbar.close()

    iterators = None
    use_demo_buffer = len(demo_buffers["stage2"]) > 0
    timer = Timer()
    learner_pbar = tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True, desc="stage2 learner")

    def ensure_iterators():
        nonlocal iterators
        if iterators is not None:
            return True
        if len(replay_buffers["stage2"]) < FLAGS.training_starts:
            return False

        replay_batch_size = FLAGS.batch_size // 2 if use_demo_buffer else FLAGS.batch_size
        replay_iterator = replay_buffers["stage2"].get_iterator(
            sample_args={
                "batch_size": replay_batch_size,
                "pack_obs_and_next_obs": True,
            },
            device=sharding,
        )
        demo_iterator = None
        if use_demo_buffer:
            demo_iterator = demo_buffers["stage2"].get_iterator(
                sample_args={
                    "batch_size": FLAGS.batch_size // 2,
                    "pack_obs_and_next_obs": True,
                },
                device=sharding,
            )
        iterators = (replay_iterator, demo_iterator)
        print_green("stage2 replay buffer is ready; learner updates enabled.")
        return True

    for _ in learner_pbar:
        did_update = False
        if ensure_iterators():
            replay_iterator, demo_iterator = iterators

            for critic_step in range(FLAGS.critic_actor_ratio - 1):
                with timer.context("stage2_sample_replay_buffer"):
                    batch = next(replay_iterator)
                    if use_demo_buffer:
                        demo_batch = next(demo_iterator)
                        batch = concat_batches(batch, demo_batch, axis=0)

                with timer.context("stage2_train_critics"):
                    agents["stage2"], critics_info = agents["stage2"].update_critics(batch)

            with timer.context("stage2_train"):
                batch = next(replay_iterator)
                if use_demo_buffer:
                    demo_batch = next(demo_iterator)
                    batch = concat_batches(batch, demo_batch, axis=0)
                agents["stage2"], update_info = agents["stage2"].update_high_utd(batch, utd_ratio=1)

            if update_steps > 0 and update_steps % FLAGS.steps_per_update == 0:
                agents["stage2"] = jax.block_until_ready(agents["stage2"])
                server.publish_network(agents["stage2"].state.params)

            if update_steps % FLAGS.log_period == 0 and wandb_logger:
                prefixed_update_info = {
                    f"stage2_{k}": v for k, v in update_info.items()
                }
                wandb_logger.log(prefixed_update_info, step=update_steps)
                wandb_logger.log(
                    {"stage2_timer": timer.get_average_times()},
                    step=update_steps,
                )
                wandb_logger.log(
                    {
                        "stage2_replay_size": len(replay_buffers["stage2"]),
                        "stage2_demo_size": len(demo_buffers["stage2"]),
                    },
                    step=update_steps,
                )

            checkpoint_path = get_stage_checkpoint_path("stage2")
            if FLAGS.checkpoint_period and update_steps % FLAGS.checkpoint_period == 0:
                checkpoints.save_checkpoint(
                    checkpoint_path,
                    agents["stage2"].state,
                    step=update_steps,
                    keep=100,
                    overwrite=True,
                )

            update_steps += 1
            learner_pbar.set_postfix(
                update_step=int(update_steps),
                replay_size=len(replay_buffers["stage2"]),
                demo_size=len(demo_buffers["stage2"]),
                refresh=False,
            )
            did_update = True

        if not did_update:
            time.sleep(1)
    learner_pbar.close()


##############################################################################


def load_demo_buffer(demo_buffer, training_image_keys, stage_name="stage1"):
    demo_path = resolve_demo_path()
    if stage_name == "stage2" and FLAGS.enable_two_stage_training:
        print("Stage2 learner: skipping demo loading and training from its own online replay buffer.")
        return

    print(f"Resolved demo path ({FLAGS.demo_pkl_variant}): {demo_path}")
    if not demo_path:
        print("WARNING: No demo path provided. Demo buffer will be empty.")
        return

    if not os.path.exists(demo_path):
        raise FileNotFoundError(f"File {demo_path} not found")

    with open(demo_path, "rb") as f:
        transitions = pkl.load(f)

    print(f"Loading {len(transitions)} transitions from {demo_path} into {stage_name} demo buffer...")
    demo_over_limit_count = 0
    demo_loaded_count = 0
    demo_skipped_count = 0

    for i, t in enumerate(transitions):
        obs_state = t['observations']['state']
        if t['next_observations'] is not None:
            next_obs_state = t['next_observations']['state']
            next_obs_dict = {k: v for k, v in t["next_observations"].items() if k != "state"}
        else:
            next_obs_state = obs_state
            next_obs_dict = {k: v for k, v in t["observations"].items() if k != "state"}

        if obs_state.shape[0] != 13:
            print(f"Warning: Demo state shape mismatch. Expected 13, got {obs_state.shape[0]}")

        final_state = obs_state
        final_next_state = next_obs_state

        action_physical = t["actions"]
        action_pos_physical = action_physical[:3]
        action_rot_physical = action_physical[3:6]
        gripper_physical = action_physical[6]

        SCALE_POS = 0.01
        SCALE_ROT = 0.05

        action_pos_norm = action_pos_physical / SCALE_POS
        action_rot_norm = action_rot_physical / SCALE_ROT
        gripper_norm = (gripper_physical / 0.5236) + 1.0
        over_limit = (
            np.any(np.abs(action_pos_norm) > 1.0)
            or np.any(np.abs(action_rot_norm) > 1.0)
            or abs(float(gripper_norm)) > 1.0
        )
        if over_limit:
            demo_over_limit_count += 1
            if FLAGS.demo_drop_over_limit_transitions:
                demo_skipped_count += 1
                continue

        action_pos_norm = np.clip(action_pos_norm, -1.0, 1.0)
        action_rot_norm = np.clip(action_rot_norm, -1.0, 1.0)
        gripper_norm = np.clip(gripper_norm, -1.0, 1.0)
        final_action = np.concatenate([action_pos_norm, action_rot_norm, [gripper_norm]])

        reward_val = t['rewards']
        valid_keys = set(training_image_keys) | {"state"}
        obs_dict = {}
        next_obs_dict_final = {}

        def process_obs_dict(source_dict, target_dict):
            for k, v in source_dict.items():
                if k not in valid_keys and k != "state":
                    continue
                if k == "image_left":
                    target_dict[k] = np.zeros_like(v)
                else:
                    target_dict[k] = v

        process_obs_dict(t["observations"], obs_dict)
        process_obs_dict(next_obs_dict, next_obs_dict_final)
        obs_dict["state"] = final_state
        next_obs_dict_final["state"] = final_next_state

        done = bool(t["dones"])
        transition = {
            "observations": obs_dict,
            "next_observations": next_obs_dict_final,
            "actions": final_action,
            "rewards": np.array(reward_val, dtype=np.float32),
            "masks": np.array(0.0 if done else 1.0, dtype=np.float32),
            "dones": done,
        }
        demo_buffer.insert(transition)
        demo_loaded_count += 1

    print(
        f"{stage_name} demo buffer size: {len(demo_buffer)} "
        f"(loaded={demo_loaded_count}, over_limit={demo_over_limit_count}, skipped={demo_skipped_count})"
    )


##############################################################################


def main(_):
    assert FLAGS.batch_size % num_devices == 0
    verify_classifier_camera_alignment()
    # 设置随机种子
    rng = jax.random.PRNGKey(FLAGS.seed)

    # 定义包含摄像头的配置，以触发 Observation Space 的创建
    # 键名必须与 _update_currpos 生成的匹配 ("image_primary" 等)
    class TrainConfig(DefaultOpenArmConfig):
        REALSENSE_CAMERAS = {
            "image_primary": "real_camera_serial_or_empty", # <--- 使用真实摄像头
            "image_left": "real_camera_serial_or_empty",
            "image_right": "real_camera_serial_or_empty"
        }
        
    # 创建环境并加载数据集
    # 硬编码为 Right Arm
    # Use LocalOpenArmEnv instead of OpenArmEnv
    env = LocalOpenArmEnv(
        fake_env=FLAGS.learner or FLAGS.mock, # Learner 使用模拟环境
        save_video=FLAGS.eval_checkpoint_step,
        arm=FLAGS.arm,
        hz=FLAGS.hz,
        config=TrainConfig(),

        max_episode_length=FLAGS.max_traj_length # Pass max_traj_length
    )
    
    # Wrappers (必须与 record_demo.py 匹配)
    env = RelativeFrame(env) # 相对坐标系
    env = Quat2EulerWrapper(env) # 四元数转欧拉角
    env = SERLObsWrapper(env) # 图像观测处理
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None) # 动作分块


    # 获取图像键列表
    # 原逻辑: image_keys = [key for key in env.observation_space.keys() if key != "state"]
    # 修改逻辑: 恢复 3 相机结构以保持 Checkpoint 兼容性。但 Left 相机会在后续被替换为全黑。
    training_image_keys = ["image_primary", "image_left", "image_right"]
    print(f"Training with image keys: {training_image_keys} (Left will be masked)")

    # Create random keys
    rng, sampling_rng = jax.random.split(rng)

    if FLAGS.actor or FLAGS.eval:
        # initialize the reward wrapper
        if FLAGS.reward_classifier_ckpt_path is None:
             raise ValueError("reward_classifier_ckpt_path must be specified for actor/eval")
        if not os.path.exists(FLAGS.reward_classifier_ckpt_path):
             raise FileNotFoundError(
                 f"reward_classifier_ckpt_path not found: {FLAGS.reward_classifier_ckpt_path}"
             )

        # Use ClassifierRewardWrapper
        env = ClassifierRewardWrapper(
            env, 
            classifier_ckpt_path=FLAGS.reward_classifier_ckpt_path,
            reward_image_key="image_primary",
            enable_animation=FLAGS.enable_success_animation
        )

        # 3D Mouse intervention for actor training (evdev, same source as test_3dx_operation.py).
        # Wrapper injects info["intervene_action"], and actor loop already writes it to replay buffer.
        if (FLAGS.actor or FLAGS.eval) and FLAGS.enable_spacemouse_intervention:
            env = EvdevSpacemouseIntervention(
                env,
                event_path=FLAGS.spacemouse_event_path,
                trans_denom=FLAGS.spacemouse_trans_denom,
                rot_denom=FLAGS.spacemouse_rot_denom,
                deadzone=FLAGS.spacemouse_deadzone,
                rot_deadzone=FLAGS.spacemouse_rot_deadzone,
                ee_x=FLAGS.spacemouse_ee_x,
                ee_y=FLAGS.spacemouse_ee_y,
                ee_z=FLAGS.spacemouse_ee_z,
                realtime_servo=FLAGS.spacemouse_realtime_servo,
                control_hz=FLAGS.spacemouse_control_hz,
                servo_backend=FLAGS.spacemouse_servo_backend,
                servo_hz=FLAGS.spacemouse_servo_hz,
                servo_trans_step=FLAGS.spacemouse_servo_trans_step,
                servo_rot_step=FLAGS.spacemouse_servo_rot_step,
                servo_gripper_step=FLAGS.spacemouse_servo_gripper_step,
                gripper_open_cmd=FLAGS.spacemouse_gripper_open_cmd,
                gripper_close_cmd=FLAGS.spacemouse_gripper_close_cmd,
                print_raw=FLAGS.spacemouse_print_raw,
            )
            mode_label = "Actor" if FLAGS.actor else "Eval"
            print(f"[{mode_label}] EvdevSpacemouseIntervention enabled: manual mode can override arm pose and gripper.")

        print("[Train] Policy controls the gripper by default; 3D mouse controls it only during manual intervention.")
    
    # Move RecordEpisodeStatistics here (outermost) to capture rewards from ForwardReachRewardWrapper
    env = RecordEpisodeStatistics(env) # 记录统计信息

    # rng, sampling_rng = jax.random.split(rng) # Moved up
    # 创建 DrQ Agent
    # make_drq_agent 是一个工厂函数，它不仅初始化网络结构（Actor, Critic, Encoder），
    # 还会初始化优化器状态（Optimizer State）。此时 Agent 还只存在于 CPU 内存中。
    agent: DrQAgent = make_drq_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),   # 提供一个观测样本，让网络自动推断输入层形状
        sample_action=env.action_space.sample(),     # 提供一个动作样本，让网络推断输出层形状
        image_keys=training_image_keys,              # 告诉 Agent 哪些观测是图像（需要经过 CNN 编码器）
        encoder_type=FLAGS.encoder_type,             # 选择编码器架构（如 "resnet-pretrained"）
    )

    # 在设备间复制 Agent (用于多 GPU 训练，这里通常是单 GPU)
    # jax.device_put: 将数据从 CPU 内存移动到 GPU 显存
    # sharding.replicate(): 指定分片策略为“复制”。如果有多个 GPU，Agent 会在每个 GPU 上都拷贝一份（数据并行）。
    # jax.tree_util.tree_map(jnp.array, agent): 这是一个防御性编程。
    # 有时候 Python 的原生类型（int/float）混在 Agent 结构里会导致 device_put 报错，
    # 这里强制把 Agent 里的所有叶子节点都转成 JAX 数组 (jnp.array)。
    agent: DrQAgent = jax.device_put(
        jax.tree_util.tree_map(jnp.array, agent), sharding
    )

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding)

        if FLAGS.enable_two_stage_training:
            agents = {}
            replay_buffers = {}
            demo_buffers = {}

            agents["stage1"], _ = restore_agent_from_checkpoint(
                agent,
                FLAGS.stage1_checkpoint_path,
                label="stage1 learner",
            )

            stage2_agent = make_drq_agent(
                seed=FLAGS.seed,
                sample_obs=env.observation_space.sample(),
                sample_action=env.action_space.sample(),
                image_keys=training_image_keys,
                encoder_type=FLAGS.encoder_type,
            )
            stage2_agent = jax.device_put(
                jax.tree_util.tree_map(jnp.array, stage2_agent), sharding
            )
            agents["stage2"], _ = restore_agent_from_checkpoint(
                stage2_agent,
                FLAGS.stage2_checkpoint_path,
                label="stage2 learner",
                fallback_checkpoint_path=FLAGS.stage1_checkpoint_path,
                reset_step_on_fallback=True,
            )

            for stage_name in STAGE_TASKS:
                replay_buffers[stage_name] = MemoryEfficientReplayBufferDataStore(
                    env.observation_space,
                    env.action_space,
                    capacity=FLAGS.replay_buffer_capacity,
                    image_keys=training_image_keys,
                )
                demo_buffers[stage_name] = MemoryEfficientReplayBufferDataStore(
                    env.observation_space,
                    env.action_space,
                    capacity=10000,
                    image_keys=training_image_keys,
                )

            load_demo_buffer(demo_buffers["stage2"], training_image_keys, stage_name="stage2")

            print_green("starting stage2-only learner loop with frozen stage1 policy")
            learner_two_stage(
                sampling_rng,
                agents,
                replay_buffers,
                demo_buffers,
            )
        else:
            agent, _ = restore_agent_from_checkpoint(
                agent,
                FLAGS.stage1_checkpoint_path,
                label="stage1 learner",
            )

            replay_buffer = MemoryEfficientReplayBufferDataStore(
                env.observation_space,
                env.action_space,
                capacity=FLAGS.replay_buffer_capacity,
                image_keys=training_image_keys,
            )
            demo_buffer = MemoryEfficientReplayBufferDataStore(
                env.observation_space,
                env.action_space,
                capacity=10000,
                image_keys=training_image_keys,
            )
            load_demo_buffer(demo_buffer, training_image_keys, stage_name="stage1")

            print_green("starting learner loop")
            learner(
                sampling_rng,
                agent,
                replay_buffer,
                demo_buffer=demo_buffer,
                stage_name="stage1",
            )

    elif FLAGS.actor or FLAGS.eval:
        # Actor/Eval 模式
        sampling_rng = jax.device_put(sampling_rng, sharding)
        if FLAGS.enable_two_stage_training:
            agents = {}
            agents["stage1"], _ = restore_agent_from_checkpoint(
                agent,
                FLAGS.stage1_checkpoint_path,
                label="stage1 actor",
            )
            stage2_agent = make_drq_agent(
                seed=FLAGS.seed,
                sample_obs=env.observation_space.sample(),
                sample_action=env.action_space.sample(),
                image_keys=training_image_keys,
                encoder_type=FLAGS.encoder_type,
            )
            stage2_agent = jax.device_put(
                jax.tree_util.tree_map(jnp.array, stage2_agent), sharding
            )
            agents["stage2"], _ = restore_agent_from_checkpoint(
                stage2_agent,
                FLAGS.stage2_checkpoint_path,
                label="stage2 actor",
                fallback_checkpoint_path=FLAGS.stage1_checkpoint_path,
                reset_step_on_fallback=True,
            )
            data_store = QueuedDataStore(2000)
        else:
            agent, _ = restore_agent_from_checkpoint(
                agent,
                FLAGS.stage1_checkpoint_path,
                label="stage1 actor/eval",
            )
            data_store = QueuedDataStore(2000)  # Actor 上的队列大小
        # 启动过程
        if FLAGS.eval:
             print_green("Starting evaluation...")
        else:
             print_green("Starting actor loop...")
        if FLAGS.enable_two_stage_training:
            actor(agents, data_store, env, sampling_rng)
        else:
            actor(agent, data_store, env, sampling_rng)

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
