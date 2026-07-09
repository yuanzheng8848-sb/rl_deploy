"""Gym Interface for OpenArm (Optimized & Commented)"""
import time
import copy
import queue
import threading
from typing import Dict, Tuple, Optional, Any

import numpy as np
import gymnasium as gym
import cv2
import requests
from scipy.spatial.transform import Rotation
from collections import OrderedDict
import base64 # Added by instruction

# OpenArm IK, camera, control, configs, and description now live under
# rl_robot_infra; this module should not inject paths to an external workspace.

# --- 辅助函数 ---
def euler_2_quat(euler: np.ndarray) -> np.ndarray:
    """将欧拉角 [x, y, z] 转换为四元数 [x, y, z, w]"""
    return Rotation.from_euler("xyz", euler).as_quat()

def quat_2_euler(quat: np.ndarray) -> np.ndarray:
    """将四元数 [x, y, z, w] 转换为欧拉角 [x, y, z]"""
    return Rotation.from_quat(quat).as_euler("xyz")


class ImageDisplayer(threading.Thread):
    """
    后台线程：用于显示摄像头画面�?
    目的：防�?cv2.imshow �?cv2.waitKey 阻塞主强化学习训练循环，保证控制频率�?
    """
    def __init__(self, queue_obj):
        threading.Thread.__init__(self)
        self.queue = queue_obj
        self.daemon = True

    def run(self):
        while True:
            img_array = self.queue.get()
            if img_array is None:
                break
            # 过滤掉全景图(full)，将剩余视角的图片拼接显�?
            valid_imgs = [v for k, v in img_array.items() if "full" not in k]
            if valid_imgs:
                frame = np.concatenate(valid_imgs, axis=0) # 垂直拼接
                cv2.imshow("RealSense Cameras", frame)
                cv2.waitKey(1)


class DefaultOpenArmConfig:
    """OpenArm 环境的默认配置参�?""
    SERVER_URL: str = "http://127.0.0.1:5000/"
    # 相机配置: {名称: 序列号}
    REALSENSE_CAMERAS: Dict[str, str] = {}
    
    # 任务相关: 目标位姿和奖励阈�?
    TARGET_POSE: np.ndarray = np.zeros((6,))
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))
    
    # 动作缩放系数: [平移, 旋转, 夹爪]
    # 将神经网络输出的 [-1, 1] 映射为实际的物理增量 (�?弧度)
    ACTION_SCALE: np.ndarray = np.array([0.01, 0.05, 1.0]) 
    
    # 复位状�?(Cartesian XYZ + Euler RPY)
    RESET_POSE: np.ndarray = np.zeros((6,)) 
    
    # 安全边界 [x, y, z, r, p, y]
    ABS_POSE_LIMIT_HIGH: np.ndarray = np.array([0.5, 0.5, 0.8, 3.14, 3.14, 3.14])
    ABS_POSE_LIMIT_LOW: np.ndarray = np.array([-0.5, -0.5, 0.0, -3.14, -3.14, -3.14])
    
    # 夹爪参数
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
    if hasattr(config_or_env, "gripper_open_threshold") and hasattr(config_or_env, "gripper_close_threshold"):
        return float(config_or_env.gripper_open_threshold), float(config_or_env.gripper_close_threshold)
    if hasattr(config_or_env, "config"):
        source = config_or_env.config
    else:
        source = config_or_env

    if not hasattr(source, "GRIPPER_OPEN_THRESHOLD") or not hasattr(source, "GRIPPER_CLOSE_THRESHOLD"):
        raise AttributeError("config_or_env must provide gripper thresholds via env fields or config")

    open_threshold = float(source.GRIPPER_OPEN_THRESHOLD)
    close_threshold = float(source.GRIPPER_CLOSE_THRESHOLD)
    return open_threshold, close_threshold


class OpenArmEnv(gym.Env):
    def __init__(
        self,
        hz=5,
        fake_env=False,
        save_video=False,
        config: DefaultOpenArmConfig = None,
        max_episode_length=100,
    ):
        self.hz = hz
        self.fake_env = fake_env
        self.save_video = save_video
        self.config = config or DefaultOpenArmConfig()
        self.max_episode_length = max_episode_length
        self.arm = "both"
        
        # Initialize session
        self.session = requests.Session()
        self.url = self.config.SERVER_URL

        # --- Internal State Init ---
        self.action_scale = self.config.ACTION_SCALE
        self._TARGET_POSE = self.config.TARGET_POSE
        self._REWARD_THRESHOLD = self.config.REWARD_THRESHOLD
        
        # Safety Bounding Box
        self.xyz_bounding_box = gym.spaces.Box(
            self.config.ABS_POSE_LIMIT_LOW[:3],
            self.config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )

        # State shape: (2, N) for dual arm
        single_arm_quat = euler_2_quat(self.config.RESET_POSE[3:])
        single_arm_reset = np.concatenate([self.config.RESET_POSE[:3], single_arm_quat])
        self.resetpos = np.vstack([single_arm_reset, single_arm_reset]) # Shape (2, 7)

        # Init current state variables
        self.currpos = self.resetpos.copy()
        self.currvel = np.zeros((2, 6))
        self.q = np.zeros((2, 7))      # Joint angles
        self.dq = np.zeros((2, 7))     # Joint velocities
        self.currforce = np.zeros((2, 3))
        self.currtorque = np.zeros((2, 3))
        
        self.curr_gripper_pos = np.zeros((2,))
        self.gripper_binary_state = np.zeros((2,), dtype=int) # 0:开, 1:�?
        self.gripper_open_threshold, self.gripper_close_threshold = get_gripper_thresholds(self.config)
        self.safe_gripper_open_cmd = float(
            getattr(self.config, "SAFE_GRIPPER_OPEN_CMD", -1.0)
        )
        self.safe_gripper_close_cmd = float(
            getattr(self.config, "SAFE_GRIPPER_CLOSE_CMD", 0.05)
        )
        
        self.curr_path_length = 0
        self.cycle_count = 0
        self.latest_images = {} # Store decoded images
        
        # --- Observation Space ---
        # OpenArmEnv is hard-coded to bimanual mode.
        tcp_shape = (2, 7)
        gripper_shape = (2, 1)
        action_dim = 14

        obs_dict = {
            "state": gym.spaces.Dict({
                "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=tcp_shape),
                "gripper_pose": gym.spaces.Box(-1, 1, shape=gripper_shape),
            })
        }
        
        # Add Image Spaces based on config
        # Nest them under "images" key for SERLObsWrapper compatibility
        img_spaces = {}
        if self.config.REALSENSE_CAMERAS:
            for name in self.config.REALSENSE_CAMERAS.keys():
                # Resize all images to 128x128 for RL training to save memory
                shape = (128, 128, 3)
                img_spaces[name] = gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)
        
        if img_spaces:
            obs_dict["images"] = gym.spaces.Dict(img_spaces)
                
        # Add tcp_vel to observation space (required for RelativeFrame).
        # Velocity feedback is currently unavailable, so zeros are returned.
        obs_dict["state"]["tcp_vel"] = gym.spaces.Box(
            -np.inf, np.inf, shape=(2, 6)
        )
        
        self.observation_space = gym.spaces.Dict(obs_dict)
        
        # --- Action Space ---
        self.action_space = gym.spaces.Box(
            -1 * np.ones((action_dim,), dtype=np.float32),
            np.ones((action_dim,), dtype=np.float32),
        )
        
        if fake_env:
            print(f"Initialized OpenArm Env (FAKE Mode) - Arm: {self.arm}")
            return

        # --- 硬件与可视化初始�?---
        self.displayer = None
        if self.config.REALSENSE_CAMERAS:
            self.init_cameras(self.config.REALSENSE_CAMERAS)
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue)
            self.displayer.start()
        
        print(f"Initialized OpenArm Env (Bimanual) connected to {self.url} - Arm: {self.arm}")

    def sync_binary_gripper_state_from_position(self):
        current = np.asarray(self.curr_gripper_pos, dtype=np.float32).reshape(-1)
        if current.size < 2:
            return

        open_cmd = float(self.safe_gripper_open_cmd)
        close_cmd = float(self.safe_gripper_close_cmd)
        updated = np.zeros((2,), dtype=np.int32)
        for arm_idx in range(2):
            pos = float(current[arm_idx])
            dist_open = abs(pos - open_cmd)
            dist_close = abs(pos - close_cmd)
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

        commands = []
        for arm_idx, gripper_idx in enumerate((6, 13)):
            commands.append(self._apply_gripper_action(float(action[gripper_idx]), arm_idx))
        return commands

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """
        安全裁剪：将目标位姿 (2, 7) �?XYZ 限制在安全盒子内�?
        使用�?NumPy 向量化操作，同时处理双臂�?
        """
        clipped_pose = pose.copy()
        clipped_pose[:, :3] = np.clip(
            clipped_pose[:, :3], 
            self.xyz_bounding_box.low, 
            self.xyz_bounding_box.high
        )
        return clipped_pose

    def step(self, action: np.ndarray) -> tuple:
        """
        环境步进函数�?
        1. 解析动作 (delta) -> 2. 计算目标位姿 -> 3. 发送指�?-> 4. 获取新状�?
        """
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        self.nextpos = self.currpos.copy()
        gripper_cmds = []

        arm_indices = [0, 1]
        
        for i in arm_indices:
            idx_start = i * 7
            current_arm_action = action[idx_start : idx_start + 7]

            # 1. 平移更新 (Translation)
            xyz_delta = current_arm_action[:3]
            self.nextpos[i, :3] += xyz_delta * self.action_scale[0]
            
            # 2. 旋转更新 (Rotation)
            rpy_delta = current_arm_action[3:6] * self.action_scale[1]
            rot_delta = Rotation.from_euler("xyz", rpy_delta)
            rot_curr = Rotation.from_quat(self.nextpos[i, 3:])
            self.nextpos[i, 3:] = (rot_delta * rot_curr).as_quat()
            
            # 3. 夹爪控制由下面的统一二值状态机构建

        final_gripper_cmds = self.gripper_actions_to_commands(action)

        # 安全限制并发送笛卡尔位置指令 (包含夹爪)
        # Note: self.nextpos for inactive arms remains as self.currpos (copied at start)
        target_pos = self.clip_safety_box(self.nextpos)
        self._send_pos_command(target_pos, gripper_pos=final_gripper_cmds)

        # 频率控制: 动态休眠以维持稳定�?Hz
        self.curr_path_length += 1
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        # 更新状�?
        self._update_currpos()

        # 计算奖励与结束标�?
        ob = self._get_obs()
        reward = self.compute_reward(ob, False) # gripper_effective is removed/ignored for now
        done = self.curr_path_length >= self.max_episode_length or reward == 1.0
        
        return ob, reward, done, False, {}

    def compute_reward(self, obs: Dict, gripper_effective: bool) -> float:
        """
        计算奖励 (稀疏奖励逻辑)
        目前逻辑：已被禁用。奖励完全由外部 Wrapper (Reward Classifier) 提供�?
        """
        # Internal reward is disabled to avoid interference with Classifier Reward
        return 0.0

    def _get_obs(self) -> dict:
        """组装观测字典"""
        # Return images from latest update
        images = {}
        # Check if "images" key exists in observation space (it might not if no cameras)
        if "images" in self.observation_space.spaces:
            for key, space in self.observation_space["images"].spaces.items():
                if key in self.latest_images:
                    images[key] = self.latest_images[key]
                else:
                    # Return black image if missing
                    images[key] = np.zeros(space.shape, dtype=np.uint8)

        tcp_obs = self.currpos.copy() # (2, 7)
        # Expose gripper state as the same binary semantics used for execution:
        # open -> -1, close -> +1.
        gripper_obs = np.where(self.gripper_binary_state[:, None] == 1, 1.0, -1.0).astype(
            np.float32
        )

        state_observation = {
            "tcp_pose": tcp_obs,
            "tcp_vel": np.zeros((2, 6), dtype=np.float32), # Dummy velocity
            "gripper_pose": gripper_obs,
        }
        
        # Return nested dictionary
        return {"images": images, "state": state_observation}

    def render(self, mode="human"):
        """Render method for Gym compatibility. Visualization is handled by Server."""
        pass

    def reset(self, **kwargs):
        """重置环境"""
        if self.save_video:
            self.save_video_recording()

        self.cycle_count += 1
        # 总是执行关节回零 (确保每次 Reset 都回到正确的初始位置)
        self.go_to_rest()

        self.curr_path_length = 0
        self.currpos = self.resetpos.copy()
        self.currvel = np.zeros((2, 6))

        self.gripper_binary_state = np.zeros((2,), dtype=int)

        # 发送复位指�?(这一步其实是多余的，因为 go_to_rest 已经回零了，
        # 但为了更�?self.currpos 对应�?Cartesian 状态，保留也无妨，
        # 或者应该在 go_to_rest 后直�?update_currpos)
        # self._send_pos_command(self.currpos)
        # Better: Just update current state from server
        self._update_currpos()
        self.sync_binary_gripper_state_from_position()

        # Update initial reset pose for station keeping of inactive arms
        self.initial_reset_pose = self.currpos.copy()

        return self._get_obs(), {}

    def go_to_rest(self):
        """强制服务器执行关节回�?(Home Reset)"""
        try:
            # Server blocks for 10s, so we need a timeout > 10s
            self.session.post(self.url + "jointreset", timeout=5)
            time.sleep(1) # Extra buffer
        except requests.exceptions.RequestException as e:
            print(f"[Env Warning] Joint reset failed: {e}")
        self._update_currpos()

    def _send_pos_command(self, pos: np.ndarray, gripper_pos: list = None):
        """发送目标位�?(2, 7) 到服务器"""
        arr = pos.astype(np.float32)
        data = {"arr": arr.tolist()} # 转换为嵌套列�?
        
        # Calculate duration for smooth blocking movement
        # Ensure it matches the control frequency
        duration = 1.0 / self.hz
        data["duration"] = duration
        
        if gripper_pos is not None:
            data["gripper"] = [float(x) for x in gripper_pos]
            
        try:
            resp = self.session.post(self.url + "pose", json=data, timeout=5.0)
            if resp.status_code != 200:
                print(f"[Env Error] Pose command failed: {resp.text}")
        except requests.exceptions.RequestException as e:
            print(f"[Env Error] Pose request failed: {e}")

    def refresh_obs(self) -> dict:
        """Refresh state from server without issuing a new motion command."""
        self._update_currpos()
        return self._get_obs()

    def _update_currpos(self):
        """从服务器获取最新状态并更新内部变量"""
        try:
            resp = self.session.post(self.url + "getstate", timeout=5.0)
            ps = resp.json()
            
            # 辅助函数: 确保数据形状�?(2, cols)
            def ensure_shape(arr_list, cols):
                arr = np.array(arr_list)
                if arr.size == 2 * cols:
                    return arr.reshape(2, cols)
                return arr 

            # Update Cartesian pose from server-side FK result.
            if "pose" in ps:
                self.currpos[:] = ensure_shape(ps["pose"], 7)
            
            self.q[:] = ensure_shape(ps["q"], 7)
            self.dq[:] = ensure_shape(ps.get("dq", [0]*14), 7) # Server might not return dq
            self.curr_gripper_pos = np.array(ps["gripper_pos"])
            self.sync_binary_gripper_state_from_position()
            
            # Update Images
            if "images" in ps:
                for name, b64_str in ps["images"].items():
                    try:
                        # Decode Base64 -> Bytes -> Numpy -> CV2 Decode
                        img_bytes = base64.b64decode(b64_str)
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        # Resize to 128x128 for RL
                        img = cv2.resize(img, (128, 128))
                        # Convert to RGB (Model expects RGB)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # Map keys
                        key = None
                        if name == "head": key = "image_primary"
                        elif name == "left": key = "image_left"
                        elif name == "right": key = "image_right"
                        
                        if key:
                            self.latest_images[key] = img
                    except Exception as e:
                        print(f"[Env Warning] Failed to decode image {name}: {e}")

        except Exception as e:
            print(f"[Env Error] Update state failed: {e}")

    def init_cameras(self, cameras):
        pass 

    def save_video_recording(self):
        pass

    def close(self):
        # fake_env init returns early and never sets self.displayer, so guard with getattr.
        if getattr(self, "displayer", None):
            self.img_queue.put(None)
            self.displayer.join()
        self.session.close()
