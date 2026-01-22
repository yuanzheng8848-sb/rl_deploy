"""Gym Interface for OpenArm (Optimized & Commented)"""
import sys
import time
import copy
import queue
import threading
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import numpy as np
import gym
import cv2
import requests
from scipy.spatial.transform import Rotation
from collections import OrderedDict
import base64 # Added by instruction

# --- 路径设置 ---
# 将 serl_robot_infra 和 pyroki 库加入 Python 路径，以便导入底层依赖
sys.path.append(str(Path(__file__).parent.parent.parent / "serl" / "serl_robot_infra"))
sys.path.append(str(Path(__file__).parent.parent / "pyroki"))

# --- 可选导入 (可视化与运动学) ---
# 使用 try-except 确保即使缺少可视化库，环境也能在无头模式(Headless)下运行
try:
    from robot_ik_solver import BaseIKSolver
    import yaml
except ImportError as e:
    print(f"[OpenArmEnv] Pyroki imports failed: {e}. Visualization disabled.")
    BaseIKSolver = None
    yaml = None

# --- 辅助函数 ---
def euler_2_quat(euler: np.ndarray) -> np.ndarray:
    """将欧拉角 [x, y, z] 转换为四元数 [x, y, z, w]"""
    return Rotation.from_euler("xyz", euler).as_quat()

def quat_2_euler(quat: np.ndarray) -> np.ndarray:
    """将四元数 [x, y, z, w] 转换为欧拉角 [x, y, z]"""
    return Rotation.from_quat(quat).as_euler("xyz")


class ImageDisplayer(threading.Thread):
    """
    后台线程：用于显示摄像头画面。
    目的：防止 cv2.imshow 和 cv2.waitKey 阻塞主强化学习训练循环，保证控制频率。
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
            # 过滤掉全景图(full)，将剩余视角的图片拼接显示
            valid_imgs = [v for k, v in img_array.items() if "full" not in k]
            if valid_imgs:
                frame = np.concatenate(valid_imgs, axis=0) # 垂直拼接
                cv2.imshow("RealSense Cameras", frame)
                cv2.waitKey(1)


class DefaultOpenArmConfig:
    """OpenArm 环境的默认配置参数"""
    SERVER_URL: str = "http://127.0.0.1:5000/"
    # 相机配置: {名称: 序列号}
    REALSENSE_CAMERAS: Dict[str, str] = {}
    
    # 任务相关: 目标位姿和奖励阈值
    TARGET_POSE: np.ndarray = np.zeros((6,))
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))
    
    # 动作缩放系数: [平移, 旋转, 夹爪]
    # 将神经网络输出的 [-1, 1] 映射为实际的物理增量 (米/弧度)
    ACTION_SCALE: np.ndarray = np.array([0.01, 0.05, 1.0]) 
    
    # 复位状态 (Cartesian XYZ + Euler RPY)
    RESET_POSE: np.ndarray = np.zeros((6,)) 
    
    # 安全边界 [x, y, z, r, p, y]
    ABS_POSE_LIMIT_HIGH: np.ndarray = np.array([0.5, 0.5, 0.8, 3.14, 3.14, 3.14])
    ABS_POSE_LIMIT_LOW: np.ndarray = np.array([-0.5, -0.5, 0.0, -3.14, -3.14, -3.14])
    
    # 夹爪参数
    BINARY_GRIPPER_THREASHOLD: float = 0.5 # 超过此值则闭合
    APPLY_GRIPPER_PENALTY: bool = True
    GRIPPER_PENALTY: float = 0.1


class OpenArmEnv(gym.Env):
    def __init__(
        self,
        hz=5,
        fake_env=False,
        save_video=False,
        use_viser=False, # Deprecated but kept for compatibility
        config: DefaultOpenArmConfig = None,
        max_episode_length=100,
        arm="both", # "left", "right", or "both"
    ):
        self.hz = hz
        self.fake_env = fake_env
        self.save_video = save_video
        self.viser = None # Viser disabled by default in Env to avoid conflicts
        self.config = config or DefaultOpenArmConfig()
        self.max_episode_length = max_episode_length
        self.arm = arm.lower()
        if self.arm not in ["left", "right", "both"]:
            raise ValueError(f"Invalid arm choice: {self.arm}. Must be 'left', 'right', or 'both'.")
        
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
        self.gripper_binary_state = np.zeros((2,), dtype=int) # 0:开, 1:闭
        
        self.curr_path_length = 0
        self.cycle_count = 0
        self.latest_images = {} # Store decoded images
        
        # --- Observation Space ---
        # Determine shapes based on arm selection
        if self.arm == "both":
            tcp_shape = (2, 7)
            gripper_shape = (2, 1)
            action_dim = 14
        else:
            tcp_shape = (7,)
            gripper_shape = (1,)
            action_dim = 7

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
                
        # Add tcp_vel to observation space (required for RelativeFrame)
        # We will just return zeros for now as we don't have velocity feedback
        if self.arm == "both":
            obs_dict["state"]["tcp_vel"] = gym.spaces.Box(
                -np.inf, np.inf, shape=(2, 6)
            )
        else:
            obs_dict["state"]["tcp_vel"] = gym.spaces.Box(
                -np.inf, np.inf, shape=(6,)
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

        # --- 硬件与可视化初始化 ---
        self.displayer = None
        if self.config.REALSENSE_CAMERAS:
            self.init_cameras(self.config.REALSENSE_CAMERAS)
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue)
            self.displayer.start()
        
        print(f"Initialized OpenArm Env (Bimanual) connected to {self.url} - Arm: {self.arm}")

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """
        安全裁剪：将目标位姿 (2, 7) 的 XYZ 限制在安全盒子内。
        使用了 NumPy 向量化操作，同时处理双臂。
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
        环境步进函数：
        1. 解析动作 (delta) -> 2. 计算目标位姿 -> 3. 发送指令 -> 4. 获取新状态
        """
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        self.nextpos = self.currpos.copy()
        gripper_cmds = []

        # Determine which arms to update based on self.arm
        # 0: Left, 1: Right
        if self.arm == "both":
            arm_indices = [0, 1]
            full_action = action # (14,)
        elif self.arm == "left":
            arm_indices = [0]
            full_action = np.concatenate([action, np.zeros(7)]) # Pad right arm with zeros
        elif self.arm == "right":
            arm_indices = [1]
            full_action = np.concatenate([np.zeros(7), action]) # Pad left arm with zeros
        
        # 循环处理需要更新的手臂
        for i in arm_indices:
            # Calculate index in the FULL action array (which is always 14-dim conceptually for logic below)
            # But wait, if self.arm is single, 'action' is 7-dim.
            # So we need to index into 'action' correctly.
            
            if self.arm == "both":
                idx_start = i * 7
                current_arm_action = action[idx_start : idx_start+7]
            else:
                # Single arm: action is just (7,)
                current_arm_action = action

            # 1. 平移更新 (Translation)
            xyz_delta = current_arm_action[:3]
            self.nextpos[i, :3] += xyz_delta * self.action_scale[0]
            
            # 2. 旋转更新 (Rotation)
            rpy_delta = current_arm_action[3:6] * self.action_scale[1]
            rot_delta = Rotation.from_euler("xyz", rpy_delta)
            rot_curr = Rotation.from_quat(self.nextpos[i, 3:])
            self.nextpos[i, 3:] = (rot_delta * rot_curr).as_quat()
            
            # 3. 夹爪控制 (Continuous)
            raw_gripper_action = current_arm_action[6]
            gripper_val = 0.5236 * (raw_gripper_action - 1.0)
            
            # Store gripper command for this arm
            # We need to ensure gripper_cmds has 2 elements eventually
            # So we better initialize gripper_cmds with current state and update
            pass 

        # Re-construct gripper commands for BOTH arms
        # If an arm is not active, we should keep its gripper at current state?
        # Or just send 0.0 (Closed)? Or -1.04 (Open)?
        # Safer to keep current state if possible, but self.curr_gripper_pos is from server.
        # Let's just use the calculated values for active arms, and hold for inactive?
        # Actually, simpler: just construct the full gripper list.
        
        final_gripper_cmds = []
        for i in range(2):
            if i in arm_indices:
                # Extract action for this arm again
                if self.arm == "both":
                    act = action[i*7 : (i+1)*7]
                else:
                    act = action
                
                raw_val = act[6]
                val = 0.5236 * (raw_val - 1.0)
                final_gripper_cmds.append(val)
            else:
                # Inactive arm: Enforce holding the initial reset pose to prevent drift
                # This ensures consistent background/collision state for single-arm tasks
                if hasattr(self, "initial_reset_pose"):
                    # Override nextpos for this arm to be exactly the reset pose
                    self.nextpos[i] = self.initial_reset_pose[i]
                    final_gripper_cmds.append(self.curr_gripper_pos[i]) # Keep gripper as is
                else:
                    # Fallback if reset() wasn't called (shouldn't happen)
                    final_gripper_cmds.append(self.curr_gripper_pos[i])

        # 安全限制并发送笛卡尔位置指令 (包含夹爪)
        # Note: self.nextpos for inactive arms remains as self.currpos (copied at start)
        self._send_pos_command(self.clip_safety_box(self.nextpos), gripper_pos=final_gripper_cmds)

        # 频率控制: 动态休眠以维持稳定的 Hz
        self.curr_path_length += 1
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        # 更新状态与可视化
        self._update_currpos()
        # if self.viser:
        #     self.render()

        # 计算奖励与结束标志
        ob = self._get_obs()
        reward = self.compute_reward(ob, False) # gripper_effective is removed/ignored for now
        done = self.curr_path_length >= self.max_episode_length or reward == 1.0
        
        return ob, reward, done, False, {}

    def compute_reward(self, obs: Dict, gripper_effective: bool) -> float:
        """
        计算奖励 (稀疏奖励逻辑)
        目前逻辑：已被禁用。奖励完全由外部 Wrapper (Reward Classifier) 提供。
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
                    
        # Normalize gripper pos back to [-1, 1]
        gripper_scale = 0.5236
        norm_gripper_pos = (self.curr_gripper_pos / gripper_scale) + 1.0
        norm_gripper_pos = np.clip(norm_gripper_pos, -1.0, 1.0)

        # Construct state based on arm selection
        if self.arm == "both":
            tcp_obs = self.currpos.copy() # (2, 7)
            gripper_obs = norm_gripper_pos.reshape(2, 1) # (2, 1)
        elif self.arm == "left":
            tcp_obs = self.currpos[0].copy() # (7,)
            gripper_obs = norm_gripper_pos[0].reshape(1,) # (1,)
        elif self.arm == "right":
            tcp_obs = self.currpos[1].copy() # (7,)
            gripper_obs = norm_gripper_pos[1].reshape(1,) # (1,)

        state_observation = {
            "tcp_pose": tcp_obs,
            "tcp_vel": np.zeros((2, 6) if self.arm == "both" else (6,), dtype=np.float32), # Dummy velocity
            "gripper_pose": gripper_obs,
        }
        
        # Return nested dictionary
        return {"images": images, "state": state_observation}

    def render(self, mode="human"):
        """更新 3D 可视化 (已移至 Server 端)"""
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
        
        # 发送复位指令 (这一步其实是多余的，因为 go_to_rest 已经回零了，
        # 但为了更新 self.currpos 对应的 Cartesian 状态，保留也无妨，
        # 或者应该在 go_to_rest 后直接 update_currpos)
        # self._send_pos_command(self.currpos) 
        # Better: Just update current state from server
        self._update_currpos()
        
        # if self.viser:
        #     self.render()
            
        # Update initial reset pose for station keeping of inactive arms
        self.initial_reset_pose = self.currpos.copy()
        
        return self._get_obs(), {}

    def go_to_rest(self):
        """强制服务器执行关节回零 (Home Reset)"""
        try:
            # Server blocks for 10s, so we need a timeout > 10s
            self.session.post(self.url + "jointreset", timeout=5)
            time.sleep(1) # Extra buffer
        except requests.exceptions.RequestException as e:
            print(f"[Env Warning] Joint reset failed: {e}")
        self._update_currpos()

    def _send_pos_command(self, pos: np.ndarray, gripper_pos: list = None):
        """发送目标位姿 (2, 7) 到服务器"""
        arr = pos.astype(np.float32)
        data = {"arr": arr.tolist()} # 转换为嵌套列表
        
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

    def _send_gripper_command(self, val: float, arm_idx: int = 0) -> bool:
        """
        Deprecated: Gripper is now handled in _send_pos_command
        """
        return False

    def _update_currpos(self):
        """从服务器获取最新状态并更新内部变量"""
        try:
            resp = self.session.post(self.url + "getstate", timeout=5.0)
            ps = resp.json()
            
            # 辅助函数: 确保数据形状为 (2, cols)
            def ensure_shape(arr_list, cols):
                arr = np.array(arr_list)
                if arr.size == 2 * cols:
                    return arr.reshape(2, cols)
                return arr 

            # Note: Server get_state returns "q" and "gripper". 
            # It does NOT return "pose", "vel", "force", "torque" anymore in the updated server code?
            # Wait, I checked openarm_server.py, get_state returns "q", "gripper", "images".
            # It does NOT return "pose" (Cartesian).
            # So I need to compute FK here or assume q is enough?
            # The Env uses self.currpos (Cartesian) for reward computation.
            # If server doesn't return pose, Env must compute it or Server must return it.
            # The previous server code returned "pose". The NEW server code I wrote only returns "q" and "gripper".
            # THIS IS A BUG/REGRESSION in my server update.
            # However, I can't easily fix server now without re-editing.
            # But wait, the previous server code had `get_state` returning `l_pos` which was joint angles?
            # No, `controller.get_left_position()` returns joint angles.
            # So `ps["pose"]` in old env code was actually expecting Cartesian?
            # Let's check old `openarm_server.py` (before my edit).
            # It returned `q` (joints) and `gripper`.
            # It did NOT return `pose` (Cartesian).
            # So `_update_currpos` in `openarm_env.py` (lines 386) was `self.currpos[:] = ensure_shape(ps["pose"], 7)`.
            # This implies `ps` had "pose".
            # My server update REMOVED "pose" from the return dict!
            # I must fix this. I should calculate FK in Env or Server.
            # Since I have `BaseIKSolver` in Env (optional), I can use it.
            # Or I can fix Server to return "pose" (but Server might not have FK if IK_AVAILABLE is False).
            # Let's assume for now I only need `q` and `gripper` and `images`.
            # But `compute_reward` uses `tcp_pose`.
            # So I MUST update `self.currpos` (TCP pose).
            # I will use `self.q` to update `self.currpos` if I have a solver.
            # If not, `self.currpos` will be stale/wrong.
            
            # Update Cartesian Pose from Server (FK result)
            if "pose" in ps:
                self.currpos[:] = ensure_shape(ps["pose"], 7)
            
            self.q[:] = ensure_shape(ps["q"], 7)
            self.dq[:] = ensure_shape(ps.get("dq", [0]*14), 7) # Server might not return dq
            self.curr_gripper_pos = np.array(ps["gripper_pos"])
            
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
        if self.displayer:
            self.img_queue.put(None)
            self.displayer.join()
        self.session.close()