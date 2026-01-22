
from sympy.printing.glsl import print_glsl
import sys
import time
import numpy as np
import threading
from pathlib import Path
from flask import Flask, request, jsonify
import cv2
import base64

# --- 配置 ---
USE_MOCK = True  # Set to True to use Mock Hardware, False for Real Hardware

# --- 路径配置 ---
ROOT_DIR = Path(__file__).resolve().parent.parent

# --- 导入路径 ---
sys.path.append(str(ROOT_DIR / "openarm"))
sys.path.append(str(ROOT_DIR / "pyroki"))
from realsense_camera import RealsenseCamera

# --- 导入控制器 ---
if USE_MOCK:
    print(">>> MODE: MOCK HARDWARE <<<")
    try:
        from mock_hardware import MockOpenArmController as HardwareController
    except ImportError:
        # Fallback if mock_hardware is not found in path, though it should be in rl_deploy
        sys.path.append(str(ROOT_DIR / "rl_deploy"))
        from mock_hardware import MockOpenArmController as HardwareController
else:
    print(">>> MODE: REAL HARDWARE <<<")
    try:
        from openarm_controller_2 import OpenArmController as HardwareController
    except ImportError as e:
        print(f"[Server Error] Cannot import OpenArmController: {e}")
        print("Falling back to Mock Hardware due to import error.")
        from mock_hardware import MockOpenArmController as HardwareController
        USE_MOCK = True

# --- 导入 IK 求解器 (可选但推荐) ---
# 用于将 Gym 的笛卡尔指令转换为 Controller 的关节指令
try:
    from robot_ik_solver import BaseIKSolver
    from viser_base import ViserBase
    import yaml
    IK_AVAILABLE = True
except ImportError:
    print("[Server Warning] robot_ik_solver/pyroki not found. Cartesian control will fail.")
    BaseIKSolver = None
    ViserBase = None
    IK_AVAILABLE = False

app = Flask(__name__)

class OpenArmServer:
    def __init__(self):
        print(f"Initializing {'Mock' if USE_MOCK else 'Real'} OpenArm Hardware Controller...")
        # 1. 初始化硬件控制器 (双臂)
        self.controller = HardwareController(enable_left=True, enable_right=True)
        
        # 2. 初始化 IK 求解器 和 Viser
        self.ik_solver = None
        self.viser = None
        if IK_AVAILABLE:
            self._init_ik_and_viser()

        # 用于线程安全的锁 (虽然 Flask 是多线程的，但 CAN 通讯通常不是线程安全的)
        self.lock = threading.Lock()
        self.camera_lock = threading.Lock()
        self.latest_frames = {}
        
        # Initialize Cameras
        self.cameras = {}
        if not USE_MOCK:
            self._init_cameras()
            # Start camera thread
            self.running = True
            self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
            self.camera_thread.start()

    def _init_ik_and_viser(self):
        """初始化逆运动学求解器和可视化"""
        try:
            cfg_path = ROOT_DIR / "pyroki" / "config"
            with open(cfg_path / "robot.yaml") as f: r_cfg = yaml.safe_load(f)
            with open(cfg_path / "solver.yaml") as f: s_cfg = yaml.safe_load(f)
            with open(cfg_path / "viser.yaml") as f: v_cfg = yaml.safe_load(f)
            
            # 修正 URDF 路径指向
            r_cfg["description"]["package_path"] = str(ROOT_DIR / "openarm")
            
            self.ik_solver = BaseIKSolver(s_cfg, r_cfg, visualize_collision=False)
            
            # JAX Warmup (预编译 IK 计算图)
            print("Warming up JAX IK Solver...")
            dummy_q = np.zeros(14)
            # 这里的 target 格式取决于求解器，假设为 [w, x, y, z, px, py, pz]
            dummy_target = np.array([
                [1,0,0,0, 0.3, 0.2, 0.3], 
                [1,0,0,0, 0.3, -0.2, 0.3]
            ])
            self.ik_solver.solve_ik(dummy_target, dummy_q)
            print("IK Solver Ready.")

            # 初始化 Viser
            if ViserBase:
                print("Initializing Viser...")
                v_cfg["nb_vis_frames"] = 6
                self.viser = ViserBase(
                    v_cfg, self.ik_solver.urdf,
                    self.ik_solver.get_actuated_joint_order(),
                    self.ik_solver.get_target_link_indices(),
                    self.ik_solver.forward_kinematics,
                    use_sim=False, use_teleop=False
                )
                print("Viser Initialized.")

        except Exception as e:
            print(f"[Server Warning] IK/Viser Init Failed: {e}")
            self.ik_solver = None
            self.viser = None

    def _init_cameras(self):
        """Initialize Realsense Cameras"""
        # Camera Configs
        self.cam_configs = {
            "head": {"serial": "248622302807", "width": 1280, "height": 720, "fps": 30},
            "left": {"serial": "150622074105", "width": 640, "height": 480, "fps": 30},
            "right": {"serial": "236422072385", "width": 640, "height": 480, "fps": 30}
        }
        
        try:
            for name, cfg in self.cam_configs.items():
                print(f"Initializing Camera {name} ({cfg['serial']})...")
                cam = RealsenseCamera(device_id=cfg["serial"], width=cfg["width"], height=cfg["height"], fps=cfg["fps"], enable_depth=False)
                # cam.start() # Auto-started in __init__
                self.cameras[name] = cam
                self.latest_frames[name] = None
            print("All cameras initialized.")
        except Exception as e:
            print(f"[Server Warning] Camera Init Failed: {e}")

    def _camera_loop(self):
        """Background thread to read camera frames"""
        print("Starting Camera Loop...")
        while self.running:
            for name, cam in self.cameras.items():
                img, _ = cam.get_data()
                if img is not None:
                    with self.camera_lock:
                        self.latest_frames[name] = img
            time.sleep(0.01)

    def get_state(self):
        """获取机器人状态 (包含图像)"""
        with self.lock:
            # 读取关节和夹爪状态
            l_pos, l_grip = self.controller.get_left_position()
            r_pos, r_grip = self.controller.get_right_position()
            
        # 处理 None
        if l_pos is None: l_pos = np.zeros(7)
        if r_pos is None: r_pos = np.zeros(7)
        if not l_grip: l_grip = [0.0]
        if not r_grip: r_grip = [0.0]
        
        # 拼接关节数据 (14维)
        q = np.concatenate([l_pos, r_pos])
        
        # 拼接夹爪数据 (2维)
        gripper = np.array([l_grip[0], r_grip[0]])

        # 计算末端位姿 (Forward Kinematics)
        # 即使没有 IK 求解器，也尽量返回数据，但 pose 将为 0
        pose = np.zeros(14)
        if self.ik_solver:
            # IK Solver 通常返回: [w, qx, qy, qz, px, py, pz]
            # Gym Env 通常期望: [px, py, pz, rx, ry, rz, rw] (OpenArmEnv logic)
            ik_poses = self.ik_solver.get_current_ee_pose(q)
            for i in range(2):
                p = ik_poses[i]
                # 转换格式: [w, x, y, z, x, y, z] -> [x, y, z, x, y, z, w]
                pose[i*7 : i*7+3] = p[4:7] # Pos
                pose[i*7+3 : i*7+6] = p[1:4] # Rot (quat xyz)
                pose[i*7+6] = p[0]         # Rot (quat w)

        # 构造返回字典
        # controller 2.0 暂时没有直接返回速度和力矩，这里填 0 防止 Env 报错
        
        # 更新可视化
        if self.viser:
            self.viser.update_joints(q)

        # 获取最新图像并编码
        encoded_images = {}
        with self.camera_lock:
            for name, frame in self.latest_frames.items():
                if frame is not None:
                    try:
                        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        b64_str = base64.b64encode(buffer).decode('utf-8')
                        encoded_images[name] = b64_str
                    except Exception as e:
                        print(f"[Server Error] Encode image {name} failed: {e}")

        return {
            "pose": pose.tolist(),
            "q": q.tolist(),
            "gripper_pos": gripper.tolist(),
            "vel": [0.0] * 12,     # 笛卡尔速度
            "dq": [0.0] * 14,      # 关节速度
            "force": [0.0] * 6,
            "torque": [0.0] * 6,
            "images": encoded_images
        }

    def move_ik(self, target_pose_flat, duration=1, gripper_pos=None):
        """
        笛卡尔空间控制
        Args:
            target_pose_flat: list/array, length 14. [L_pos(3), L_quat(4), R_pos(3), R_quat(4)]
                              Env 发送格式通常为 [px, py, pz, qx, qy, qz, qw]
            gripper_pos: list/array, length 2. [L_gripper, R_gripper] (Optional)
        """
        if not self.ik_solver:
            print("[Server] Cannot move_ik: No solver initialized.")
            return False

        # 1. 获取当前关节角作为 IK 迭代初值
        with self.lock:
            q_l_curr, _ = self.controller.get_left_position()
            q_r_curr, _ = self.controller.get_right_position()
        
        if q_l_curr is None: q_l_curr = np.zeros(7)
        if q_r_curr is None: q_r_curr = np.zeros(7)
        q_curr = np.concatenate([q_l_curr, q_r_curr])

        # 2. 转换目标格式 (Env -> IK Solver)
        # Env input: [px, py, pz, qx, qy, qz, qw]
        # IK expects: [w, qx, qy, qz, px, py, pz] (假设 robot_ik_solver 遵循此约定)
        target_pose_input = np.array(target_pose_flat).reshape(2, 7)
        target_ik = np.zeros((2, 7))
        
        for i in range(2):
            pos = target_pose_input[i, :3]
            quat = target_pose_input[i, 3:] # qx, qy, qz, qw
            
            target_ik[i, 0] = quat[3]   # w
            target_ik[i, 1:4] = quat[0:3] # x, y, z
            target_ik[i, 4:7] = pos     # px, py, pz

        # 3. 求解 IK
        q_target = self.ik_solver.solve_ik(target_ik, q_curr)
        
        if q_target is None or np.any(np.isnan(q_target)):
            print("[Server] IK Solution Failed (NaN or None)")
            return False

        # 4. 执行关节移动 (平滑)
        return self.move_joints(q_target, duration=duration, gripper_pos=gripper_pos)

    def move_joints(self, joints, duration=3, gripper_pos=None):
        """
        移动关节到指定位置 (平滑移动)
        Args:
            joints: 14维关节角度列表 [left_7, right_7]
            duration: 移动时间 (秒)
            gripper_pos: 2维夹爪位置列表 [left, right] (Optional)
        """
            
        try:
            print(f"[Server] move_joints called with duration={duration}")
            joints = np.array(joints)
            if joints.shape != (14,):
                print(f"[Server] Invalid joints shape: {joints.shape}")
                return False
                
            left_target = joints[:7]
            right_target = joints[7:]
            print(f"[Server] Target Left: {left_target}")
            print(f"[Server] Target Right: {right_target}")
            
            with self.lock:
                # Reading current position (outside the loop, as start point for interpolation)
                left_current, g_l_current = self.controller.get_left_position()
                right_current, g_r_current = self.controller.get_right_position()

                # Handle potential None values for current positions
                if left_current is None: left_current = np.zeros(7)
                if right_current is None: right_current = np.zeros(7)
                if not g_l_current: g_l_current = [0.0]
                if not g_r_current: g_r_current = [0.0]
                
                # Extract current gripper values
                # If gripper_pos is provided, use it as target. Otherwise keep current.
                if gripper_pos is not None:
                    g_l_target = gripper_pos[0]
                    g_r_target = gripper_pos[1]
                else:
                    g_l_target = g_l_current[0]
                    g_r_target = g_r_current[0]

                # Unified Smooth Movement Logic (for both Real and Mock)
                start_time = time.time()
                # Aim for ~50Hz update rate
                steps = int(duration * 50) 
                if steps == 0: steps = 1 # Ensure at least one step for very short durations
                step_interval = duration / steps

                if duration <= 0:
                    # Direct Control Mode (No Smoothing)
                    # Used for high-frequency control (e.g. RL step)
                    
                    # Send Command directly
                    self.controller.set_left_position(left_target, g_l_target, left_current, g_l_current[0])
                    self.controller.set_right_position(right_target, g_r_target, right_current, g_r_current[0])
                    
                    if self.viser:
                        self.viser.update_joints(np.concatenate([left_target, right_target]))
                        
                    return True

                # Smooth Movement Mode
                print(f"[Server] Smooth move start: {duration}s, steps: {steps}")
                
                for i in range(steps + 1): # +1 to ensure target is reached at the end
                    elapsed = time.time() - start_time
                    progress = min(elapsed / duration, 1.0)
                    
                    # Smoothstep interpolation
                    t = progress
                    smooth_progress = t * t * (3.0 - 2.0 * t)

                    # Interpolate joint commands
                    left_cmd = left_current + (left_target - left_current) * smooth_progress
                    right_cmd = right_current + (right_target - right_current) * smooth_progress
                    
                    # Interpolate gripper commands (if moving)
                    # Note: Gripper usually moves fast, but smoothing is safer if duration is long.
                    # If gripper_pos was not provided, g_l_target == g_l_current[0], so it stays still.
                    g_l_cmd = g_l_current[0] + (g_l_target - g_l_current[0]) * smooth_progress
                    g_r_cmd = g_r_current[0] + (g_r_target - g_r_current[0]) * smooth_progress
                    
                    # Get actual current state for set_position (PD control)
                    # This ensures the PD controller has the most up-to-date feedback
                    curr_l_real, g_l_real = self.controller.get_left_position()
                    curr_r_real, g_r_real = self.controller.get_right_position()
                    
                    # Handle potential None values for real current positions
                    if curr_l_real is None: curr_l_real = left_cmd # Fallback to command if read fails
                    if curr_r_real is None: curr_r_real = right_cmd # Fallback to command if read fails
                    if not g_l_real: g_l_real = g_l_current # Fallback to initial gripper if read fails
                    if not g_r_real: g_r_real = g_r_current # Fallback to initial gripper if read fails

                    # Send Command using set_position
                    # We pass the interpolated `left_cmd`/`right_cmd` as the target.
                    # We pass the actual `curr_l_real`/`curr_r_real` as current for PD calculation.
                    self.controller.set_left_position(left_cmd, g_l_cmd, curr_l_real, g_l_real[0])
                    self.controller.set_right_position(right_cmd, g_r_cmd, curr_r_real, g_r_real[0])
                    
                    # Update Viser with the interpolated command
                    if self.viser:
                        self.viser.update_joints(np.concatenate([left_cmd, right_cmd]))
                        
                    # Sleep to control update rate
                    time_to_sleep = start_time + (i + 1) * step_interval - time.time()
                    if time_to_sleep > 0:
                        time.sleep(time_to_sleep)

            return True
        except Exception as e:
            print(f"[Server] move_joints failed: {e}")
            import traceback
            traceback.print_exc()
            return False



# --- 实例化 Server ---
server = OpenArmServer()

# --- Flask 路由定义 ---

@app.route("/getstate", methods=["POST"])
def route_get_state():
    try:
        return jsonify(server.get_state())
    except Exception as e:
        print(f"[API Error] getstate: {e}")
        return str(e), 500

@app.route("/pose", methods=["POST"])
def route_pose():
    """接收笛卡尔位姿指令 -> IK -> 关节控制"""
    try:
        arr = request.json.get("arr")
        gripper = request.json.get("gripper") # Optional: [left, right]
        duration = request.json.get("duration", 3)
        if arr is None:
            return "Missing array", 400
        
        if server.move_ik(arr, duration=duration, gripper_pos=gripper):
            return "OK", 200
        else:
            return "IK Fail", 500
    except Exception as e:
        print(f"[API Error] pose: {e}")
        return str(e), 500

@app.route("/move_joints", methods=["POST"])
def route_move_joints():
    """直接接收关节角度指令"""
    try:
        joints = request.json.get("joints")
        gripper = request.json.get("gripper") # Optional: [left, right]
        if joints is None:
            return "Missing joints", 400
            
        if server.move_joints(joints, gripper_pos=gripper):
            return "OK", 200
        return "Fail", 500
    except Exception as e:
        print(f"[API Error] move_joints: {e}")
        return str(e), 500



@app.route("/jointreset", methods=["POST"])
def route_reset():
    """
    复位接口
    用于 Episode 结束或开始时将机器人移动到安全位置
    """
    try:
        # 定义一个安全的 Home 位置 (弧度)
        # 这里的 14维数组需要根据实际机器人的 "Zero" 姿态调整
        # 参考 Controller 2.0 中的 target
        # Updated Home Position from openarm_controller_2.py
        # Left Arm (0-6)
        home_pos_l = [-0.166811, -0.497863 , 0.635447, 1.499999, -0.627859, 0.507960, -0.168161]
        # Right Arm (7-13)
        home_pos_r = [0.166811, 0.497863, -0.635447, 1.499999, 0.627859, -0.507960, 0.168161]
        home_pos = np.concatenate([home_pos_l, home_pos_r])
        
        # 使用 controller 自带的平滑移动更好，但那是阻塞的。
        # 既然是 Reset，阻塞一下也没关系。
        # 或者直接调用 move_joints (非平滑，直接 PID)
        server.move_joints(home_pos, duration=3)
        
        return "OK", 200
    except Exception as e:
        print(f"[API Error] reset: {e}")
        return str(e), 500

if __name__ == "__main__":
    # 启动 Flask 服务
    # threaded=True 允许并发请求（虽然 CAN 操作加了锁）
    print("Starting OpenArm Server on port 5000...")
    app.run(host="0.0.0.0", port=5000, threaded=True)