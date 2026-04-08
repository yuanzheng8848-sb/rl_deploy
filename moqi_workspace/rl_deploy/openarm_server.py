
from sympy.printing.glsl import print_glsl
import sys
import time
import numpy as np
import threading
from pathlib import Path
from flask import Flask, request, jsonify
import cv2
import base64
from scipy.spatial.transform import Rotation
import importlib.util

# --- 配置 ---
USE_MOCK = False  # Set to True to use Mock Hardware, False for Real Hardware

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
        self.servo_lock = threading.Lock()
        self.latest_frames = {}
        self.running = True

        # Opt-in real-time servo state. Disabled by default to preserve training behavior.
        self.servo_enabled = False
        self.servo_hz = 80.0
        self.servo_trans_step = 0.004
        self.servo_rot_step = 0.012
        self.servo_gripper_step = 0.02
        self.servo_timeout = 0.25
        self.servo_pos_epsilon = 5e-4
        self.servo_rot_epsilon = 1e-2
        self.servo_target_pose = None
        self.servo_target_gripper = None
        self.servo_last_update_ts = 0.0
        self.servo_active_arms = (0, 1)
        self.servo_backend = "analytic"
        self.servo_debug = {
            "status": "idle",
            "last_error": "",
            "last_backend": "baseik",
            "last_active_arms": (0, 1),
            "last_target_pose": None,
            "last_current_pose": None,
            "last_q_target": None,
            "last_current_gripper": None,
            "last_target_gripper": None,
            "last_stepped_gripper": None,
            "last_update_ts": 0.0,
            "solve_fail_count": 0,
            "command_count": 0,
        }
        self.last_servo_log_ts = 0.0
        self.servo_thread = threading.Thread(target=self._servo_loop, daemon=True)
        self.servo_thread.start()
        
        # Initialize Cameras
        self.cameras = {}
        if not USE_MOCK:
            self._init_cameras()
            # Start camera thread
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
        self._init_analytic_servo()

    def _init_analytic_servo(self):
        """Initialize the VR-style analytic IK stack used by record_demo."""
        self.analytic_servo_ready = False
        self.analytic_triangle = None
        self.analytic_workspace_constraint = None
        self.analytic_collision_checker = None
        self.analytic_left_shoulder_position = None
        self.analytic_right_shoulder_position = None
        self.analytic_left_shoulder_orientation = None
        self.analytic_right_shoulder_orientation = None
        self.analytic_joints_upper_limit = None
        self.analytic_joints_lower_limit = None

        if not self.ik_solver:
            return

        try:
            pyroki_dir = ROOT_DIR / "pyroki"
            if str(pyroki_dir) not in sys.path:
                sys.path.insert(0, str(pyroki_dir))
            from workspace_constraint import create_openarm_constraint

            def _load_module(module_name: str, path: Path):
                spec = importlib.util.spec_from_file_location(module_name, str(path))
                module = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(module)
                return module

            ik_module = _load_module("analytic_IK_runtime", ROOT_DIR / "IK" / "analytic_IK.py")
            collision_module = _load_module("collision_check_runtime", ROOT_DIR / "IK" / "collision_check.py")

            origin_position = np.array([0.0, 0.0, 0.0])
            l1 = 0.22
            l2 = 0.216
            self.analytic_triangle = ik_module.Triangle(l1, l2, origin_position)
            current_ee_pose = self.ik_solver.get_current_ee_pose(np.zeros(14))
            self.analytic_triangle.set_init_ee_pose(current_ee_pose[0], current_ee_pose[1])
            self.analytic_workspace_constraint = create_openarm_constraint(l1=l1, l2=l2, safety_margin=0.016)

            T = self.ik_solver.forward_kinematics(np.zeros(14))
            left_shoulder_position = T[3][4:].copy()
            left_shoulder_position[1] += T[4][4:][1] - T[3][4:][1]
            right_shoulder_position = T[11][4:].copy()
            right_shoulder_position[1] += T[12][4:][1] - T[11][4:][1]
            shoulder_rot = Rotation.from_matrix(np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]))

            self.analytic_left_shoulder_position = left_shoulder_position
            self.analytic_right_shoulder_position = right_shoulder_position
            self.analytic_left_shoulder_orientation = shoulder_rot
            self.analytic_right_shoulder_orientation = shoulder_rot
            self.analytic_collision_checker = collision_module.OpenArmCollisionChecker(
                left_shoulder_position,
                right_shoulder_position,
                None,
            )
            self.analytic_joints_upper_limit = np.array(self.ik_solver._robot.joints.upper_limits) + 0.0001
            self.analytic_joints_lower_limit = np.array(self.ik_solver._robot.joints.lower_limits) - 0.0001
            self.analytic_servo_ready = True
            print("[Server] Analytic servo backend ready.")
        except Exception as e:
            print(f"[Server Warning] Analytic servo init failed: {e}")
            self.analytic_servo_ready = False

    def _init_cameras(self):
        """Initialize Realsense Cameras"""
        # Camera Configs
        self.cam_configs = {
            "head": {"type": "opencv", "device_id": "/dev/video10", "width": 1280, "height": 960, "fps": 30, "exposure": 150},
            "left": {"type": "realsense", "serial": "150622074105", "width": 640, "height": 480, "fps": 30},
            "right": {"type": "realsense", "serial": "236422072385", "width": 640, "height": 480, "fps": 30}
        }
        
        try:
            for name, cfg in self.cam_configs.items():
                print(f"Initializing Camera {name}...")
                if cfg["type"] == "opencv":
                     from connection.cameras import OpenCVCamera
                     cam = OpenCVCamera(
                        device_id=cfg["device_id"], 
                        width=cfg["width"], 
                        height=cfg["height"], 
                        fps=cfg["fps"], 
                        exposure=cfg.get("exposure", None)
                     )
                else:
                    cam = RealsenseCamera(
                        device_id=cfg["serial"], 
                        width=cfg["width"], 
                        height=cfg["height"], 
                        fps=cfg["fps"], 
                        enable_depth=False
                    )
                # cam.start() # Auto-started in __init__
                self.cameras[name] = cam
                self.latest_frames[name] = None
                print(f"Initialized {name} camera.")
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

    def _get_current_joint_and_gripper(self):
        with self.lock:
            q_l_curr, g_l_curr = self.controller.get_left_position()
            q_r_curr, g_r_curr = self.controller.get_right_position()

        if q_l_curr is None:
            q_l_curr = np.zeros(7)
        if q_r_curr is None:
            q_r_curr = np.zeros(7)
        if not g_l_curr:
            g_l_curr = [0.0]
        if not g_r_curr:
            g_r_curr = [0.0]
        q_curr = np.concatenate([q_l_curr, q_r_curr])
        g_curr = np.array([g_l_curr[0], g_r_curr[0]], dtype=np.float64)
        return q_curr, g_curr

    def _get_current_pose(self, q_curr):
        pose = np.zeros((2, 7), dtype=np.float64)
        if self.ik_solver:
            ik_poses = self.ik_solver.get_current_ee_pose(q_curr)
            for i in range(2):
                p = ik_poses[i]
                pose[i, :3] = p[4:7]
                pose[i, 3:6] = p[1:4]
                pose[i, 6] = p[0]
        return pose

    def _step_pose_towards_target(self, current_pose, target_pose):
        next_pose = np.array(current_pose, copy=True)
        for i in range(2):
            pos_delta = target_pose[i, :3] - current_pose[i, :3]
            pos_norm = np.linalg.norm(pos_delta)
            if pos_norm > self.servo_trans_step and pos_norm > 1e-9:
                pos_delta = pos_delta * (self.servo_trans_step / pos_norm)
            next_pose[i, :3] = current_pose[i, :3] + pos_delta

            rot_curr = Rotation.from_quat(current_pose[i, 3:])
            rot_tgt = Rotation.from_quat(target_pose[i, 3:])
            rot_err = rot_curr.inv() * rot_tgt
            rotvec = rot_err.as_rotvec()
            rot_mag = np.linalg.norm(rotvec)
            if rot_mag > self.servo_rot_step and rot_mag > 1e-9:
                rotvec = rotvec * (self.servo_rot_step / rot_mag)
            next_pose[i, 3:] = (rot_curr * Rotation.from_rotvec(rotvec)).as_quat()
        return next_pose

    def _pose_error_small(self, current_pose, target_pose, active_arms):
        for arm_idx in active_arms:
            pos_err = np.linalg.norm(target_pose[arm_idx, :3] - current_pose[arm_idx, :3])
            rot_curr = Rotation.from_quat(current_pose[arm_idx, 3:])
            rot_tgt = Rotation.from_quat(target_pose[arm_idx, 3:])
            rot_err = (rot_curr.inv() * rot_tgt).magnitude()
            if pos_err > self.servo_pos_epsilon or rot_err > self.servo_rot_epsilon:
                return False
        return True

    def _gripper_error_small(self, current_gripper, target_gripper, active_arms):
        for arm_idx in active_arms:
            if abs(target_gripper[arm_idx] - current_gripper[arm_idx]) > self.servo_gripper_step:
                return False
        return True

    def _step_gripper_towards_target(self, current_gripper, target_gripper, active_arms):
        next_gripper = np.array(current_gripper, copy=True)
        for i in range(2):
            # Right-arm-only teleop: send the right gripper command directly so it can
            # overcome static friction and match the VR path more closely.
            if tuple(active_arms) == (1,) and i == 1:
                next_gripper[i] = target_gripper[i]
                continue
            delta = target_gripper[i] - current_gripper[i]
            if abs(delta) > self.servo_gripper_step:
                delta = np.sign(delta) * self.servo_gripper_step
            next_gripper[i] = current_gripper[i] + delta
        return next_gripper

    def _pose_to_ik_target(self, pose):
        target_ik = np.zeros((2, 7), dtype=np.float64)
        for i in range(2):
            target_ik[i, 0] = pose[i, 6]
            target_ik[i, 1:4] = pose[i, 3:6]
            target_ik[i, 4:7] = pose[i, :3]
        return target_ik

    def _analytic_solve(self, q_curr, target_pose, active_arms):
        if not self.analytic_servo_ready:
            self._update_servo_debug(status="analytic_not_ready", error="analytic backend not ready")
            return None

        current_pose_ik = self.ik_solver.get_current_ee_pose(q_curr)
        self.analytic_triangle.set_init_ee_pose(
            np.array(current_pose_ik[0], copy=True),
            np.array(current_pose_ik[1], copy=True),
        )

        target_ik = self._pose_to_ik_target(target_pose)
        for arm_idx in range(2):
            if arm_idx not in active_arms:
                target_ik[arm_idx] = current_pose_ik[arm_idx]

        left_constrained, right_constrained = self.analytic_workspace_constraint.constrain_dual_arm(
            target_ik[0],
            target_ik[1],
            self.analytic_left_shoulder_position,
            self.analytic_right_shoulder_position,
        )

        solved, left_arm_cmd, right_arm_cmd = self.analytic_triangle.solve(
            self.analytic_left_shoulder_position,
            self.analytic_left_shoulder_orientation,
            self.analytic_right_shoulder_position,
            self.analytic_right_shoulder_orientation,
            [left_constrained, right_constrained],
            self.analytic_collision_checker,
            self.analytic_joints_lower_limit,
            self.analytic_joints_upper_limit,
        )
        if not solved:
            self._update_servo_debug(
                status="analytic_solve_failed",
                error="Triangle.solve returned unsolved",
                target_pose=target_pose,
                current_pose=self._get_current_pose(q_curr),
            )
            return None

        q_target = np.array(q_curr, copy=True)
        if 0 in active_arms and left_arm_cmd is not None:
            q_target[:7] = left_arm_cmd
        if 1 in active_arms and right_arm_cmd is not None:
            q_target[7:] = right_arm_cmd
        return q_target

    def _update_servo_debug(
        self,
        status=None,
        error=None,
        backend=None,
        active_arms=None,
        target_pose=None,
        current_pose=None,
        q_target=None,
        current_gripper=None,
        target_gripper=None,
        stepped_gripper=None,
        solve_fail_delta=0,
        command_delta=0,
    ):
        with self.servo_lock:
            if status is not None:
                self.servo_debug["status"] = status
            if error is not None:
                self.servo_debug["last_error"] = error
            if backend is not None:
                self.servo_debug["last_backend"] = backend
            if active_arms is not None:
                self.servo_debug["last_active_arms"] = tuple(active_arms)
            if target_pose is not None:
                self.servo_debug["last_target_pose"] = np.array(target_pose, copy=True)
            if current_pose is not None:
                self.servo_debug["last_current_pose"] = np.array(current_pose, copy=True)
            if q_target is not None:
                self.servo_debug["last_q_target"] = np.array(q_target, copy=True)
            if current_gripper is not None:
                self.servo_debug["last_current_gripper"] = np.array(current_gripper, copy=True)
            if target_gripper is not None:
                self.servo_debug["last_target_gripper"] = np.array(target_gripper, copy=True)
            if stepped_gripper is not None:
                self.servo_debug["last_stepped_gripper"] = np.array(stepped_gripper, copy=True)
            self.servo_debug["solve_fail_count"] += int(solve_fail_delta)
            self.servo_debug["command_count"] += int(command_delta)
            self.servo_debug["last_update_ts"] = time.time()

    def _maybe_log_servo_debug(self):
        now = time.time()
        if now - self.last_servo_log_ts < 0.5:
            return
        self.last_servo_log_ts = now
        with self.servo_lock:
            dbg = dict(self.servo_debug)
        target_pose = dbg.get("last_target_pose")
        current_pose = dbg.get("last_current_pose")
        current_gripper = dbg.get("last_current_gripper")
        target_gripper = dbg.get("last_target_gripper")
        stepped_gripper = dbg.get("last_stepped_gripper")
        tgt_xyz = None if target_pose is None else np.round(np.array(target_pose)[list(dbg["last_active_arms"])[-1], :3], 4).tolist()
        cur_xyz = None if current_pose is None else np.round(np.array(current_pose)[list(dbg["last_active_arms"])[-1], :3], 4).tolist()
        print(
            "[ServoDebug] "
            f"status={dbg['status']} backend={dbg['last_backend']} active_arms={dbg['last_active_arms']} "
            f"fails={dbg['solve_fail_count']} cmds={dbg['command_count']} "
            f"target_xyz={tgt_xyz} current_xyz={cur_xyz} "
            f"gripper_cur={None if current_gripper is None else np.round(np.array(current_gripper), 4).tolist()} "
            f"gripper_tgt={None if target_gripper is None else np.round(np.array(target_gripper), 4).tolist()} "
            f"gripper_step={None if stepped_gripper is None else np.round(np.array(stepped_gripper), 4).tolist()} "
            f"error={dbg['last_error']}"
        )

    def _servo_loop(self):
        print("Starting Servo Loop...")
        while self.running:
            time.sleep(max(1.0 / self.servo_hz, 0.001))

            with self.servo_lock:
                enabled = self.servo_enabled
                target_pose = None if self.servo_target_pose is None else np.array(self.servo_target_pose, copy=True)
                target_gripper = None if self.servo_target_gripper is None else np.array(self.servo_target_gripper, copy=True)
                last_update_ts = self.servo_last_update_ts
                active_arms = tuple(self.servo_active_arms)
                backend = self.servo_backend

            if not enabled or target_pose is None or not self.ik_solver:
                continue

            q_curr, g_curr = self._get_current_joint_and_gripper()
            current_pose = self._get_current_pose(q_curr)
            self._update_servo_debug(
                status="target_received",
                backend=backend,
                active_arms=active_arms,
                target_pose=target_pose,
                current_pose=current_pose,
                current_gripper=g_curr,
                target_gripper=target_gripper,
            )

            if time.time() - last_update_ts > self.servo_timeout:
                target_pose = current_pose
                target_gripper = g_curr
            elif target_gripper is None:
                target_gripper = g_curr

            pose_is_small = self._pose_error_small(current_pose, target_pose, active_arms)
            grip_is_small = self._gripper_error_small(g_curr, target_gripper, active_arms)
            if pose_is_small and grip_is_small:
                self._update_servo_debug(
                    status="pose_and_gripper_already_satisfied",
                    backend=backend,
                    active_arms=active_arms,
                    target_pose=target_pose,
                    current_pose=current_pose,
                    current_gripper=g_curr,
                    target_gripper=target_gripper,
                )
                continue

            if backend == "analytic":
                stepped_pose = np.array(target_pose, copy=True)
            else:
                stepped_pose = np.array(current_pose, copy=True) if pose_is_small else self._step_pose_towards_target(current_pose, target_pose)
            stepped_gripper = self._step_gripper_towards_target(g_curr, target_gripper, active_arms)
            inactive_arms = [idx for idx in range(2) if idx not in active_arms]
            for arm_idx in inactive_arms:
                stepped_pose[arm_idx] = current_pose[arm_idx]
            if backend == "analytic":
                q_target = self._analytic_solve(q_curr, stepped_pose, active_arms)
            else:
                target_ik = self._pose_to_ik_target(stepped_pose)
                q_target = self.ik_solver.solve_ik(target_ik, q_curr)
            if q_target is None or np.any(np.isnan(q_target)):
                self._update_servo_debug(
                    status="solve_failed",
                    error="q_target is None or NaN",
                    backend=backend,
                    active_arms=active_arms,
                    target_pose=stepped_pose,
                    current_pose=current_pose,
                    current_gripper=g_curr,
                    target_gripper=target_gripper,
                    stepped_gripper=stepped_gripper,
                    solve_fail_delta=1,
                )
                self._maybe_log_servo_debug()
                continue

            with self.lock:
                if 0 in active_arms:
                    self.controller.set_left_position(q_target[:7], float(stepped_gripper[0]), q_curr[:7], float(g_curr[0]))
                if 1 in active_arms:
                    self.controller.set_right_position(q_target[7:], float(stepped_gripper[1]), q_curr[7:], float(g_curr[1]))
            self._update_servo_debug(
                status="command_sent",
                backend=backend,
                active_arms=active_arms,
                target_pose=stepped_pose,
                current_pose=current_pose,
                q_target=q_target,
                current_gripper=g_curr,
                target_gripper=target_gripper,
                stepped_gripper=stepped_gripper,
                command_delta=1,
            )
            self._maybe_log_servo_debug()

            if self.viser:
                vis_joints = np.array(q_curr, copy=True)
                if 0 in active_arms:
                    vis_joints[:7] = q_target[:7]
                if 1 in active_arms:
                    vis_joints[7:] = q_target[7:]
                self.viser.update_joints(vis_joints)

    def start_servo(
        self,
        target_pose_flat=None,
        gripper_pos=None,
        servo_hz=80.0,
        trans_step=0.004,
        rot_step=0.012,
        gripper_step=0.02,
        arm="both",
        backend="analytic",
    ):
        if not self.ik_solver:
            print("[Server] Cannot start servo: No solver initialized.")
            return False
        if backend == "analytic" and not self.analytic_servo_ready:
            print("[Server] Cannot start analytic servo: backend not ready.")
            return False

        q_curr, g_curr = self._get_current_joint_and_gripper()
        current_pose = self._get_current_pose(q_curr)

        if target_pose_flat is None:
            target_pose = current_pose
        else:
            target_pose = np.array(target_pose_flat, dtype=np.float64).reshape(2, 7)

        if gripper_pos is None:
            target_gripper = g_curr
        else:
            target_gripper = np.array(gripper_pos, dtype=np.float64)

        active_arms_map = {"left": (0,), "right": (1,), "both": (0, 1)}
        active_arms = active_arms_map.get(str(arm).lower())
        if active_arms is None:
            print(f"[Server] Invalid servo arm: {arm}")
            return False

        with self.servo_lock:
            self.servo_enabled = True
            self.servo_hz = float(servo_hz)
            self.servo_trans_step = float(trans_step)
            self.servo_rot_step = float(rot_step)
            self.servo_gripper_step = float(gripper_step)
            self.servo_target_pose = target_pose
            self.servo_target_gripper = target_gripper
            self.servo_last_update_ts = time.time()
            self.servo_active_arms = active_arms
            self.servo_backend = str(backend).lower()
        return True

    def update_servo_target(self, target_pose_flat, gripper_pos=None):
        target_pose = np.array(target_pose_flat, dtype=np.float64).reshape(2, 7)
        with self.servo_lock:
            if not self.servo_enabled:
                return False
            self.servo_target_pose = target_pose
            if gripper_pos is not None:
                self.servo_target_gripper = np.array(gripper_pos, dtype=np.float64)
            self.servo_last_update_ts = time.time()
        return True

    def stop_servo(self):
        with self.servo_lock:
            self.servo_enabled = False
            self.servo_target_pose = None
            self.servo_target_gripper = None
            self.servo_last_update_ts = 0.0
            self.servo_active_arms = (0, 1)
            self.servo_backend = "analytic"
        return True

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
        response = {
            "pose": pose.tolist(),
            "q": q.tolist(),
            "gripper_pos": gripper.tolist(),
            "vel": [0.0] * 12,     # 笛卡尔速度
            "dq": [0.0] * 14,      # 关节速度
            "force": [0.0] * 6,
            "torque": [0.0] * 6,
        }
        
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

        response["images"] = encoded_images
        return response

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


@app.route("/servo/start", methods=["POST"])
def route_servo_start():
    try:
        payload = request.json or {}
        arr = payload.get("arr")
        gripper = payload.get("gripper")
        servo_hz = payload.get("servo_hz", 80.0)
        trans_step = payload.get("trans_step", 0.012)
        rot_step = payload.get("rot_step", 0.008)
        gripper_step = payload.get("gripper_step", 0.02)
        arm = payload.get("arm", "both")
        backend = payload.get("backend", "baseik")
        if server.start_servo(
            arr,
            gripper_pos=gripper,
            servo_hz=servo_hz,
            trans_step=trans_step,
            rot_step=rot_step,
            gripper_step=gripper_step,
            arm=arm,
            backend=backend,
        ):
            return "OK", 200
        return "Servo Start Fail", 500
    except Exception as e:
        print(f"[API Error] servo/start: {e}")
        return str(e), 500


@app.route("/servo/target", methods=["POST"])
def route_servo_target():
    try:
        payload = request.json or {}
        arr = payload.get("arr")
        gripper = payload.get("gripper")
        if arr is None:
            return "Missing array", 400
        if server.update_servo_target(arr, gripper_pos=gripper):
            return "OK", 200
        return "Servo Not Enabled", 409
    except Exception as e:
        print(f"[API Error] servo/target: {e}")
        return str(e), 500


@app.route("/servo/stop", methods=["POST"])
def route_servo_stop():
    try:
        server.stop_servo()
        return "OK", 200
    except Exception as e:
        print(f"[API Error] servo/stop: {e}")
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
