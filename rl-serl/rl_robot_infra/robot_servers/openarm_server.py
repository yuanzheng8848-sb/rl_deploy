import os

# This process only serves hardware/camera/IK requests and should never reserve
# training GPU memory. Force all downstream frameworks (e.g. JAX) onto CPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from sympy.printing.glsl import print_glsl
import time
import numpy as np
import threading
from pathlib import Path
from flask import Flask, request, jsonify
import cv2
import base64
from scipy.spatial.transform import Rotation
import yaml

from openarm_env.camera.camera_factory import build_camera, camera_frame

# --- Paths ---

# Server module location.
# rl-serl/rl_robot_infra/robot_servers/openarm_server.py
# OpenArm control, IK, camera, configs, and description now live under rl_robot_infra.
# parents: [0]=robot_servers [1]=rl_robot_infra [2]=rl-serl [3]=zy
INFRA_ROOT = Path(__file__).resolve().parents[1]

# --- Hardware controller ---

# Real hardware only.
print(">>> MODE: REAL HARDWARE <<<")
from openarm_control.controller import OpenArmController as HardwareController

# --- IK solver imports ---
# Converts Gym Cartesian commands into controller joint commands.
try:
    from openarm_ik.robot_ik_solver import BaseIKSolver
    from openarm_ik.viser_base import ViserBase
    IK_AVAILABLE = True
except ImportError as e:
    print(f"[Server Warning] OpenArm IK imports failed: {e}. Cartesian control will fail.")
    BaseIKSolver = None
    ViserBase = None
    IK_AVAILABLE = False

app = Flask(__name__)

class OpenArmServer:
    def __init__(self):
        print("Initializing Real OpenArm Hardware Controller...")
        # Hardware controller.
        self.controller = HardwareController(enable_left=True, enable_right=True)
        
        # IK solver and optional Viser visualization.
        self.ik_solver = None
        self.viser = None
        if IK_AVAILABLE:
            self._init_ik_and_viser()

        # Locks protect hardware, camera, and servo state across Flask threads.
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
            "target_update_count": 0,
        }
        self.last_servo_log_ts = 0.0
        self.servo_thread = threading.Thread(target=self._servo_loop, daemon=True)
        self.servo_thread.start()
        
        # Initialize Cameras
        self.cameras = {}
        self._init_cameras()
        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.camera_thread.start()

    def _init_ik_and_viser(self):
        """Initialize the IK solver and optional Viser visualization."""
        try:
            cfg_path = INFRA_ROOT / "openarm_configs"
            with open(cfg_path / "robot.yaml") as f: r_cfg = yaml.safe_load(f)
            with open(cfg_path / "solver.yaml") as f: s_cfg = yaml.safe_load(f)
            with open(cfg_path / "viser.yaml") as f: v_cfg = yaml.safe_load(f)
            
            # Resolve URDF package paths to this repository.
            r_cfg["description"]["package_path"] = str(INFRA_ROOT)
            
            self.ik_solver = BaseIKSolver(s_cfg, r_cfg, visualize_collision=False)
            
            # Warm up JAX IK compilation.
            print("Warming up JAX IK Solver...")
            dummy_q = np.zeros(14)
            # IK target format: [w, x, y, z, px, py, pz].
            dummy_target = np.array([
                [1,0,0,0, 0.3, 0.2, 0.3], 
                [1,0,0,0, 0.3, -0.2, 0.3]
            ])
            self.ik_solver.solve_ik(dummy_target, dummy_q)
            print("IK Solver Ready.")

            # Optional Viser visualization.
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
            from openarm_ik import analytic_IK as ik_module
            from openarm_ik import collision_check as collision_module
            from openarm_ik.workspace_constraint import create_openarm_constraint

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
        """Initialize configured cameras."""
        cfg_path = INFRA_ROOT / "openarm_configs" / "cameras.yaml"
        with open(cfg_path) as f:
            self.cam_configs = yaml.safe_load(f) or {}
        
        for name, cfg in self.cam_configs.items():
            try:
                print(f"Initializing Camera {name}...")
                cam = build_camera(name, cfg, virtual=False)
                self.cameras[name] = cam
                self.latest_frames[name] = None
                print(f"Initialized {name} camera.")
            except Exception as e:
                print(f"[Server Warning] Camera {name} init failed: {e}")

    def _camera_loop(self):
        """Background thread to read camera frames"""
        print("Starting Camera Loop...")
        while self.running:
            for name, cam in self.cameras.items():
                img = camera_frame(cam.get_data())
                if img is not None:
                    with self.camera_lock:
                        self.latest_frames[name] = img
            time.sleep(0.01)

    def _get_current_joint_and_gripper(self):
        with self.lock:
            state = self.controller.read_state()
            return state.q.copy(), state.gripper_q.copy()

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
            # Teleop gripper commands should land directly on the active arm(s) so the
            # hardware can overcome static friction. The old special-case only covered
            # right-arm single-arm teleop, which made bimanual teleop appear unresponsive.
            if i in active_arms:
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
        target_update_delta=0,
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
            self.servo_debug["target_update_count"] += int(target_update_delta)
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
            f"fails={dbg['solve_fail_count']} updates={dbg['target_update_count']} "
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
                ok, command_info = self.controller.command_joint_target(
                    q_target,
                    gripper_target=stepped_gripper,
                    active_arms=active_arms,
                    dt=max(1.0 / self.servo_hz, 0.001),
                    source=f"servo:{backend}",
                )
                if not ok:
                    self._update_servo_debug(
                        status="safety_blocked",
                        error=str(command_info.get("safety", {})),
                        backend=backend,
                        active_arms=active_arms,
                        target_pose=stepped_pose,
                        current_pose=current_pose,
                        q_target=q_target,
                        current_gripper=g_curr,
                        target_gripper=target_gripper,
                        stepped_gripper=stepped_gripper,
                    )
                    self._maybe_log_servo_debug()
                    continue
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
        self._update_servo_debug(
            status="servo_started",
            backend=str(backend).lower(),
            active_arms=active_arms,
            target_pose=target_pose,
            target_gripper=target_gripper,
            target_update_delta=1,
        )
        return True

    def update_servo_target(self, target_pose_flat, gripper_pos=None):
        target_pose = np.array(target_pose_flat, dtype=np.float64).reshape(2, 7)
        target_gripper = None if gripper_pos is None else np.array(gripper_pos, dtype=np.float64)
        with self.servo_lock:
            if not self.servo_enabled:
                return False
            self.servo_target_pose = target_pose
            if target_gripper is not None:
                self.servo_target_gripper = target_gripper
            self.servo_last_update_ts = time.time()
        self._update_servo_debug(
            status="target_updated",
            target_pose=target_pose,
            target_gripper=target_gripper,
            target_update_delta=1,
        )
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

    def get_state(self, include_images=True):
        """Return robot state, including encoded images."""
        rich_state = None
        with self.lock:
            # Read joint and gripper state.
            rich_state = self.controller.read_state()
            l_pos = rich_state.left.q
            r_pos = rich_state.right.q
            l_grip = rich_state.left.gripper_q
            r_grip = rich_state.right.gripper_q
            
        # Normalize missing values.
        if l_pos is None: l_pos = np.zeros(7)
        if r_pos is None: r_pos = np.zeros(7)
        if not l_grip: l_grip = [0.0]
        if not r_grip: r_grip = [0.0]
        
        # Concatenate joint positions into a 14-D vector.
        q = np.concatenate([l_pos, r_pos])
        
        # Pack gripper positions into a 2-D vector.
        gripper = np.array([l_grip[0], r_grip[0]])

        # Compute end-effector pose with forward kinematics.
        # If IK is unavailable, keep pose as zeros and still return state data.
        pose = np.zeros(14)
        if self.ik_solver:
            # IK solver returns [w, qx, qy, qz, px, py, pz].
            # OpenArmEnv expects [px, py, pz, qx, qy, qz, qw].
            ik_poses = self.ik_solver.get_current_ee_pose(q)
            for i in range(2):
                p = ik_poses[i]
                # Convert [w, x, y, z, px, py, pz] -> [px, py, pz, x, y, z, w].
                pose[i*7 : i*7+3] = p[4:7] # Pos
                pose[i*7+3 : i*7+6] = p[1:4] # Rot (quat xyz)
                pose[i*7+6] = p[0]         # Rot (quat w)

        if rich_state is not None:
            state_dict = rich_state.to_dict()
            dq = state_dict["dq"]
            joint_torque = state_dict["joint_torque"]
            joint_accel_est = state_dict["joint_accel_est"]
            motor_temperature = state_dict["motor_temperature"]
            motor_enabled = state_dict["motor_enabled"]
        else:
            dq = [0.0] * 14
            joint_torque = [0.0] * 14
            joint_accel_est = [0.0] * 14
            motor_temperature = {"t_mos": [], "t_rotor": []}
            motor_enabled = []
        safety_status = (
            self.controller.safety.last_status.to_dict()
            if hasattr(self.controller, "safety")
            else {"mode": "UNKNOWN", "ok_to_send": True, "speed_scale": 1.0, "reasons": []}
        )

        response = {
            "pose": pose.tolist(),
            "q": q.tolist(),
            "gripper_pos": gripper.tolist(),
            "vel": [0.0] * 12,     # Cartesian velocity placeholder.
            "dq": dq,
            "force": [0.0] * 6,
            "torque": [0.0] * 6,
            "joint_torque": joint_torque,
            "joint_accel_est": joint_accel_est,
            "motor_temperature": motor_temperature,
            "motor_enabled": motor_enabled,
            "safety_status": safety_status,
            "servo_debug": self._jsonify_servo_debug(),
        }
        
        # Update Viser visualization.
        if self.viser:
            self.viser.update_joints(q)

        # Encode latest camera frames.
        encoded_images = {}
        if include_images:
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

    def _jsonify_servo_debug(self):
        with self.servo_lock:
            dbg = dict(self.servo_debug)
        result = {}
        for key, value in dbg.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif isinstance(value, tuple):
                result[key] = list(value)
            else:
                result[key] = value
        if hasattr(self.controller, "diagnostics"):
            diagnostics = self.controller.diagnostics()
            result["controller"] = diagnostics.get("controller", {})
            result["safety"] = diagnostics.get("safety", {})
        return result

# --- Server instance ---
server = OpenArmServer()

# --- Flask routes ---

@app.route("/getstate", methods=["POST"])
def route_get_state():
    try:
        return jsonify(server.get_state(include_images=True))
    except Exception as e:
        print(f"[API Error] getstate: {e}")
        return str(e), 500


@app.route("/state", methods=["POST"])
def route_state():
    try:
        payload = request.get_json(silent=True) or {}
        return jsonify(server.get_state(include_images=bool(payload.get("include_images", False))))
    except Exception as e:
        print(f"[API Error] state: {e}")
        return str(e), 500


@app.route("/diagnostics", methods=["POST"])
def route_diagnostics():
    try:
        diagnostics = server.controller.diagnostics() if hasattr(server.controller, "diagnostics") else {}
        diagnostics["servo_debug"] = server._jsonify_servo_debug()
        return jsonify(diagnostics)
    except Exception as e:
        print(f"[API Error] diagnostics: {e}")
        return str(e), 500


@app.route("/motor_params/query", methods=["POST"])
def route_motor_params_query():
    try:
        if hasattr(server.controller, "query_motor_params"):
            return jsonify(server.controller.query_motor_params())
        return jsonify({})
    except Exception as e:
        print(f"[API Error] motor_params/query: {e}")
        return str(e), 500

@app.route("/control/start", methods=["POST"])
def route_control_start():
    try:
        payload = request.json or {}
        arr = payload.get("arr")
        gripper = payload.get("gripper")
        servo_hz = payload.get("servo_hz", 100.0)
        trans_step = payload.get("trans_step", 0.004)
        rot_step = payload.get("rot_step", 0.012)
        gripper_step = payload.get("gripper_step", 0.05)
        arm = payload.get("arm", "both")
        backend = payload.get("backend", "analytic")
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
            return jsonify({"ok": True}), 200
        return jsonify({"ok": False, "error": "control start failed"}), 500
    except Exception as e:
        print(f"[API Error] control/start: {e}")
        return str(e), 500


@app.route("/control/target", methods=["POST"])
def route_control_target():
    try:
        payload = request.json or {}
        arr = payload.get("arr")
        joints = payload.get("joints")
        gripper = payload.get("gripper")
        if joints is not None:
            q_target = np.array(joints, dtype=np.float64).reshape(14)
            active_arms = (0, 1)
            with server.lock:
                ok, info = server.controller.command_joint_target(
                    q_target,
                    gripper_target=gripper,
                    active_arms=active_arms,
                    dt=max(1.0 / server.servo_hz, 0.001),
                    source="control:joint_target",
                )
            return jsonify({"ok": bool(ok), "info": info}), 200 if ok else 409
        if arr is None:
            return "Missing array", 400
        if not server.servo_enabled:
            if not server.start_servo(arr, gripper_pos=gripper):
                return "Servo Start Fail", 500
        if server.update_servo_target(arr, gripper_pos=gripper):
            return jsonify({"ok": True}), 200
        return "Servo Not Enabled", 409
    except Exception as e:
        print(f"[API Error] control/target: {e}")
        return str(e), 500


@app.route("/control/stop", methods=["POST"])
def route_control_stop():
    try:
        server.stop_servo()
        if hasattr(server.controller, "hold_position"):
            with server.lock:
                server.controller.hold_position()
        return jsonify({"ok": True}), 200
    except Exception as e:
        print(f"[API Error] control/stop: {e}")
        return str(e), 500


@app.route("/control/home", methods=["POST"])
def route_control_home():
    try:
        home_pos_l = [-0.166811, -0.497863, 0.635447, 1.499999, -0.627859, 0.507960, -0.168161]
        home_pos_r = [0.166811, 0.497863, -0.635447, 1.499999, 0.627859, -0.507960, 0.168161]
        home_pos = np.concatenate([home_pos_l, home_pos_r])
        payload = request.json or {}
        duration = float(payload.get("duration", 3.0))
        gripper = payload.get("gripper")
        servo_hz = float(payload.get("servo_hz", server.servo_hz))
        dt = max(1.0 / servo_hz, 0.001)
        deadline = time.time() + max(duration, dt)
        server.stop_servo()
        last_info = {}
        ok = True
        while time.time() < deadline:
            with server.lock:
                ok, last_info = server.controller.command_joint_target(
                    home_pos,
                    gripper_target=gripper,
                    active_arms=(0, 1),
                    dt=dt,
                    source="control:home",
                )
            if not ok:
                break
            time.sleep(dt)
        return jsonify({"ok": bool(ok), "info": last_info}), 200 if ok else 409
    except Exception as e:
        print(f"[API Error] control/home: {e}")
        return str(e), 500

if __name__ == "__main__":
    print("Starting OpenArm Server on port 5000...")
    app.run(host="0.0.0.0", port=5000, threaded=True)
