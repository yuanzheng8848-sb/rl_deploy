import importlib.util
import json
import os
from pathlib import Path
import pickle as pkl
import sys
import time
import traceback
from datetime import datetime

import cv2
import jax
import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml

# Resolve project paths first so local modules still import after moving this file.
FILE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = FILE_DIR.parents[1]
PROJECT_ROOT = WORKSPACE_DIR.parent
PYROKI_DIR = WORKSPACE_DIR / "pyroki"
SERL_LAUNCHER_PARENT = PROJECT_ROOT / "serl" / "serl_launcher"
SERL_ROBOT_INFRA_PARENT = PROJECT_ROOT / "serl" / "serl_robot_infra"

for path in (PYROKI_DIR, SERL_LAUNCHER_PARENT, SERL_ROBOT_INFRA_PARENT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from data_recorder import DataRecorder
from ik_performance_monitor import create_monitor, create_gui_components
from realsense_camera import RealsenseCamera
from robot_ik_solver import BaseIKSolver
from serl_launcher.networks.reward_classifier import load_classifier_func
from viser_base import ViserBase
from vr import VRUpperBodyTeleop
from workspace_constraint import create_openarm_constraint

cur_dir = str(WORKSPACE_DIR)
IK_dir = os.path.join(cur_dir, "IK")
file_path = os.path.join(IK_dir, "analytic_IK.py")
# Import using importlib
spec_ik = importlib.util.spec_from_file_location("analytic_IK", file_path)
ik_module = importlib.util.module_from_spec(spec_ik)
spec_ik.loader.exec_module(ik_module)

file_path = os.path.join(IK_dir, "collision_check.py")
spec_collision = importlib.util.spec_from_file_location("collision_check", file_path)
collision_module = importlib.util.module_from_spec(spec_collision)
spec_collision.loader.exec_module(collision_module)

file_path = os.path.join(cur_dir, "openarm", "openarm_controller.py")
spec_openarm = importlib.util.spec_from_file_location("openarm_controller", file_path)
openarm_module = importlib.util.module_from_spec(spec_openarm)
spec_openarm.loader.exec_module(openarm_module)

USE_VR = True
USE_REAL = True
USE_CAMERA = True
SAVE_DEPTH = False # Whether to save depth map
# Remember to set DEBUG = False when recording, otherwise camera window will pop up and affect performance
DEBUG = False

# IK Performance Monitor Config
DEBUG_IK_PERF = False  # Master switch
DEBUG_IK_PERF_GUI = False  # Show in GUI
DEBUG_IK_PERF_CONSOLE = False  # Print to console
DEBUG_IK_PERF_CSV = False  # Save CSV log

'''
Add VR Operation

All pose order is qw qx qy qz x y z
'''


# Camera Config: (serial_number, width, height, fps)
CAMERA_CONFIGS = [
    ("150622074105", 640, 480, 30),  # left
    ("236422072385", 640, 480, 30),  # right
    ("/dev/video10", 640, 480, 30),  # head (USB Camera Device Path)
]

# Map serials to SERL keys
CAM_SERIAL_TO_KEY = {
    "150622074105": "image_left",
    "236422072385": "image_right",
    "/dev/video10": "image_primary", # USB Camera
}

class USBCamera:
    def __init__(self, device_id, width=640, height=480, fps=30):
        self.cap = cv2.VideoCapture(device_id)
        if not self.cap.isOpened():
            raise Exception(f"Could not open USB camera {device_id}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        self.width = width
        self.height = height
        self.device_id = device_id
        
        # Attributes expected by DataRecorder
        self.color_image = None
        self.depth_image = None
        
    def get_data(self, viz=False):
        ret, frame = self.cap.read()
        data = [None, None]
        if ret:
            # OpenCV returns BGR, convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            data[0] = rgb_frame
            
            # Update attributes
            self.color_image = rgb_frame
            self.depth_image = None # USB camera has no depth
            
            if viz:
                cv2.imshow(f'USB Camera-{self.device_id}', frame) # Show BGR
                cv2.waitKey(1)
        else:
            self.color_image = None
            self.depth_image = None
            
        return data
        
    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

def average_time(func):
    """
    Decorator: Calculate average function call time and record to file
    
    Args:
        func: Decorated function
    """
    import functools
    import time
    import atexit
    
    # Use dictionary to store stats
    stats = {'count': 0, 'total_time': 0.0}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        
        stats['count'] += 1
        stats['total_time'] += elapsed
        save_stats()
        
        return result
    
    def save_stats():
        """Save stats to file on exit"""
        if stats['count'] > 0:
            avg_time = stats['total_time'] / stats['count']
            with open("runtime.log", 'a', encoding='utf-8') as f:
                f.write(f"Function: {func.__name__}\n")
                f.write(f"  Calls: {stats['count']}\n")
                f.write(f"  Total Time: {stats['total_time']:.6f} s\n")
                f.write(f"  Avg Time: {avg_time:.6f} s\n")
                f.write(f"  Min Est: {avg_time * 1000:.3f} ms\n")
                f.write("-" * 50 + "\n")

    return wrapper


def _pose_headers():
    pose_fields = [
        ("qw", "Quaternion w"),
        ("qx", "Quaternion x"),
        ("qy", "Quaternion y"),
        ("qz", "Quaternion z"),
        ("x", "Position x (m)"),
        ("y", "Position y (m)"),
        ("z", "Position z (m)"),
    ]
    headers = []
    for side in ("left", "right"):
        for key, desc in pose_fields:
            headers.append(
                {
                    "name": f"target_{side}_{key}",
                    "description": f"{side.title()} arm {desc}",
                }
            )
    return headers


def _flatten_pose_pair(pose_pair):
    def _pose_list(one_pose):
        if one_pose is None:
            return [None] * 7
        pose = list(one_pose)
        if len(pose) >= 7:
            return pose[:7]
        return pose + [None] * (7 - len(pose))

    left = _pose_list(pose_pair[0]) if pose_pair is not None and len(pose_pair) > 0 else [None] * 7
    right = _pose_list(pose_pair[1]) if pose_pair is not None and len(pose_pair) > 1 else [None] * 7
    return left + right


def _joint_headers(joint_names):
    if not joint_names:
        return [{"name": "joint_value", "description": "Joint position (rad)"}]
    return [
        {
            "name": f"joint_{name}",
            "description": f"Joint {name} position (rad)",
        }
        for name in joint_names
    ]


def _flatten_joint_state(joint_state):
    if isinstance(joint_state, (list, tuple, np.ndarray)):
        return list(joint_state)
    return None


def create_runtime_logger(pyroki_root, joint_names):
    """
    Returns a lightweight logger function, only records IK failure frames.
    JSON Structure:
    {
        "header": [...fields...],
        "data": [
            [timestamp, target values..., joint values...],
            ...
        ]
    }
    """
    log_dir = os.path.join(pyroki_root, "log")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"teleop_{timestamp}.json")
    log_file = open(log_path, "w", encoding="utf-8")
    header = [{"name": "timestamp", "description": "UTC ISO timestamp"}]
    header.extend(_pose_headers())
    header.extend(_joint_headers(joint_names))

    log_file.write("{\n")
    log_file.write('"header": ')
    json.dump(header, log_file, ensure_ascii=False)
    log_file.write(',\n"data": [\n')
    first_entry = True

    def _log_state(timestamp_value, target_pose, joint_state):
        nonlocal first_entry
        if log_file.closed:
            return

        row = [timestamp_value]
        row.extend(_flatten_pose_pair(target_pose))
        joints = _flatten_joint_state(joint_state)
        if joints is not None:
            row.extend(joints)

        if not first_entry:
            log_file.write(",\n")
        else:
            first_entry = False

        json.dump(row, log_file, ensure_ascii=False)
        log_file.flush()

    def _close():
        nonlocal first_entry
        if not log_file.closed:
            if not first_entry:
                log_file.write("\n")
            log_file.write("]\n}\n")
            log_file.close()

    _log_state.close = _close
    _log_state.path = log_path
    return _log_state

def get_relative_action(current_ee_pose, target_ee_pose):
    """
    Compute relative action (delta) between current and target pose.
    Poses are [x, y, z, qx, qy, qz, qw].
    Returns [dx, dy, dz, droll, dpitch, dyaw] (6D) in BODY frame.
    """
    # Position delta
    delta_pos = target_ee_pose[:3] - current_ee_pose[:3]
    
    # Orientation
    quat_curr = current_ee_pose[3:]
    rot_curr = R.from_quat(quat_curr)
    quat_target = target_ee_pose[3:]
    rot_target = R.from_quat(quat_target)
    
    # Relative rotation: R_delta = R_curr^T * R_target
    rot_delta = rot_curr.inv() * rot_target
    delta_euler = rot_delta.as_euler('xyz')
    
    # Relative position in body frame: P_delta = R_curr^T * (P_target - P_curr)
    delta_pos_body = rot_curr.inv().apply(delta_pos)
    
    return np.concatenate([delta_pos_body, delta_euler])


def main():
    # Original Record Dir (for DataRecorder if used)
    record_dir = os.path.join(cur_dir, "rl_deploy", "demo", "record_data")
    os.makedirs(record_dir, exist_ok=True)
    
    # SERL Demo Dir
    serl_demo_dir = os.path.join(cur_dir, "rl_deploy", "demo", "demos_origin")
    os.makedirs(serl_demo_dir, exist_ok=True)

    recorder = None
    state_logger = None
    cfgs_path = Path(os.path.join(cur_dir,"pyroki","config"))
    # Config IK solver
    cfgs_robot = yaml.safe_load( (cfgs_path / "robot.yaml").read_text())
    cfgs_solver = yaml.safe_load( (cfgs_path / "solver.yaml").read_text())
    solver = BaseIKSolver(cfgs_solver, cfgs_robot, True)

    # Config viser
    cfg_viser = yaml.safe_load( (cfgs_path / "viser.yaml").read_text())

    if USE_VR:
        cfg_viser["nb_vis_frames"] = 2
    else:
        cfg_viser["nb_vis_frames"] = 6
    
    viser = ViserBase(  
            cfg_viser,
            solver.urdf,
            solver.get_actuated_joint_order(),
            solver.get_target_link_indices(),
            solver.forward_kinematics,
            use_sim=True,
            use_teleop=True,
            )
    joint_names = solver.get_actuated_joint_order()
    state_logger = create_runtime_logger(os.path.join(cur_dir, "pyroki"), joint_names)

    target_pos_dir = os.path.join(cur_dir, "rl_deploy", "classifier", "target_position")
    os.makedirs(target_pos_dir, exist_ok=True)

    manip_weight = viser._server.gui.add_slider("Manipulability Weight", 0.0, 10.0, 0.001, 0.0)
    limit_weight = viser._server.gui.add_slider("Limit Avoidance Weight", 0.0, 100.0, 0.01, 0.0)
    
    # IK Performance Monitor GUI
    ik_perf_gui_displays = None
    if DEBUG_IK_PERF and DEBUG_IK_PERF_GUI:
        ik_perf_gui_displays = create_gui_components(viser._server)
    
    # Recording State
    recording_state = {"active": False, "toggle_requested": False, "button": None, "status_text": None}
    episode_transitions = []
    
    # Helper to update record button
    def update_record_button(is_recording):
        """Update record button color and text"""
        if recording_state["button"] is not None:
            recording_state["button"].remove()
        
        if is_recording:
            button = viser._server.gui.add_button("🔴 Stop Recording", color="red")
            recording_state["status_text"].value = "🔴 RECORDING"
        else:
            button = viser._server.gui.add_button("⚪ Start Recording", color="white")
            recording_state["status_text"].value = "Not Recording"
        
        recording_state["button"] = button
        
        @button.on_click
        def _(_):
            recording_state["toggle_requested"] = True
        
        return button
    
    # Init button and status text
    recording_state["status_text"] = viser._server.gui.add_text("Status", initial_value="Not Recording", disabled=True)
    update_record_button(False)
    
    # Add instructions
    viser._server.gui.add_markdown(
        "**Recording Controls:**\n"
        "- Click button above or press **SPACE** key\n"
        "- Button color: 🔴 Red = Recording | ⚪ White = Not Recording"
    )
    
    # VR target visualization switch and frames
    vr_target_vis_enabled = viser._server.gui.add_checkbox("Show VR Target Frames", initial_value=False)
    vr_target_vis_prev_state = False
    vr_target_frames = None
    if USE_VR:
        vr_target_frames = [
            viser._server.scene.add_frame(
                "/vr_target_left",
                position=np.array([0.0, 0.0, 0.0]),
                wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                axes_length=0.25,
                axes_radius=0.015,
            ),
            viser._server.scene.add_frame(
                "/vr_target_right",
                position=np.array([0.0, 0.0, 0.0]),
                wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                axes_length=0.25,
                axes_radius=0.015,
            ),
        ]
        for frame in vr_target_frames:
            frame.visible = False
        print("[INFO] VR target frames created (left and right)")

    
    # main loop
    current_joints = viser.get_init_joints_for_sim()
    start_time = None

    # openarm configuration
    origin_position = np.array([0.0, 0.0, 0.0])
    l1 = 0.22
    l2 = 0.216
    IK_triangle = ik_module.Triangle(l1, l2, origin_position)
    current_ee_pose = solver.get_current_ee_pose(current_joints)
    IK_triangle.set_init_ee_pose(current_ee_pose[0], current_ee_pose[1])

    # Workspace Constraint
    workspace_constraint = create_openarm_constraint(l1=l1, l2=l2, safety_margin=0.016)

    T = solver.forward_kinematics(np.zeros(14)) # Show zero state

    # shoulder in world frame
    left_shoulder_position = T[3][4:]
    left_shoulder_position[1] += T[4][4:][1] - T[3][4:][1]
    left_shoulder_orientation = R.from_matrix(np.array([[0,-1,0],[0,0,1],[-1,0,0]]))
    left_shoulder_pose = np.concatenate((left_shoulder_orientation.as_quat(scalar_first=True), left_shoulder_position), 0)

    right_shoulder_position = T[3+8][4:]
    right_shoulder_position[1] += T[4+8][4:][1] - T[3+8][4:][1]
    right_shoulder_orientation = R.from_matrix(np.array([[0,-1,0],[0,0,1],[-1,0,0]]))
    right_shoulder_pose = np.concatenate((right_shoulder_orientation.as_quat(scalar_first=True), right_shoulder_position), 0)

    # collision checker
    collision_checker = collision_module.OpenArmCollisionChecker(left_shoulder_position, right_shoulder_position, viser._server)
    
    # joint limit checker
    print("active joint: ", solver._robot.joints.actuated_names)
    joints_upper_limit = np.array(solver._robot.joints.upper_limits) + 0.0001
    joints_lower_limit = np.array(solver._robot.joints.lower_limits) - 0.0001

    print("upper bound: ", joints_upper_limit)
    print("lower bound: ", joints_lower_limit)

    if USE_REAL:
        # interface with real robot
        controller = openarm_module.OpenArmController(enable_left=True, enable_right=True)


    '''
    Config VR Device
    '''
    if USE_VR:
        cfg_vr = yaml.safe_load((cfgs_path / "vr.yaml").read_text())
        ip_vr = "10.255.27.103"
        
        vr = VRUpperBodyTeleop(
            cfg_vr,
            ip_vr,
            IK_triangle.get_current_ee_pose,
            os.path.join(cur_dir,"pyroki","config"),
            )

        try:
            target_pose, gripper_width = vr.wait_for_initial_states()
            print("vr inital cmd: ", target_pose, gripper_width)
        except Exception as e:
            print("can not receive VR signal")
            vr.stop()
            viser.stop()
            exit(1)

    cameras = []
    cam_dict = {}
    if USE_CAMERA:
        for serial, width, height, fps in CAMERA_CONFIGS:
            try:
                if isinstance(serial, int) or (isinstance(serial, str) and serial.startswith("/dev/video")):
                    # USB Camera
                    cam = USBCamera(device_id=serial, width=width, height=height, fps=fps)
                else:
                    # RealSense Camera
                    cam = RealsenseCamera(
                        device_id=serial,
                        enable_depth=SAVE_DEPTH,
                        width=width,
                        height=height,
                        fps=fps
                    )
                cameras.append(cam)
                key = CAM_SERIAL_TO_KEY.get(serial, f"image_{serial}")
                cam_dict[key] = cam
                print(f"Camera {serial} ({key}): {width}x{height}@{fps}fps, depth={SAVE_DEPTH}")
            except Exception as e:
                print(f"Cannot connect to camera {serial}: {e}")
        print(f"Found {len(cameras)} cameras for recording.")

    # Load Reward Classifier
    print("Loading Reward Classifier...")
    try:
        # Create dummy sample for initialization
        dummy_sample = {"image_0": np.zeros((256, 256, 3), dtype=np.uint8)}
        rng = jax.random.PRNGKey(0)
        ckpt_path = os.path.join(
            cur_dir,
            "rl_deploy",
            "classifier",
            "classifier_ckpt",
            "checkpoint_100",
        )
        classifier_func = load_classifier_func(
            key=rng,
            sample=dummy_sample,
            image_keys=["image_0"],
            checkpoint_path=os.path.abspath(ckpt_path),
        )
        print("Classifier Loaded.")
    except Exception as e:
        print(f"Failed to load classifier: {e}")
        classifier_func = None

    init_time = time.time()
    # main loop

    # IK Performance Monitor
    ik_monitor = None
    if DEBUG_IK_PERF:
        log_dir = os.path.join(cur_dir, "pyroki", "log") if DEBUG_IK_PERF_CSV else None
        ik_monitor = create_monitor(
            log_dir=log_dir,
            print_interval=10.0,
            enable_logging=DEBUG_IK_PERF_CSV,
            enable_console=DEBUG_IK_PERF_CONSOLE
        )
        if ik_perf_gui_displays:
            ik_monitor.set_gui_displays(*ik_perf_gui_displays)
    
    is_recording = False
    
    # Keyboard Listener
    import threading
    from pynput import keyboard
    
    def on_press(key):
        try:
            if key == keyboard.Key.space:
                recording_state["toggle_requested"] = True
        except:
            pass
    
    keyboard_listener = keyboard.Listener(on_press=on_press)
    keyboard_listener.start()
    print("[INFO] Keyboard listener started. Press SPACE to toggle recording.")

    try:
        while True:

            start_time = time.time()

            # Handle Recording Toggle Request
            if recording_state["toggle_requested"]:
                recording_state["toggle_requested"] = False
                
                if not is_recording:
                    # Start Recording
                    try:
                        # 1. Setup Raw Recorder
                        recorder = DataRecorder(save_dir=os.path.join(record_dir, f"session_{time.strftime('%Y%m%d_%H%M%S')}"), save_depth=SAVE_DEPTH)
                        
                        # 2. Setup RL Recorder
                        episode_transitions = []
                        
                        is_recording = True
                        recording_state["active"] = True
                        update_record_button(True)
                        print("🔴 Recording started.")
                    except Exception as e:
                        print("Failed to start recording:", e)
                        recorder = None
                        is_recording = False
                        recording_state["active"] = False
                else:
                    # Stop Recording
                    try:
                        # 1. Save Raw Data
                        if recorder is not None:
                            recorder.save()
                            print("⚪ Raw data saved.")
                            
                        # 2. Save RL Data
                        if len(episode_transitions) > 0:
                            timestamp = time.strftime('%Y%m%d_%H%M%S')
                            pkl_filename = f"wd_{timestamp}.pkl" # Follows pattern wd_YYYYMMDD_HHMMSS.pkl
                            pkl_path = os.path.join(serl_demo_dir, pkl_filename)
                            
                            # Handle last transition's next_observation (terminal)
                            # Usually we mark done=True or just leave it. 
                            # For now, we assume the episode ends here.
                            episode_transitions[-1]["dones"] = True
                            
                            with open(pkl_path, 'wb') as f:
                                pkl.dump(episode_transitions, f)
                            print(f"⚪ RL transitions saved to {pkl_path} ({len(episode_transitions)} steps)")
                            
                    except Exception as e:
                        print("Failed to save recording:", e)
                        traceback.print_exc()
                    finally:
                        recorder = None
                        episode_transitions = []
                        is_recording = False
                        recording_state["active"] = False
                        update_record_button(False)
                        print("⚪ Recording stopped.")
                
                try:
                    pass # Placeholder for removed position saving logic
                except Exception as e:
                    pass

            # Capture Images
            current_images = {}
            if USE_CAMERA:
                for key, cam in cam_dict.items():
                    # Get data (RGB)
                    data = cam.get_data(viz=DEBUG)
                    if data[0] is not None:
                        current_images[key] = data[0]

            # Capture Robot State (Before Action)
            obs_joints = current_joints
            obs_ee_poses_raw = solver.get_current_ee_pose(obs_joints) # [left, right]
            
            # Convert to [x,y,z,qx,qy,qz,qw] for RL/SERL
            def to_xyzquat(p): return np.concatenate([p[4:], p[1:4], [p[0]]])
            obs_left_ee = to_xyzquat(obs_ee_poses_raw[0])
            obs_right_ee = to_xyzquat(obs_ee_poses_raw[1])
            
            if USE_REAL:
                l_pos, l_grip = controller.get_left_position()
                r_pos, r_grip = controller.get_right_position()
                obs_joints = np.concatenate([l_pos, r_pos])
                # Re-compute EE with real joints
                obs_ee_poses_raw = solver.get_current_ee_pose(obs_joints)
                obs_left_ee = to_xyzquat(obs_ee_poses_raw[0])
                obs_right_ee = to_xyzquat(obs_ee_poses_raw[1])
            else:
                l_grip = 0.0
                r_grip = 0.0

            target_pose = None
            target_gripper = None

            if USE_VR:
                target_cmd = vr.get_vr_command() # [left, right, body] 3 x 7
                
                if target_cmd is None:
                    time.sleep(0.001)
                    continue

                target_pose_raw = target_cmd[0][:2] # left + right
                target_gripper = target_cmd[1] # left + right gripper
                
                # Workspace Constraint
                left_constrained, right_constrained = workspace_constraint.constrain_dual_arm(
                    target_pose_raw[0], 
                    target_pose_raw[1],
                    left_shoulder_position,
                    right_shoulder_position
                )
                target_pose = [left_constrained, right_constrained]
                
                # VR Target Visualization
                if vr_target_frames is not None:
                    current_state = vr_target_vis_enabled.value
                    if current_state != vr_target_vis_prev_state:
                        if current_state:
                            print("[INFO] VR Target Frames visualization ENABLED")
                        else:
                            print("[INFO] VR Target Frames visualization DISABLED")
                        vr_target_vis_prev_state = current_state
                    
                    if current_state:
                        for i, frame in enumerate(vr_target_frames):
                            if i < len(target_pose):
                                pose = target_pose[i]
                                frame.position = pose[4:7]
                                frame.wxyz = pose[0:4]
                                frame.visible = True
                    else:
                        for frame in vr_target_frames:
                            frame.visible = False
            else:
                target_pose = viser.get_target_pose()
                target_gripper = np.array([1, 1])

            if target_pose is None:
                raise "target pose is None"

            # IK Solve
            if ik_monitor:
                ik_start_time = time.perf_counter()
            
            solved, left_arm_cmd, right_arm_cmd = IK_triangle.solve(left_shoulder_position, left_shoulder_orientation,
                                right_shoulder_position, right_shoulder_orientation,
                                target_pose,
                                collision_checker, joints_lower_limit, joints_upper_limit)
            
            if ik_monitor:
                ik_elapsed = time.perf_counter() - ik_start_time
                ik_monitor.record_solve(ik_elapsed, solved)

            '''
            update real/simulated robot
            '''
            if solved:
                solution = np.concatenate((left_arm_cmd, right_arm_cmd), 0)
                current_joints = solution
                current_ee_pose_analytic = IK_triangle.get_current_ee_pose()
            else:
                pass

            T = solver.forward_kinematics(current_joints)
            
            if USE_VR:
                viser.update_vis_frame(np.array([target_pose[0], target_pose[1], T[0]]))
            else:
                viser.update_vis_frame(np.array([current_ee_pose[0], T[0], T[0], 
                                                current_ee_pose[1], T[0], T[0]]))

            elapsed_time = time.time() - start_time

            if USE_REAL:
                current_left_arm_position,  current_left_gripper_position = controller.get_left_position()
                current_right_arm_position,  current_right_gripper_position = controller.get_right_position()
                real_robot_joint_position = np.concatenate((current_left_arm_position, current_right_arm_position), 0)
                viser.update_results(real_robot_joint_position, elapsed_time)

                left_target_cmd = - target_gripper[0] * 0.95
                right_target_cmd = - target_gripper[1] * 0.95

                # Send Commands
                if time.time() - init_time < 5:
                    target_joints_left = (np.array(current_joints[:7])-np.array(current_left_arm_position)) * 0.1 + np.array(current_left_arm_position)
                    target_joints_right = (np.array(current_joints[7:14])-np.array(current_right_arm_position)) * 0.1 + np.array(current_right_arm_position)
                    target_joints_left = target_joints_left.tolist()
                    target_joints_right = target_joints_right.tolist()
                else:
                    target_joints_left = current_joints[:7]
                    target_joints_right = current_joints[7:14]

                controller.set_left_position(target_joints_left, left_target_cmd, current_left_arm_position, current_left_gripper_position)            
                controller.set_right_position(target_joints_right, right_target_cmd, current_right_arm_position, current_right_gripper_position)          
                
                # Record Data (Raw + RL)
                if is_recording:
                    # 1. Raw Data Recording (DataRecorder)
                    if recorder is not None:
                        # Prepare data for DataRecorder
                        obs_joints_raw = real_robot_joint_position
                        obs_ee_poses_raw_rec = solver.get_current_ee_pose(obs_joints_raw)
                        obs_gripper_joints_raw = [current_left_gripper_position, current_right_gripper_position]
                        obs_rec = [obs_joints_raw, obs_ee_poses_raw_rec, obs_gripper_joints_raw]

                        action_joints_rec = np.array([target_joints_left , target_joints_right]).flatten()
                        action_ee_poses_rec = solver.get_current_ee_pose(action_joints_rec)
                        # Main v4 uses 'gripper' variable which is [left_target_cmd, right_target_cmd]
                        action_gripper_joints_rec = [left_target_cmd, right_target_cmd]
                        act_rec = [action_joints_rec, action_ee_poses_rec, action_gripper_joints_rec]

                        if USE_CAMERA:
                            recorder.record(*obs_rec, *act_rec, cameras)
                        else:
                            recorder.record(*obs_rec, *act_rec)

                    # 2. RL Data Recording (SERL Format)
                    # Construct Action (Relative)
                    # We use Right Arm for RL usually (based on record_demo_vr_v4.py)
                    # Target Pose for Right Arm is target_pose[1] (qw, qx, qy, qz, x, y, z)
                    # Convert to [x,y,z,qx,qy,qz,qw]
                    target_right_ee = np.concatenate([target_pose[1][4:], target_pose[1][1:4], [target_pose[1][0]]])
                    
                    delta_action = get_relative_action(obs_right_ee, target_right_ee)
                    action_gripper = right_target_cmd # Use command
                    action = np.concatenate([delta_action, [action_gripper]])
                    
                    # Compute Reward
                    reward = 0.0
                    if classifier_func is not None and "image_right" in current_images:
                        # Resize to 256x256 for classifier
                        img_resized = cv2.resize(current_images["image_right"], (256, 256))
                        rew_input = {"image_0": img_resized}
                        logit = classifier_func(rew_input).item()
                        prob = 1 / (1 + np.exp(-logit))
                        reward = 1.0 if prob > 0.5 else 0.0
                    
                    # Construct Obs
                    # State: [x,y,z,qx,qy,qz,qw, gripper]
                    # r_grip might be an array, ensure it's flat
                    r_grip_val = np.array(r_grip).flatten()
                    obs_state = np.concatenate([obs_right_ee, r_grip_val])
                    obs = {"state": obs_state, **current_images}
                    
                    transition = {
                        "observations": obs,
                        "actions": action,
                        "rewards": reward,
                        "dones": False,
                        "masks": 1.0,
                        "next_observations": None
                    }
                    
                    if len(episode_transitions) > 0:
                        episode_transitions[-1]["next_observations"] = obs
                    
                    episode_transitions.append(transition)

            else:
                viser.update_results(current_joints, elapsed_time)

            if not solved:
                state_logger(
                    datetime.utcnow().isoformat(),
                    target_pose,
                    current_joints,
                )

            time.sleep(0.01)

    except Exception as e:
        print("Exception: ", e)
        traceback.print_exc()
    finally:
        keyboard_listener.stop()
        if state_logger is not None:
            state_logger.close()
        if ik_monitor is not None:
            ik_monitor.close()
        if 'workspace_constraint' in locals():
            workspace_constraint.print_stats()
        viser.stop()
        
        # Save if pending
        if recorder is not None:
            recorder.save()
        if len(episode_transitions) > 0:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            fname = f"vr_demo_crash_save_{len(episode_transitions)}.pkl"
            fpath = os.path.join(serl_demo_dir, fname)
            try:
                with open(fpath, "wb") as f:
                    pkl.dump(episode_transitions, f)
                print(f"Emergency save to {fpath}")
            except:
                pass

if __name__ == "__main__":
    main()
