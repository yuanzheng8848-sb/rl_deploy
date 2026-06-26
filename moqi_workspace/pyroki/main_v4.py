from viser_base import ViserBase
from robot_ik_solver import BaseIKSolver
from vr import VRUpperBodyTeleop

import numpy as np
import time
import yaml
from pathlib import Path
import traceback
from datetime import datetime
import json

from realsense_camera import RealsenseCamera


# 解析解IK用的
from scipy.spatial.transform import Rotation as R
import importlib.util
import sys
import os
from data_recorder import DataRecorder
from ik_performance_monitor import create_monitor, create_gui_components
from workspace_constraint import create_openarm_constraint


USE_VR = True
USE_REAL = True
USE_CAMERA = True
SAVE_DEPTH = False # 是否保存深度图
# 录制时，记得设置DEBUG = False，否则会弹出相机窗口影响性能
DEBUG = False

# IK性能监控配置
DEBUG_IK_PERF = False  # 是否启用IK性能监控（总开关）
DEBUG_IK_PERF_GUI = False  # 是否在GUI显示
DEBUG_IK_PERF_CONSOLE = False  # 是否在控制台打印
DEBUG_IK_PERF_CSV = False  # 是否保存CSV日志

'''
加入 VR 操作

所有的 pose 顺序为 qw qx qy qz x y z
'''


# 相机配置：(serial_number, width, height, fps)
CAMERA_CONFIGS = [
    ("150622074105", 640, 480, 30),  # left
    ("236422072385", 640, 480, 30),  # right
    ("248622302807", 1280, 720, 30),  # head
]

# 定义文件绝对路径
cur_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# path of moqi_workspace
IK_dir = os.path.join(cur_dir, "IK")
file_path = os.path.join(IK_dir, "analytic_IK.py")
# 使用 importlib 导入
spec_ik = importlib.util.spec_from_file_location("analytic_IK", file_path)
ik_module = importlib.util.module_from_spec(spec_ik)
spec_ik.loader.exec_module(ik_module)

file_path = os.path.join(IK_dir, "collision_check.py")
spec_collision = importlib.util.spec_from_file_location("analytic_IK", file_path)
collision_module = importlib.util.module_from_spec(spec_collision)
spec_collision.loader.exec_module(collision_module)

file_path = os.path.join(cur_dir, "openarm","openarm_controller.py")
spec_openarm = importlib.util.spec_from_file_location("openarm_controller", file_path)
openarm_module = importlib.util.module_from_spec(spec_openarm)
spec_openarm.loader.exec_module(openarm_module)



def average_time(func):
    """
    装饰器：计算函数平均调用时间并记录到文件
    
    Args:
        func: 被装饰的函数
    """
    import functools
    import time
    import atexit
    
    # 使用字典存储统计信息
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
        """程序退出时保存统计信息到文件"""
        if stats['count'] > 0:
            avg_time = stats['total_time'] / stats['count']
            with open("runtime.log", 'a', encoding='utf-8') as f:
                f.write(f"函数: {func.__name__}\n")
                f.write(f"  调用次数: {stats['count']}\n")
                f.write(f"  总时长: {stats['total_time']:.6f} 秒\n")
                f.write(f"  平均时长: {avg_time:.6f} 秒\n")
                f.write(f"  单次最小估计: {avg_time * 1000:.3f} 毫秒\n")
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
    返回一个轻量级记录函数，仅记录 IK 失败帧。
    JSON 结构:
    {
        "header": [...字段说明...],
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




import pdb

def main():
    record_dir = os.path.join(cur_dir, "record_data")
    os.makedirs(record_dir, exist_ok=True)
    recorder = None
    state_logger = None
    cfgs_path = Path(os.path.join(cur_dir,"pyroki","config"))
    # 配置 IK solver
    cfgs_robot = yaml.safe_load( (cfgs_path / "robot.yaml").read_text())
    cfgs_solver = yaml.safe_load( (cfgs_path / "solver.yaml").read_text())
    solver = BaseIKSolver(cfgs_solver, cfgs_robot, True)

    # 配置 viser
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

    manip_weight = viser._server.gui.add_slider("Manipulability Weight", 0.0, 10.0, 0.001, 0.0)
    limit_weight = viser._server.gui.add_slider("Limit Avoidance Weight", 0.0, 100.0, 0.01, 0.0)
    
    # IK 性能监控 GUI（如果启用）
    ik_perf_gui_displays = None
    if DEBUG_IK_PERF and DEBUG_IK_PERF_GUI:
        ik_perf_gui_displays = create_gui_components(viser._server)
    
    # 录制状态
    recording_state = {"active": False, "toggle_requested": False, "button": None, "status_text": None}
    
    # 创建/更新录制按钮的辅助函数
    def update_record_button(is_recording):
        """更新录制按钮的颜色和文本"""
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
    
    # 初始化按钮和状态文本
    recording_state["status_text"] = viser._server.gui.add_text("Status", initial_value="Not Recording", disabled=True)
    update_record_button(False)
    
    # 添加说明
    viser._server.gui.add_markdown(
        "**Recording Controls:**\n"
        "- Click button above or press **SPACE** key\n"
        "- Button color: 🔴 Red = Recording | ⚪ White = Not Recording"
    )
    
    # VR target 可视化开关和坐标系
    vr_target_vis_enabled = viser._server.gui.add_checkbox("Show VR Target Frames", initial_value=False)
    vr_target_vis_prev_state = False  # 跟踪之前的状态
    vr_target_frames = None
    if USE_VR:
        # 创建左右臂target的坐标系可视化
        # 左臂：红色标签，右臂：蓝色标签
        # 增大尺寸以便更容易看到
        vr_target_frames = [
            viser._server.scene.add_frame(
                "/vr_target_left",
                position=np.array([0.0, 0.0, 0.0]),
                wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                axes_length=0.25,  # 增大到0.25米
                axes_radius=0.015,  # 增大到0.015米
            ),
            viser._server.scene.add_frame(
                "/vr_target_right",
                position=np.array([0.0, 0.0, 0.0]),
                wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                axes_length=0.25,  # 增大到0.25米
                axes_radius=0.015,  # 增大到0.015米
            ),
        ]
        # 初始隐藏
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

    # 工作空间约束器
    workspace_constraint = create_openarm_constraint(l1=l1, l2=l2, safety_margin=0.016)

    T = solver.forward_kinematics(np.zeros(14)) # 显示零状态

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
    配置 VR 设备
    '''
    if USE_VR:
        cfg_vr = yaml.safe_load((cfgs_path / "vr.yaml").read_text())
        ip_vr = "10.255.20.244"
        
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

    if USE_CAMERA:
        cameras = []
        for serial, width, height, fps in CAMERA_CONFIGS:
            try:
                cam = RealsenseCamera(
                    device_id=serial,
                    enable_depth=SAVE_DEPTH,
                    width=width,
                    height=height,
                    fps=fps
                )
                cameras.append(cam)
                print(f"Camera {serial}: {width}x{height}@{fps}fps, depth={SAVE_DEPTH}")
            except Exception as e:
                print(f"Cannot connect to camera {serial}: {e}")
        print(f"Found {len(cameras)} cameras for recording.")

    init_time = time.time()
    # main loop

    # IK 性能监控器
    ik_monitor = None
    if DEBUG_IK_PERF:
        log_dir = os.path.join(cur_dir, "pyroki", "log") if DEBUG_IK_PERF_CSV else None
        ik_monitor = create_monitor(
            log_dir=log_dir,
            print_interval=10.0,
            enable_logging=DEBUG_IK_PERF_CSV,
            enable_console=DEBUG_IK_PERF_CONSOLE
        )
        # 设置GUI显示组件
        if ik_perf_gui_displays:
            ik_monitor.set_gui_displays(*ik_perf_gui_displays)
    
    is_recording = False
    
    # 键盘监听（在单独线程中）
    import threading
    from pynput import keyboard
    
    def on_press(key):
        try:
            if key == keyboard.Key.space:
                recording_state["toggle_requested"] = True
        except:
            pass
    
    # 启动键盘监听线程
    keyboard_listener = keyboard.Listener(on_press=on_press)
    keyboard_listener.start()
    print("[INFO] Keyboard listener started. Press SPACE to toggle recording.")

    try:
        while True:

            start_time = time.time()

            # 处理录制切换请求（来自按钮或空格键）
            if recording_state["toggle_requested"]:
                recording_state["toggle_requested"] = False
                
                if not is_recording:
                    # 开始录制
                    try:
                        recorder = DataRecorder(save_dir=os.path.join(record_dir, f"session_{time.strftime('%Y%m%d_%H%M%S')}"), save_depth=SAVE_DEPTH)
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
                    # 停止录制
                    try:
                        if recorder is not None:
                            recorder.save()
                            print("⚪ Recording stopped and saved.")
                            
                            # 重命名会话目录
                            try:
                                session_dir = recorder.save_dir if hasattr(recorder, 'save_dir') else None
                                if session_dir is not None:
                                    session_dir = str(session_dir)
                                    traj_path = os.path.join(session_dir, "trajectory.json")
                                    if os.path.exists(traj_path):
                                        import json
                                        with open(traj_path, 'r') as f:
                                            traj = json.load(f)
                                        num_frames = len(traj)
                                        if num_frames >= 2:
                                            t0 = datetime.fromisoformat(traj[0]["timestamp"]) 
                                            t1 = datetime.fromisoformat(traj[-1]["timestamp"]) 
                                            duration = max((t1 - t0).total_seconds(), 1e-6)
                                            fps = int(round(num_frames / duration))
                                        else:
                                            fps = 0

                                        existing = [d for d in os.listdir(record_dir) 
                                                    if os.path.isdir(os.path.join(record_dir, d)) and d.startswith("session_")]
                                        session_index = len(existing) - 1
                                        timestamp_str = time.strftime('%Y%m%d_%H%M%S')
                                        new_name = f"session_{session_index:04d}_{fps}hz_{timestamp_str}"
                                        new_path = os.path.join(record_dir, new_name)
                                        if not os.path.exists(new_path):
                                            os.rename(session_dir, new_path)
                                            print(f"Session renamed to {new_name}")
                            except Exception as e:
                                print("Failed to rename session directory:", e)
                    except Exception as e:
                        print("Failed to save recording:", e)
                    finally:
                        recorder = None
                        is_recording = False
                        recording_state["active"] = False
                        update_record_button(False)
            
            # 旧的 checkbox 逻辑已移除

            if USE_CAMERA:
                # DEBUG 时打开可视化窗口
                for cam in cameras:
                    cam.get_data(viz=DEBUG)

            target_pose = None
            target_gripper = None

            if USE_VR:
                target_cmd = vr.get_vr_command() # [left, right, body] 3 x 7
                
                if target_cmd is None:
                    time.sleep(0.001)
                    continue

                target_pose_raw = target_cmd[0][:2] # left + right
                target_gripper = target_cmd[1] # left + right gripper
                
                # 应用工作空间约束
                left_constrained, right_constrained = workspace_constraint.constrain_dual_arm(
                    target_pose_raw[0], 
                    target_pose_raw[1],
                    left_shoulder_position,
                    right_shoulder_position
                )
                target_pose = [left_constrained, right_constrained]
                
                # 更新VR target可视化（如果开关打开）
                if vr_target_frames is not None:
                    current_state = vr_target_vis_enabled.value
                    # 检测状态变化并打印调试信息
                    if current_state != vr_target_vis_prev_state:
                        if current_state:
                            print("[INFO] VR Target Frames visualization ENABLED")
                        else:
                            print("[INFO] VR Target Frames visualization DISABLED")
                        vr_target_vis_prev_state = current_state
                    
                    if current_state:
                        for i, frame in enumerate(vr_target_frames):
                            # target_pose格式: [qw, qx, qy, qz, x, y, z]
                            if i < len(target_pose):
                                pose = target_pose[i]
                                frame.position = pose[4:7]  # x, y, z
                                frame.wxyz = pose[0:4]      # qw, qx, qy, qz
                                frame.visible = True
                    else:
                        # 开关关闭时隐藏
                        for frame in vr_target_frames:
                            frame.visible = False
                # print("target gripper: ", target_gripper)
            else:
                # get target pose and elbow angles from UI
                target_pose = viser.get_target_pose()
                target_gripper = np.array([1, 1])

            if target_pose is None:
                raise "target pose is None"
            # print("target pose: ", target_pose)

            # IK 性能监控：计时求解
            if ik_monitor:
                ik_start_time = time.perf_counter()
            
            solved, left_arm_cmd, right_arm_cmd = IK_triangle.solve(left_shoulder_position, left_shoulder_orientation,
                                right_shoulder_position, right_shoulder_orientation,
                                target_pose,
                                collision_checker, joints_lower_limit, joints_upper_limit)
            
            # IK 性能监控：记录结果
            if ik_monitor:
                ik_elapsed = time.perf_counter() - ik_start_time
                ik_monitor.record_solve(ik_elapsed, solved)

            '''
            update real/simulated robot
            '''
            if solved:
                solution = np.concatenate((left_arm_cmd, right_arm_cmd), 0)
                current_joints = solution
                # print("solution: ", solution)
                # current_ee_pose_pyroki = solver.get_current_ee_pose(current_joints)
                # print("current ee pose pyroki: ", current_ee_pose_pyroki)
                current_ee_pose_analytic = IK_triangle.get_current_ee_pose()
                # print("current ee pose analytic: ", current_ee_pose_analytic)
            else:
                # # solve IK
                    # pyroki_solution = solver.solve_ik(
                    #     target_pose,
                    #     current_joints=current_joints,
                    #     manipulability_weight=manip_weight.value,
                    #     limit_weight=limit_weight.value
                    # )
                    # print("pyroki: ", pyroki_solution)
                    # current_joints[7:14] = pyroki_solution[7:14]
                # 解析解失败时，保持当前姿态不变
                # 注意：不使用数值优化IK作为备选，因为：
                # 1. 解析解失败的目标通常也是数值优化无法达到的（物理约束）
                # 2. 数值优化IK很慢（8-15ms），会导致系统卡顿
                # 3. 数值优化IK在这些场景下成功率也很低
                # 保持 current_joints 不变，机器人停留在当前安全位置
                pass  # 不做任何更新，保持当前关节角度
                

            # print("solution: ", solution)
            T = solver.forward_kinematics(current_joints) # 显示肘部的结果
            
            if USE_VR:
                viser.update_vis_frame(np.array([target_pose[0], target_pose[1], T[0]]))
            else:
                # left arm 2 and 5
                # right arm 10 and 13
                # viser.update_vis_frame(np.array([left_shoulder_pose, T[6], T[9], 
                #                                 right_shoulder_pose, T[14], T[17]]))

                viser.update_vis_frame(np.array([current_ee_pose[0], T[0], T[0], 
                                                current_ee_pose[1], T[0], T[0]]))

            elapsed_time = time.time() - start_time
            # if DEBUG:
            #     print("elapsed_time: ", elapsed_time)

            if USE_REAL:
                current_left_arm_position,  current_left_gripper_position = controller.get_left_position()
                current_right_arm_position,  current_right_gripper_position = controller.get_right_position()
                real_robot_joint_position = np.concatenate((current_left_arm_position, current_right_arm_position), 0)
                viser.update_results(real_robot_joint_position, elapsed_time)

                left_target_cmd = - target_gripper[0] * 0.95
                right_target_cmd = - target_gripper[1] * 0.95

                gripper = [left_target_cmd, right_target_cmd]

                
                # print("left gripper position: ", current_left_gripper_position)
                # print("right gripper position: ", current_right_gripper_position)

                # print("target left gripper: ", left_target_cmd)
                # if DEBUG:
                #     print("target right gripper: ", right_target_cmd)


                # 下发指令
                if time.time() - init_time < 5:
                    target_joints_left = (np.array(current_joints[:7])-np.array(current_left_arm_position)) * 0.1 + np.array(current_left_arm_position)
                    target_joints_right = (np.array(current_joints[7:14])-np.array(current_right_arm_position)) * 0.1 + np.array(current_right_arm_position)
                    target_joints_left = target_joints_left.tolist()
                    target_joints_right = target_joints_right.tolist()
                else:
                    target_joints_left = current_joints[:7]
                    target_joints_right = current_joints[7:14]

                # record data
                obs_joints = real_robot_joint_position
                obs_ee_poses = solver.get_current_ee_pose(obs_joints)
                obs_gripper_joints = [current_left_gripper_position, current_right_gripper_position]
                obs = [obs_joints, obs_ee_poses, obs_gripper_joints]

                action_joints = np.array([target_joints_left , target_joints_right]).flatten()
                action_ee_poses = solver.get_current_ee_pose(action_joints)
                action_gripper_joints = gripper
                act = [action_joints, action_ee_poses, action_gripper_joints]


                if is_recording and recorder is not None:
                    if USE_CAMERA:
                        recorder.record(*obs, *act, cameras)
                    else:
                        recorder.record(*obs, *act)
                
                controller.set_left_position(target_joints_left, left_target_cmd, current_left_arm_position, current_left_gripper_position)            
                controller.set_right_position(target_joints_right, right_target_cmd, current_right_arm_position, current_right_gripper_position)          
            else:
                viser.update_results(current_joints, elapsed_time)

            if not solved:
                state_logger(
                    datetime.utcnow().isoformat(),
                    target_pose,
                    current_joints,
                )

            # print("real robot joint position: ", current_arm_position)
            # controller.set_left_position(current_joints[:7], -0.3,
            #                            current_arm_position, current_gripper_position)
            
            # viser.update_results(current_joints, elapsed_time)

            time.sleep(0.01)

            # once_time = time.time() - start_time
            # print("once_time: ", once_time)

    except Exception as e:
        print("Exception: ", e)
        traceback.print_exc()
    finally:
        # 停止键盘监听
        keyboard_listener.stop()
        # 保存录制数据
        if recorder is not None:
            recorder.save()
        if state_logger is not None:
            state_logger.close()
        # 关闭 IK 性能监控器
        if ik_monitor is not None:
            ik_monitor.close()
        # 打印工作空间约束统计
        if 'workspace_constraint' in locals():
            workspace_constraint.print_stats()
        viser.stop()

if __name__ == "__main__":
    main()