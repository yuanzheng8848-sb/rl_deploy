from viser_base import ViserBase
from robot_ik_solver import BaseIKSolver
from vr import VRUpperBodyTeleop

import numpy as np
import time
import yaml
from pathlib import Path
from plot import PoseRecorderMatplotlib

from realsense_camera import RealsenseCamera

import traceback
import csv
import matplotlib.pyplot as plt

# 解析解IK用的
from scipy.spatial.transform import Rotation as R
import importlib.util
import sys
# 定义文件绝对路径
file_path = "/home/sj/Documents/moqi_workspace_20251119_211400/IK/analytic_IK.py"
# 使用 importlib 导入
spec_ik = importlib.util.spec_from_file_location("analytic_IK", file_path)
ik_module = importlib.util.module_from_spec(spec_ik)
spec_ik.loader.exec_module(ik_module)

file_path = "/home/sj/Documents/moqi_workspace_20251119_211400/IK/collision_check.py"
spec_collision = importlib.util.spec_from_file_location("analytic_IK", file_path)
collision_module = importlib.util.module_from_spec(spec_collision)
spec_collision.loader.exec_module(collision_module)

file_path = "/home/sj/Documents/moqi_workspace_20251119_211400/openarm/openarm_controller.py"
spec_openarm = importlib.util.spec_from_file_location("openarm_controller", file_path)
openarm_module = importlib.util.module_from_spec(spec_openarm)
spec_openarm.loader.exec_module(openarm_module)

'''
加入 VR 操作

所有的 pose 顺序为 qw qx qy qz x y z
'''


USE_VR = True 
USE_REAL = True

import pdb


if __name__ == "__main__":

    cfgs_path = Path("/home/sj/Documents/moqi_workspace_20251119_211400/pyroki/config/")

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

    manip_weight = viser._server.gui.add_slider("Manipulability Weight", 0.0, 10.0, 0.001, 0.0)
    limit_weight = viser._server.gui.add_slider("Limit Avoidance Weight", 0.0, 100.0, 0.01, 0.0)

    
    # main loop
    current_joints = viser.get_init_joints_for_sim()
    print(current_joints)
    start_time = None

    # openarm configuration
    origin_position = np.array([0.0, 0.0, 0.0])
    l1 = 0.22
    l2 = 0.216
    IK_triangle = ik_module.Triangle(l1, l2, origin_position)
    current_ee_pose = solver.get_current_ee_pose(current_joints)
    print("current_ee_pose: ", current_ee_pose)
    IK_triangle.set_init_ee_pose(current_ee_pose[0], current_ee_pose[1])

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
        ip_vr = "10.255.8.41"
        
        vr = VRUpperBodyTeleop(
            cfg_vr,
            ip_vr,
            IK_triangle.get_current_ee_pose,
            "/home/sj/Documents/moqi_workspace_20251119_211400/pyroki/config",
            )

        try:
            target_pose, gripper_width = vr.wait_for_initial_states()
            print("vr inital cmd: ", target_pose, gripper_width)
        except Exception as e:
            print("can not receive VR signal")
            vr.stop()
            viser.stop()
            exit(1)

    cameras = []#[RealsenseCamera("150622074105"), RealsenseCamera("236422072385")]

    # main loop

    init_time = time.time()
    recorder = PoseRecorderMatplotlib()
   

    try:
        while True:


            start_time = time.time()

            for c in cameras:
                c.get_data(viz=True)

            target_pose = None
            target_gripper = None

            if USE_VR:
                target_cmd = vr.get_vr_command() # [left, right, body] 3 x 7
                
                if target_cmd is None:
                    time.sleep(0.001)
                    continue

                target_pose = target_cmd[0][:2] # left + right
                target_gripper = target_cmd[1] # left + right gripper
                # print("target gripper: ", target_gripper)
                # print("target pose: ", target_pose)
            else:
                # get target pose and elbow angles from UI
                target_pose = viser.get_target_pose()
                target_gripper = np.array([1, 1])

            if target_pose is None:
                raise "target pose is None"
            # print("target pose: ", target_pose)

            solved, left_arm_cmd, right_arm_cmd = IK_triangle.solve(left_shoulder_position, left_shoulder_orientation,
                                right_shoulder_position, right_shoulder_orientation,
                                target_pose,
                                collision_checker, joints_lower_limit, joints_upper_limit)

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
                print("No solution found!")
            
                # # solve IK
                # pyroki_solution = solver.solve_ik(
                #     target_pose,
                #     current_joints=current_joints,
                #     manipulability_weight=manip_weight.value,
                #     limit_weight=limit_weight.value
                # )
                # print("pyroki: ", pyroki_solution)
                # current_joints[7:14] = pyroki_solution[7:14]

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
            # print("elapsed_time: ", elapsed_time)

            if USE_REAL:
                current_left_arm_position,  current_left_gripper_position = controller.get_left_position()
                current_right_arm_position,  current_right_gripper_position = controller.get_right_position()
                real_robot_joint_position = np.concatenate((current_left_arm_position, current_right_arm_position), 0)
                all_joints_ee_pose = solver.forward_kinematics(real_robot_joint_position)

                 # 获取robot对象
                robot = solver._robot

                # --------- 只取 TCP ----------
                # and_tcp 是 URDF 里专门预留的 “工具中心点”，它已经把夹爪/手指长度算进去了

                left_tcp_row  = robot.links.names.index('openarm_left_link7')
                right_tcp_row = robot.links.names.index('openarm_right_link7')

                                
                # left_tcp_q  = all_joints_ee_pose[left_tcp_row, 0:4]   # [qx,qy,qz,qw]
                # right_tcp_q = all_joints_ee_pose[right_tcp_row, 0:4]
                # left_tcp_t  = all_joints_ee_pose[left_tcp_row, 4:7]   # [x,y,z]
                # right_tcp_t = all_joints_ee_pose[right_tcp_row, 4:7]

                # # 组装成 2×7，和 target_pose 同形状
                # tcp_pose = np.array([
                #     [*left_tcp_q,  *left_tcp_t],
                #     [*right_tcp_q, *right_tcp_t]
                # ])
                # left_tcp_pose = np.array([[*left_tcp_q,  *left_tcp_t]])
                # right_tcp_pose = np.array([[*right_tcp_q,  *right_tcp_t]])
                left_tcp_pose = np.array(all_joints_ee_pose[left_tcp_row,0:7])
                right_tcp_pose = np.array(all_joints_ee_pose[right_tcp_row,0:7])
                left_target_pose=target_pose[0]
                right_target_pose=target_pose[1]
               
                # 粗跟踪误差数据记录
                ts_now = time.time()
                # recorder.append(ts_now, left_target_pose, left_tcp_pose)
                recorder.append(ts_now, right_target_pose, right_tcp_pose)
                            
              
                

                viser.update_results(real_robot_joint_position, elapsed_time)

                # print("left gripper position: ", current_left_gripper_position)
                # print("right gripper position: ", current_right_gripper_position)
                left_target_cmd = - target_gripper[0] * 0.95
                right_target_cmd = - target_gripper[1] * 0.95
                # print("target left gripper: ", left_target_cmd)
                # print("target right gripper: ", right_target_cmd)


                # 下发指令
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
            else:
                viser.update_results(current_joints, elapsed_time)

            # print("real robot joint position: ", current_arm_position)
            # controller.set_left_position(current_joints[:7], -0.3,
            #                            current_arm_position, current_gripper_position)
            
            # viser.update_results(current_joints, elapsed_time)
            


            time.sleep(0.01)
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught, exiting.")        

    except Exception as e:
        print("Exception: ", e)
        traceback.print_exc()          # 把完整栈打出来
    finally:
        recorder.save_csv("pose_record_left_arm.csv")
        print("✅ 轨迹数据已保存到 pose_record_left_arm.csv")
        recorder.plot_3d(ignore_initial_seconds=10.0,align_to_target_start=True,frame_marker_ratio=0.5)
        
        
        viser.stop()

