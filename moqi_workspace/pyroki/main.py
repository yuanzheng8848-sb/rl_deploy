from viser_base import ViserBase
from robot_ik_solver import BaseIKSolver
from vr import VRUpperBodyTeleop

import numpy as np
import time
import yaml
from pathlib import Path

# 解析解IK用的
from scipy.spatial.transform import Rotation as R
import importlib.util
import sys
# 定义文件绝对路径
file_path = "/home/sj/Documents/moqi_workspace_20251119_211400/IK/analytic_IK.py"
# 使用 importlib 导入
spec_ik = importlib.util.spec_from_file_location("analytic_IK", file_path)
ik_module = importlib.util.module_from_spec(spec_ik)
# sys.modules["ik_module"] = ik_module
spec_ik.loader.exec_module(ik_module)

file_path = "/home/sj/Documents/moqi_workspace_20251119_211400/IK/collision_check.py"
spec_collision = importlib.util.spec_from_file_location("analytic_IK", file_path)
collision_module = importlib.util.module_from_spec(spec_collision)
# sys.modules["ik_module"] = ik_module
spec_collision.loader.exec_module(collision_module)




import pdb


if __name__ == "__main__":

    cfgs_path = Path("/home/sj/Documents/moqi_workspace_20251119_211400/pyroki/config/")

    # 配置 IK solver
    cfgs_robot = yaml.safe_load( (cfgs_path / "robot.yaml").read_text())
    cfgs_solver = yaml.safe_load( (cfgs_path / "solver.yaml").read_text())
    solver = BaseIKSolver(cfgs_solver, cfgs_robot, True)

    # 配置 viser
    cfg_viser = yaml.safe_load( (cfgs_path / "viser.yaml").read_text())

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

    # 配置 VR 设备
    cfg_vr = yaml.safe_load((cfgs_path / "vr.yaml").read_text())
    ip_vr = "192.168.1.4"
    vr = VRUpperBodyTeleop(
        cfg_vr,
        ip_vr,
        ros_interface.get_ee_pose,
        script_path.as_posix(),
        )

    try:
        target_pose, gripper_width = vr.wait_for_initial_states()
    except Exception as e:
        logger.error(f"Exception during waiting for vr command: {e}")
        vr.stop()
        ros_interface.stop()
        viser.stop()
        exit(1)


    # main loop
    current_joints = viser.get_init_joints_for_sim()

    start_time = None

    # test
    origin_position = np.array([0.0, 0.0, 0.0])
    l1 = 0.22
    l2 = 0.216
    test_triangle = ik_module.Triangle(l1, l2, origin_position)

    T = solver.forward_kinematics(np.zeros(14)) # 显示肘部的结果

    # shoulder in world frame
    left_shoulder_position = T[3][4:]
    left_shoulder_position[1] += T[4][4:][1] - T[3][4:][1]
    left_shoulder_orientation = R.from_matrix(np.array([[0,-1,0],[0,0,1],[-1,0,0]]))
    left_shoulder_pose = np.concatenate((left_shoulder_orientation.as_quat(scalar_first=True), left_shoulder_position), 0)

    right_shoulder_position = T[3+8][4:]
    right_shoulder_position[1] += T[4+8][4:][1] - T[3+8][4:][1]
    right_shoulder_orientation = R.from_matrix(np.array([[0,-1,0],[0,0,1],[-1,0,0]]))
    right_shoulder_pose = np.concatenate((right_shoulder_orientation.as_quat(scalar_first=True), left_shoulder_position), 0)

    # collision checker
    collision_checker = collision_module.OpenArmCollisionChecker(left_shoulder_position, right_shoulder_position, viser._server)
    
    # joint limit checker
    print("active joint: ", solver._robot.joints.actuated_names)
    joints_upper_limit = np.array(solver._robot.joints.upper_limits) + 0.0001
    joints_lower_limit = np.array(solver._robot.joints.lower_limits) - 0.0001

    print("upper bound: ", joints_upper_limit)
    print("lower bound: ", joints_lower_limit)

    try:
        while True:

            start_time = time.time()
            # get target pose and elbow angles from UI
            target_pose = viser.get_target_pose()
            target_transform = np.array([[0,-1,0],[0,0,-1],[1,0,0]])

            left_wrist_target_position = target_pose[0][4:]
            left_wrist_target_orientation = R.from_quat(target_pose[0][:4], scalar_first=True) * R.from_matrix(target_transform)
            
            right_wrist_target_position = target_pose[1][4:]
            right_wrist_target_orientation = R.from_quat(target_pose[1][:4], scalar_first=True) * R.from_matrix(target_transform)
            
            # target position 是在 shoulder坐标系下的，shoulder 就是A点位置
            left_target_position = left_wrist_target_position - left_shoulder_position
            left_target_position = left_shoulder_orientation.as_matrix().T @ left_target_position
            print("left target_position: ", left_target_position)

            right_target_position = right_wrist_target_position - right_shoulder_position
            right_target_position = right_shoulder_orientation.as_matrix().T @ right_target_position
            print("right target_position: ", right_target_position)


            solved = False

            #for theta in np.arange(0, 2*np.pi, 0.1):
            for theta in np.arange(np.pi/2.5, 0, -0.5):

                '''
                left arm
                '''

                # 限制最远长度
                if np.linalg.norm(left_target_position) > (l1 + l2):
                    left_target_position = left_target_position / np.linalg.norm(left_target_position) * (l1 + l2 - 0.001)

                left_elbow_joint = (l1**2 + l2**2 - np.linalg.norm(left_target_position)**2) / (2*l1*l2)
                direction = -1 
                left_elbow_joint = np.pi - np.arccos(left_elbow_joint)
                
                Rot, Rot2, left_C_in_A = test_triangle.cal(left_target_position, theta, direction)
                # 肩膀3关节朝向
                r = R.from_matrix(Rot)
                j1, j2, j3 = r.as_euler('ZYX')

                # 计算腕部3关节的朝向
                # Rot2 为肘关节相对于肩关节的旋转矩阵
                # left_target_orientation = R.from_quat(target_pose[0][:4], scalar_first=True)
                left_target_orientation = left_wrist_target_orientation
                left_target_orientation = left_shoulder_orientation.inv() * left_target_orientation
                left_target_orientation = Rot2.T @ left_target_orientation.as_matrix()
                j5, j6, j7 = R.from_matrix(left_target_orientation).as_euler('XYZ')

                

                left_arm_cmd = np.array([j1, j2, j3, left_elbow_joint, j5, -j6, j7])

                # joint limit check
                left_joint_out_of_limit = False
                for i in range(7):
                    if left_arm_cmd[i] < joints_lower_limit[i] or left_arm_cmd[i] > joints_upper_limit[i]:
                        print("left arm joint ", i, " cmd: ", left_arm_cmd[i] ," out of limit [", joints_lower_limit[i], ", ", joints_upper_limit[i], "]")
                        left_joint_out_of_limit = True
                        break
                
                if left_joint_out_of_limit:
                    continue
                
                '''
                right arm
                ''' 

                # 限制最远长度
                if np.linalg.norm(right_target_position) > (l1 + l2):
                    right_target_position = right_target_position / np.linalg.norm(right_target_position) * (l1 + l2 - 0.001)

                right_elbow_joint = (l1**2 + l2**2 - np.linalg.norm(right_target_position)**2) / (2*l1*l2)
                direction = -1 
                right_elbow_joint = np.pi - np.arccos(right_elbow_joint)
                
                Rot, Rot2, right_C_in_A = test_triangle.cal(right_target_position, -theta, direction)
                # 肩膀3关节朝向
                r = R.from_matrix(Rot)
                j1, j2, j3 = r.as_euler('ZYX')

                # 计算腕部3关节的朝向
                # Rot2 为肘关节相对于肩关节的旋转矩阵
                right_target_orientation = right_wrist_target_orientation
                right_target_orientation = right_shoulder_orientation.inv() * right_target_orientation
                right_target_orientation = Rot2.T @ right_target_orientation.as_matrix()
                j5, j6, j7 = R.from_matrix(right_target_orientation).as_euler('XYZ')

                right_arm_cmd = np.array([-j1, j2, j3, right_elbow_joint, j5, -j6, -j7])

                # joint limit check
                right_joint_out_of_limit = False
                for i in range(7):
                    if right_arm_cmd[i] < joints_lower_limit[7 + i] or right_arm_cmd[i] > joints_upper_limit[7 + i]:
                        print("right arm joint ", i, " cmd: " ,right_arm_cmd[i] ," out of limit [", joints_lower_limit[7 + i], ", ", joints_upper_limit[7 + i], "]")
                        right_joint_out_of_limit = True
                        break
                
                if right_joint_out_of_limit:
                    continue


                '''
                collision check
                '''
                left_B_in_world = left_shoulder_orientation.as_matrix() @ left_target_position + left_shoulder_position
                left_C_in_world = left_shoulder_orientation.as_matrix() @ left_C_in_A + left_shoulder_position

                right_B_in_world = right_shoulder_orientation.as_matrix() @ right_target_position + right_shoulder_position
                right_C_in_world = right_shoulder_orientation.as_matrix() @ right_C_in_A + right_shoulder_position
                
                collision_result = collision_checker.check_collision(left_B_in_world, left_C_in_world, right_B_in_world, right_C_in_world)

                if collision_result:
                    continue
                else:
                    solved = True
                    break

            '''
            update real/simulated robot
            '''
            if solved:
                solution = np.concatenate((left_arm_cmd, right_arm_cmd), 0)
                current_joints = solution
                print("solution: ", solution)
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
            # left arm 2 and 5
            # right arm 10 and 13
            viser.update_target_pose(np.array([left_shoulder_pose, T[6], T[9], 
                                            right_shoulder_pose, T[14], T[17]]))


            elapsed_time = time.time() - start_time
            print("elapsed_time: ", elapsed_time)
            viser.update_results(current_joints, elapsed_time)
            

            time.sleep(0.01)

    except Exception as e:
        print("Exception: ", e)
    finally:
        viser.stop()

