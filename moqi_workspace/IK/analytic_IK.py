import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import time
import pdb

import importlib.util
# 定义文件绝对路径
file_path = "/home/sj/Documents/moqi_workspace_20251119_211400/IK/utils.py"
# 使用 importlib 导入
spec_utils = importlib.util.spec_from_file_location("utils", file_path)
utils_module = importlib.util.module_from_spec(spec_utils)
# sys.modules["ik_module"] = ik_module
spec_utils.loader.exec_module(utils_module)


def rotx(theta):
    R = np.array([[1, 0, 0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta), np.cos(theta)]])
    return R

def roty(theta):
    R = np.array([[np.cos(theta), 0, np.sin(theta)],
                  [0, 1, 0],
                  [-np.sin(theta), 0, np.cos(theta)]])
    return R

def rotz(theta):
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    return R

class Triangle:

    def __init__(self, l1, l2, origin_position=np.array([0,0,0])):
        # l1: 大臂长度
        # l2: 小臂长度
        # A 为三角形原点
        self.AC = l1
        self.BC = l2
        self.A = origin_position

        self.l1 = l1
        self.l2 = l2

        # 肘关节 theta 角度
        self.left_theta = np.pi / 4
        self.right_theta = np.pi / 4
        self.left_theta_generator = utils_module.Center_Expand_Generator(0, np.pi / 2)
        self.right_theta_generator = utils_module.Center_Expand_Generator(0, np.pi / 2)

        # 记录 left ee in world frame
        self.current_left_ee_pose = None
        self.current_right_ee_pose = None

    def set_init_ee_pose(self, left_ee_pose, right_ee_pose):
        # in world frame
        self.current_left_ee_pose = left_ee_pose
        self.current_right_ee_pose = right_ee_pose

    def get_current_ee_pose(self):
        # 为了给 VR 调用的， 从原本的 qw qx qy qz x y z => x y z qx qy qz qw
        
        if self.current_left_ee_pose is None or self.current_right_ee_pose is None:
            return None
        
        left_ee_pose = np.array([self.current_left_ee_pose[4], self.current_left_ee_pose[5], self.current_left_ee_pose[6],
                        self.current_left_ee_pose[1], self.current_left_ee_pose[2], self.current_left_ee_pose[3], self.current_left_ee_pose[0]])
        
        right_ee_pose = np.array([self.current_right_ee_pose[4], self.current_right_ee_pose[5], self.current_right_ee_pose[6],
                        self.current_right_ee_pose[1], self.current_right_ee_pose[2], self.current_right_ee_pose[3], self.current_right_ee_pose[0]])
        
        return np.array([left_ee_pose, right_ee_pose, np.array([0, 0, 0, 0, 0, 0, 1])])

    def cal(self, B, theta, elbow_direction=1):
        # target_position，目标位置 np.array(3) x y z
        # theta 用于从多解中确认一个
        start = time.time()

        AB = B - self.A

        d = np.linalg.norm(AB)
        t = (1 + (self.AC**2 - self.BC**2)/d**2) / 2

        M = t * AB + self.A
        h = np.sqrt(self.AC**2 - (t*d)**2)

        N = AB / d

        # 参考向
        ref = np.zeros(3)
        ref[2] = 1
        U = np.cross(ref, N)

        if np.linalg.norm(U) < 1e-10:
            ref[0] = 0
            ref[1] = 1
            U = np.cross(ref, N)

        U = U / np.linalg.norm(U)

        V = np.cross(N, U)

        # print("V norm: ", np.linalg.norm(V))

        C = M + h * (U * np.cos(theta) + V * np.sin(theta))

        # print("N * U", np.dot(N, U))
        # print("N * V", np.dot(N, V))

        # shoulder to elbow
        AC = C - self.A
        z = np.cross(AC, AB)
        z = z / np.linalg.norm(z)
        z *= elbow_direction  # 控制肘部方向
        x = AC / np.linalg.norm(AC)
        y = np.cross(z, x)
        Rot = np.array([x, y, z]).T

        # elbow to wrist
        BC = B - C
        x2 = BC / np.linalg.norm(BC)
        y2 = np.cross(z, x2)
        Rot2 = np.array([x2, y2, z]).T

        end = time.time()
        # print("duration: ", end - start)

        return Rot, Rot2, C
        
    def solve(self, 
            left_shoulder_position, 
            left_shoulder_orientation,
            right_shoulder_position,
            right_shoulder_orientation,
            target_pose,
            collision_checker, joints_lower_limit, joints_upper_limit): # 用于确定解的合理性的

        # 针对 viser 定义的末端和 解析解实际末端差异的情况
        # target_pose: qw qx qy qz x y z

        target_transform = np.array([[0,-1,0],[0,0,-1],[1,0,0]])

        left_wrist_target_position = target_pose[0][4:]
        left_wrist_target_orientation = R.from_quat(target_pose[0][:4], scalar_first=True) * R.from_matrix(target_transform)
        
        right_wrist_target_position = target_pose[1][4:]
        right_wrist_target_orientation = R.from_quat(target_pose[1][:4], scalar_first=True) * R.from_matrix(target_transform)
        
        # target position 是在 shoulder坐标系下的，shoulder 就是A点位置
        left_target_position = left_wrist_target_position - left_shoulder_position
        left_target_position = left_shoulder_orientation.as_matrix().T @ left_target_position
        # print("left target_position in body frame: ", left_target_position)

        right_target_position = right_wrist_target_position - right_shoulder_position
        right_target_position = right_shoulder_orientation.as_matrix().T @ right_target_position
        # print("right target_position in body frame: ", right_target_position)

        solved = False
        left_solved = False
        right_solved = False

        left_theta = self.left_theta
        right_theta = self.right_theta
        theta_step_size = 0.05  # 从0.1减小到0.05，提高搜索精度
        self.left_theta_generator.set_start(left_theta, theta_step_size)
        self.right_theta_generator.set_start(right_theta, theta_step_size)

        left_arm_cmd = None
        right_arm_cmd = None

        iteration_count = 0
        max_iterations = 80  # 从50增加到80，配合更小的步长
        
        while (left_theta != None) and (right_theta != None):
            iteration_count += 1
            if iteration_count > max_iterations:
                print(f"⚠️ IK solver exceeded max iterations ({max_iterations})")
                break
                
            # print("left_theta: ", left_theta)
            # print("right theta: ", right_theta)

            if not left_solved: 
                '''
                left arm
                '''

                # 限制最远长度
                if np.linalg.norm(left_target_position) > (self.l1 + self.l2):
                    left_target_position = left_target_position / np.linalg.norm(left_target_position) * (self.l1 + self.l2 - 0.001)

                left_elbow_joint = (self.l1**2 + self.l2**2 - np.linalg.norm(left_target_position)**2) / (2 * self.l1 * self.l2)
                direction = -1 
                left_elbow_joint = np.pi - np.arccos(left_elbow_joint)
                
                Rot, Rot2, left_C_in_A = self.cal(left_target_position, left_theta, direction)
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
                    left_solved = False
                else:
                    left_solved = True
            
            
            if not right_solved:

                '''
                right arm
                ''' 

                # 限制最远长度
                if np.linalg.norm(right_target_position) > (self.l1 + self.l2):
                    right_target_position = right_target_position / np.linalg.norm(right_target_position) * (self.l1 + self.l2 - 0.001)

                right_elbow_joint = (self.l1**2 + self.l2**2 - np.linalg.norm(right_target_position)**2) / (2 * self.l1 * self.l2)
                direction = -1 
                right_elbow_joint = np.pi - np.arccos(right_elbow_joint)
                
                Rot, Rot2, right_C_in_A = self.cal(right_target_position, -right_theta, direction)
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
                    right_solved = False
                else:
                    right_solved = True

            if left_solved and right_solved:
                '''
                collision check， 左右都有解才碰撞检测
                '''
                left_B_in_world = left_shoulder_orientation.as_matrix() @ left_target_position + left_shoulder_position
                left_C_in_world = left_shoulder_orientation.as_matrix() @ left_C_in_A + left_shoulder_position

                right_B_in_world = right_shoulder_orientation.as_matrix() @ right_target_position + right_shoulder_position
                right_C_in_world = right_shoulder_orientation.as_matrix() @ right_C_in_A + right_shoulder_position
                
                collision_result = collision_checker.check_collision(left_B_in_world, left_C_in_world, right_B_in_world, right_C_in_world)

                # print("collision result: ", collision_result)
                if collision_result[0] or collision_result[1] or collision_result[2]:
                    if collision_result[0]:
                        print("❌ Left arm collision with body detected")
                        left_solved = False
                    if collision_result[1]:
                        print("❌ Right arm collision with body detected")
                        right_solved = False
                    if collision_result[2]:
                        print("❌ Left-Right arms collision detected")
                        left_solved = False
                        right_solved = False

            if not left_solved:
                left_theta = self.left_theta_generator.next()
            if not right_solved:
                right_theta = self.right_theta_generator.next()

            if left_solved and right_solved:
                solved = True
                # ToDo: waiting for test
                # pose in world frame: qw qx qy qz x y z
                self.current_left_ee_pose = target_pose[0]
                self.current_left_ee_pose[4:] = left_B_in_world
                self.current_right_ee_pose = target_pose[1]
                self.current_right_ee_pose[4:] = right_B_in_world
                break

        return solved, left_arm_cmd, right_arm_cmd


if __name__ == "__main__":

    origin_position = np.array([0.0, 0.0, 0.0])
    target_position = np.array([0.4, 0.0, 0.0])
    
    l1 = 0.633 / 2
    l2 = 0.633 / 2

    test_triangle = Triangle(l1, l2, origin_position)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot([origin_position[0], target_position[0]], [origin_position[1], target_position[1]], [origin_position[2], target_position[2]], linewidth=2, label="line")

    elbow_joint = (l1**2 + l2**2 - np.linalg.norm(target_position - origin_position)**2) / (2*l1*l2)
    elbow_joint = np.arccos(elbow_joint)
    print("elbow joint angle: ", elbow_joint)

    for theta in np.arange(0, 2*np.pi, 0.5):
        point, Rot = test_triangle.cal(target_position, theta)
        ax.scatter(*point, color='r')
        
        # j1 = 0.6
        # j2 = 0.2
        # j3 = 0.3

        # Rot = rotz(j1) @ roty(j2) @ rotx(j3)

        # ax.plot([0, Rot[:,0][0]], [0, Rot[:,0][1]], [0, Rot[:,0][2]], color='r')
        # ax.plot([0, Rot[:,1][0]], [0, Rot[:,1][1]], [0, Rot[:,1][2]], color='g')
        # ax.plot([0, Rot[:,2][0]], [0, Rot[:,2][1]], [0, Rot[:,2][2]], color='b')

        # Rot2 = np.array(
        #     [[ 0.82162381, -0.13673644, -0.55338726],
        #     [ 0.49096749, -0.32350869,  0.80888383],
        #     [-0.28962948, -0.93629337, -0.1986693 ]]
        # )

        # ax.plot([0, Rot2[:,0][0]], [0, Rot2[:,0][1]], [0, Rot2[:,0][2]], color='r')
        # ax.plot([0, Rot2[:,1][0]], [0, Rot2[:,1][1]], [0, Rot2[:,1][2]], color='g')
        # ax.plot([0, Rot2[:,2][0]], [0, Rot2[:,2][1]], [0, Rot2[:,2][2]], color='b')

        print("Rot", Rot)
        r = R.from_matrix(Rot)
        j1, j2, j3 = r.as_euler('ZYX')
        print("j1: ", j1)
        print("j2: ", j2)
        print("j3: ", j3)



    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_zlim(-5,5)

    plt.legend()
    plt.show()