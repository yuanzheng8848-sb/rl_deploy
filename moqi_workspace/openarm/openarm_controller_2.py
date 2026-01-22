import openarm_can as oa
import numpy as np
import time
# Create OpenArm instance

class OpenArmController:
    def __init__(self, enable_left=True, enable_right=True):

        self.enable_left = enable_left
        self.enable_right = enable_right

        self.motor_types = [oa.MotorType.DM8009,
                            oa.MotorType.DM8009, 
                            oa.MotorType.DM4340,
                            oa.MotorType.DM4340,
                            oa.MotorType.DM4310,
                            oa.MotorType.DM4310,
                            oa.MotorType.DM4310]

        self.send_ids = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]
        self.recv_ids = [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]

        # self.kp = np.array([20, 20, 20, 10, 5, 5, 5])
        # self.kv = np.array([2, 2, 2, 1, 0.5, 0.5, 0.5])
        self.kp = 1.0 * np.array([60, 50, 50, 50, 30, 30, 30])
        self.kv = 3.0 * np.array([2, 2, 2, 2, 0.5, 0.5, 0.5])
        self.ki = 1.0 * np.ones(7)
        self.acc_error_limit = 10.0

        self.left_last_error = np.zeros(len(self.send_ids))
        self.right_last_error = np.zeros(len(self.send_ids))

        self.left_acc_error = np.zeros(len(self.send_ids))
        self.right_acc_error = np.zeros(len(self.send_ids))

        if self.enable_left:
            self.left_arm = oa.OpenArm("can0", False)
            self.initialize_arm(self.left_arm)

        if self.enable_right:
            self.right_arm = oa.OpenArm("can1", False)
            self.initialize_arm(self.right_arm)

        # 如果双臂都启用，同时平滑移动到目标位置
        if self.enable_left and self.enable_right:
            # 左臂目标位置（弧度制）
            left_target = [-0.166811, -0.497863 , 0.635447, 1.499999, -0.627859, 0.507960, -0.168161]  # 左臂joint4抬起30度
            # 右臂目标位置（弧度制）  
            right_target = [0.166811, 0.497863, -0.635447, 1.499999, 0.627859, -0.507960, 0.168161]  # 右臂joint4抬起30度

            # 同时平滑移动双臂
            self._smooth_move_both_arms_to_position(left_target, right_target, duration=2.0)


    def initialize_arm(self, arm):
        # 初始化系统
        arm.init_arm_motors(self.motor_types, self.send_ids, self.recv_ids)

        arm.init_gripper_motor(oa.MotorType.DM4310, 0x08, 0x18)
        arm.set_callback_mode_all(oa.CallbackMode.IGNORE)

        arm.enable_all()
        arm.recv_all()

        # 平滑移动到初始位置
        arm.set_callback_mode_all(oa.CallbackMode.STATE)

        # 读取当前位置
        arm.refresh_all()
        arm.recv_all()
        current_positions = []
        for motor in arm.get_arm().get_motors():
            current_positions.append(motor.get_position())

        # 单个臂的初始化不再需要平滑移动，因为双臂会同时控制

        arm.recv_all()

        # torque control test
        arm.get_gripper().mit_control_all([oa.MITParam(0, 0, 0, 0, 0.15)])
        arm.get_arm().mit_control_all(
            [oa.MITParam(0, 0, 0, 0, 0.15), oa.MITParam(0, 0, 0, 0, 0.15)])
        arm.recv_all()

    def _smooth_move_to_position(self, arm, start_positions, target_positions, duration=3.0):
        """
        平滑移动到目标位置
        """
        import time

        start_time = time.time()
        steps = int(duration * 50)  # 50Hz控制频率

        print(f"开始平滑移动，从 {start_positions} 到 {target_positions}")

        for i in range(steps):
            elapsed = time.time() - start_time
            progress = min(elapsed / duration, 1.0)

            # 使用三次贝塞尔曲线进行平滑插值
            t = progress
            smooth_progress = t * t * (3.0 - 2.0 * t)  # smoothstep函数

            # 计算当前目标位置
            current_targets = []
            for j in range(len(start_positions)):
                current_target = start_positions[j] + (target_positions[j] - start_positions[j]) * smooth_progress
                current_targets.append(current_target)

            # 发送控制指令
            mit_params = []
            for j in range(len(current_targets)):
                # 使用较小的增益进行平滑控制
                mit_params.append(oa.MITParam(5.0, 1.0, current_targets[j], 0, 0))

            arm.get_arm().mit_control_all(mit_params)
            arm.recv_all()

            time.sleep(0.02)  # 50Hz控制频率

        print("平滑移动完成")

    def _smooth_move_both_arms_to_position(self, left_target, right_target, duration=3.0):
        """
        同时平滑移动双臂到目标位置
        """
        import time
        import threading

        # 读取当前位置
        left_current, _ = self.get_left_position()
        right_current, _ = self.get_right_position()

        start_time = time.time()
        steps = int(duration * 50)  # 50Hz控制频率

        print(f"开始同时平滑移动双臂")
        print(f"左臂: {left_current} -> {left_target}")
        print(f"右臂: {right_current} -> {right_target}")

        for i in range(steps):
            elapsed = time.time() - start_time
            progress = min(elapsed / duration, 1.0)

            # 使用三次贝塞尔曲线进行平滑插值
            t = progress
            smooth_progress = t * t * (3.0 - 2.0 * t)  # smoothstep函数

            # 计算当前目标位置
            left_current_targets = []
            right_current_targets = []

            for j in range(len(left_target)):
                left_target_val = left_current[j] + (left_target[j] - left_current[j]) * smooth_progress
                right_target_val = right_current[j] + (right_target[j] - right_current[j]) * smooth_progress
                left_current_targets.append(left_target_val)
                right_current_targets.append(right_target_val)

            # 同时控制双臂
            if self.enable_left:
                left_mit_params = []
                for j in range(len(left_current_targets)):
                    left_mit_params.append(oa.MITParam(5.0, 1.0, left_current_targets[j], 0, 0))
                self.left_arm.get_arm().mit_control_all(left_mit_params)

            if self.enable_right:
                right_mit_params = []
                for j in range(len(right_current_targets)):
                    right_mit_params.append(oa.MITParam(5.0, 1.0, right_current_targets[j], 0, 0))
                self.right_arm.get_arm().mit_control_all(right_mit_params)

            # 接收反馈
            if self.enable_left:
                self.left_arm.recv_all()
            if self.enable_right:
                self.right_arm.recv_all()

            time.sleep(0.02)  # 50Hz控制频率

        print("双臂同时平滑移动完成")

    def get_position(self, arm):
        # max 500Hz

        arm.refresh_all()
        arm.recv_all()

        arm_positions = []
        gripper_positions = []
        for i, motor in enumerate(arm.get_arm().get_motors()):
            arm_positions.append(motor.get_position())
        for motor in arm.get_gripper().get_motors():
            gripper_positions.append(motor.get_position())

        return arm_positions, gripper_positions

    def set_position(self, arm, arm_target_positions, gripper_target_position,
                    current_arm_positions, current_gripper_position):
        # positions: list of target positions for each motor
        # MITParam: kp, kd, q, dq, tau
        error = np.array(arm_target_positions) - np.array(current_arm_positions)

        error = np.clip(error, -0.3, 0.3)

        if arm == self.left_arm:
            # 微分项
            derror = error - self.left_last_error
            self.left_last_error = error
            # 积分项
            self.left_acc_error += error
            self.left_acc_error = np.clip(self.left_acc_error, -self.acc_error_limit, self.acc_error_limit)
            ierror = self.left_acc_error

        elif arm == self.right_arm:
            # 微分项
            derror = error - self.right_last_error
            self.right_last_error = error
            # 积分项
            self.right_acc_error += error
            self.right_acc_error = np.clip(self.right_acc_error, -self.acc_error_limit, self.acc_error_limit)
            ierror = self.right_acc_error

        # self.acc_error += 0.01 * error
        # np.clip(self.acc_error, -0.03, 0.03)
        # print("acc error: ", self.acc_error)

        mit_params = []
        for i in range(len(arm_target_positions)):
            # pos = arm_target_positions[i] + self.acc_error[i]
            # vel_cmd = self.kp[i] * error[i] + self.kv[i] * derror[i] + self.ki[i] * ierror[i]
            mit_params.append(oa.MITParam(self.kp[i], self.kv[i] , arm_target_positions[i], derror[i], ierror[i]))

        # arm
        arm.get_arm().mit_control_all(mit_params)
        # gripper
        arm.get_gripper().mit_control_all([oa.MITParam(2, 0, gripper_target_position, 0, 0)])
        arm.recv_all()

    # left arm
    def get_left_position(self):
        return self.get_position(self.left_arm)

    def set_left_position(self, left_arm_position, left_gripper_position,
                                current_arm_position, current_gripper_position):

        self.set_position(self.left_arm, left_arm_position, left_gripper_position,
                        current_arm_position, current_gripper_position)
    # right arm
    def get_right_position(self):
        return self.get_position(self.right_arm)

    def set_right_position(self, right_arm_position, right_gripper_position,
                                current_arm_position, current_gripper_position):

        self.set_position(self.right_arm, right_arm_position, right_gripper_position,
                        current_arm_position, current_gripper_position)


    def test_run(self):

        left_arm_position = [-1, 0, 0, 0, 0, 0, -1]
        left_gripper_position = -0.3

        # read motor position
        while True:
            if self.enable_left:
                current_arm_position,  current_gripper_position = self.get_position(self.left_arm)
                print("current position: ", current_arm_position)
                # self.set_position(self.left_arm, left_arm_position, left_gripper_position,
                #                        current_arm_position, current_gripper_position)


if __name__ == "__main__":
    controller = OpenArmController(enable_left=True, enable_right=True)
    controller.test_run()