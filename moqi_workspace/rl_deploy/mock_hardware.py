
import numpy as np
import time

class MockOpenArmController:
    """
    Mock controller that simulates the interface of OpenArmController.
    Used for testing logic without physical hardware.
    """
    def __init__(self, enable_left=True, enable_right=True):
        print("[Mock] Initializing Mock OpenArm Controller...")
        self.enable_left = enable_left
        self.enable_right = enable_right
        
        # Initialize at home position (all zeros)
        self.left_q = np.zeros(7)
        self.right_q = np.zeros(7)
        
        self.left_gripper = 0.0
        self.right_gripper = 0.0
        
        # Simulating hardware delay
        self.left_arm = "MockLeftArmHandle"
        self.right_arm = "MockRightArmHandle"

    def get_left_position(self):
        # Returns: joint_positions (list/array), gripper_positions (list)
        # Adding slight noise to simulate sensor noise
        noise = 0
        return self.left_q + noise, [self.left_gripper]

    def get_right_position(self):
        noise = 0
        return self.right_q + noise, [self.right_gripper]

    def set_left_position(self, target_joints, target_gripper, current_joints, current_gripper):
        # In simulation, we just instantly update the "state" to the target
        # In a more complex mock, we could interpolate over time
        self.left_q = np.array(target_joints)
        self.left_gripper = target_gripper

    def set_right_position(self, target_joints, target_gripper, current_joints, current_gripper):
        self.right_q = np.array(target_joints)
        self.right_gripper = target_gripper

    def _smooth_move_to_position(self, arm, start_positions, target_positions, duration=2.0):
        print(f"[Mock] Smooth moving arm from {start_positions[:3]}... to {target_positions[:3]}...")
        time.sleep(0.5) # Simulate some time passing
        if arm == self.left_arm:
            self.left_q = np.array(target_positions)
        else:
            self.right_q = np.array(target_positions)
        print("[Mock] Move complete.")