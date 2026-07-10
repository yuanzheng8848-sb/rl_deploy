
import numpy as np
import time

from openarm_control.types import ArmState, RobotState

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
        self.left_dq = np.zeros(7)
        self.right_dq = np.zeros(7)
        self.last_ts = time.time()
        self.last_state = RobotState.zeros()
        
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

    def _arm_state(self, q, dq, gripper):
        now = time.time()
        return ArmState(
            q=np.asarray(q, dtype=float).copy(),
            dq=np.asarray(dq, dtype=float).copy(),
            tau=np.zeros(7),
            t_mos=np.ones(7) * 30.0,
            t_rotor=np.ones(7) * 30.0,
            enabled=np.ones(7, dtype=bool),
            gripper_q=np.array([float(gripper)]),
            gripper_dq=np.zeros(1),
            gripper_tau=np.zeros(1),
            gripper_t_mos=np.ones(1) * 30.0,
            gripper_t_rotor=np.ones(1) * 30.0,
            gripper_enabled=np.ones(1, dtype=bool),
            timestamp=now,
        )

    def read_state(self):
        now = time.time()
        state = RobotState(
            left=self._arm_state(self.left_q, self.left_dq, self.left_gripper),
            right=self._arm_state(self.right_q, self.right_dq, self.right_gripper),
            timestamp=now,
        )
        state.joint_accel_est = np.zeros(14)
        self.last_state = state
        return state

    def command_joint_target(self, q_target, gripper_target=None, active_arms=(0, 1), dt=0.0125, source="mock"):
        q_target = np.asarray(q_target, dtype=float).reshape(14)
        dt = max(float(dt), 1e-4)
        if gripper_target is None:
            gripper_target = [self.left_gripper, self.right_gripper]
        if 0 in active_arms:
            self.left_dq = (q_target[:7] - self.left_q) / dt
            self.left_q = q_target[:7].copy()
            self.left_gripper = float(gripper_target[0])
        if 1 in active_arms:
            self.right_dq = (q_target[7:] - self.right_q) / dt
            self.right_q = q_target[7:].copy()
            self.right_gripper = float(gripper_target[1])
        return True, {"safety": {"mode": "NORMAL", "ok_to_send": True, "speed_scale": 1.0, "reasons": []}}

    def set_target(self, q=None, gripper=None, active_arms=(0, 1), source="mock"):
        self._target = (q, gripper, active_arms, source)
        return self._target

    def consume_latest_target(self):
        return None

    def hold_position(self):
        pass

    def diagnostics(self):
        return {
            "state": self.read_state().to_dict(),
            "safety": {"mode": "NORMAL", "ok_to_send": True, "speed_scale": 1.0, "reasons": []},
            "controller": {"mock": True},
        }

    def query_motor_params(self):
        return {"left": {}, "right": {}}

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

class MockCamera:
    """
    Mock Camera that returns black (zero) images.
    """
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        print(f"[MockCamera] Initialized {width}x{height}@{fps}fps (Black Image Generator)")

    def get_data(self, viz=False):
        # Return black image (all zeros)
        # RealsenseCamera usually returns [color, depth] (list)
        # USB camera usually returns frame (ndarray)
        # We will return [color, None] to mimic Realsense structure mostly, 
        # or just color if caller handles it.
        # LocalOpenArmEnv handles list or ndarray.
        # Let's return a list [color, None] to be safe for Realsense logic.
        
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return [img, None]
