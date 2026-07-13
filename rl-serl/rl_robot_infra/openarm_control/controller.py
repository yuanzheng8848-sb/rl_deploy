"""High-level OpenArm controller composed from hardware, trajectory, and safety."""

import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import yaml

from openarm_control.safety import SafetyConfig, SafetyMonitor
from openarm_control.trajectory import LatestTargetBuffer, ServoGains, ServoLimits, JointTrajectoryLimiter
from openarm_control.types import CommandTarget, MITCommand, RobotState


DEFAULT_CAN_CONFIG = {
    "left": {"interface": "can0", "can_fd": False},
    "right": {"interface": "can1", "can_fd": False},
    "motors": {
        "arm_send_ids": [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07],
        "arm_recv_ids": [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17],
        "gripper_send_id": 0x08,
        "gripper_recv_id": 0x18,
    },
}


def _infra_root():
    return Path(__file__).resolve().parents[1]


def _default_can_config_path():
    return _infra_root() / "openarm_configs" / "can.yaml"


def _default_joint_limits_path():
    return _infra_root() / "openarm_description" / "config" / "arm" / "v10" / "joint_limits.yaml"


def _load_can_config(path=None):
    cfg = {
        "left": DEFAULT_CAN_CONFIG["left"].copy(),
        "right": DEFAULT_CAN_CONFIG["right"].copy(),
        "motors": DEFAULT_CAN_CONFIG["motors"].copy(),
    }
    config_path = Path(path) if path is not None else _default_can_config_path()
    if config_path.exists():
        with open(config_path) as f:
            loaded = yaml.safe_load(f) or {}
        for section in ("left", "right", "motors"):
            cfg[section].update(loaded.get(section, {}) or {})
    else:
        print(f"[OpenArmController] CAN config not found, using defaults: {config_path}")
    return cfg


def _load_joint_limits(path=None):
    config_path = Path(path) if path is not None else _default_joint_limits_path()
    velocity = np.array([1.2] * 7, dtype=float)
    effort = np.array([6.0] * 7, dtype=float)
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        velocity_values = []
        effort_values = []
        for i in range(1, 8):
            limits = (cfg.get(f"joint{i}", {}) or {}).get("limit", {}) or {}
            velocity_values.append(float(limits.get("velocity", velocity[i - 1])))
            effort_values.append(float(limits.get("effort", effort[i - 1])))
        velocity = np.asarray(velocity_values, dtype=float)
        effort = np.asarray(effort_values, dtype=float)

    # Conservative runtime limits. The URDF limits are hardware maxima; starting
    # with a fraction produces smoother training behavior.
    runtime_velocity = np.minimum(velocity, np.array([1.2, 1.2, 1.0, 1.0, 1.5, 1.5, 1.5]))
    runtime_effort = effort
    return np.concatenate([runtime_velocity, runtime_velocity]), np.concatenate([runtime_effort, runtime_effort])


def _split_command(command: MITCommand, start: int, stop: int) -> MITCommand:
    return MITCommand(
        q=command.q[start:stop],
        dq=command.dq[start:stop],
        tau_ff=command.tau_ff[start:stop],
        kp=command.kp[start:stop],
        kd=command.kd[start:stop],
    )


class OpenArmController:
    """High-level bimanual controller with state and joint-command APIs."""

    def __init__(
        self,
        enable_left=True,
        enable_right=True,
        can_config_path=None,
        control_config_path=None,
        bus_factory: Optional[Callable] = None,
    ):
        self.enable_left = bool(enable_left)
        self.enable_right = bool(enable_right)
        self.can_config = _load_can_config(can_config_path)
        self.left_arm = None
        self.right_arm = None
        if bus_factory is None:
            # Keep the native extension out of simulation and unit-test processes.
            from openarm_control.hardware import OpenArmCanBus

            bus_factory = OpenArmCanBus
        if self.enable_left:
            cfg = self.can_config["left"]
            self.left_arm = bus_factory(cfg["interface"], bool(cfg.get("can_fd", False)), self.can_config["motors"])
        if self.enable_right:
            cfg = self.can_config["right"]
            self.right_arm = bus_factory(cfg["interface"], bool(cfg.get("can_fd", False)), self.can_config["motors"])

        velocity_limits, effort_limits = _load_joint_limits()
        limits = ServoLimits(
            velocity=velocity_limits,
            acceleration=np.maximum(velocity_limits * 3.0, 1.0),
            jerk=np.maximum(velocity_limits * 15.0, 5.0),
            torque=effort_limits,
        )
        self.trajectory = JointTrajectoryLimiter(limits, ServoGains.default(14))
        self.target_buffer = LatestTargetBuffer()
        self.safety = SafetyMonitor(SafetyConfig(), effort_limits=effort_limits, velocity_limits=velocity_limits)
        self.last_state = RobotState.zeros()
        self.last_command_time = 0.0
        self.debug = {
            "last_target_source": "",
            "last_command_q": None,
            "last_command_dq": None,
            "last_safety": self.safety.last_status.to_dict(),
        }
        state = self.read_state()
        if state.timestamp > 0:
            self.trajectory.reset(state.q, state.dq)

    def read_state(self) -> RobotState:
        now = time.time()
        left = self.left_arm.read_state() if self.left_arm is not None else self.last_state.left
        right = self.right_arm.read_state() if self.right_arm is not None else self.last_state.right
        state = RobotState(left=left, right=right, timestamp=max(left.timestamp, right.timestamp))
        state.joint_accel_est = self.trajectory.estimate_feedback_accel(state.dq, state.timestamp or now)
        self.last_state = state
        return state

    def set_target(self, q=None, gripper=None, active_arms=(0, 1), source="unknown"):
        target = CommandTarget(
            q=None if q is None else np.asarray(q, dtype=float),
            gripper=None if gripper is None else np.asarray(gripper, dtype=float),
            active_arms=tuple(active_arms),
            timestamp=time.time(),
            source=str(source),
        )
        self.target_buffer.update(target)
        return target

    def consume_latest_target(self) -> Optional[CommandTarget]:
        return self.target_buffer.get()

    def command_joint_target(
        self,
        q_target,
        gripper_target=None,
        active_arms=(0, 1),
        dt=0.0125,
        source="direct",
    ) -> Tuple[bool, Dict]:
        state = self.read_state()
        now = time.time()
        safety_status = self.safety.evaluate(state, now)
        if not safety_status.ok_to_send:
            self.hold_position()
            self.debug["last_safety"] = safety_status.to_dict()
            return False, {"safety": safety_status.to_dict()}

        q_target = np.asarray(q_target, dtype=float).reshape(14)
        active_mask = np.zeros(14, dtype=bool)
        if 0 in active_arms:
            active_mask[:7] = True
        if 1 in active_arms:
            active_mask[7:] = True

        command = self.trajectory.step(
            q_target=q_target,
            q_feedback=state.q,
            dq_feedback=state.dq,
            dt=dt,
            speed_scale=safety_status.speed_scale,
            active_mask=active_mask,
        )
        if gripper_target is None:
            gripper_target = state.gripper_q
        gripper_target = np.asarray(gripper_target, dtype=float).reshape(-1)
        if gripper_target.size < 2:
            gripper_target = np.array([state.left.gripper_q[0], state.right.gripper_q[0]], dtype=float)

        if self.left_arm is not None and 0 in active_arms:
            self.left_arm.send_mit(_split_command(command, 0, 7), gripper_position=float(gripper_target[0]))
        if self.right_arm is not None and 1 in active_arms:
            self.right_arm.send_mit(_split_command(command, 7, 14), gripper_position=float(gripper_target[1]))

        self.last_command_time = now
        self.debug.update(
            {
                "last_target_source": source,
                "last_command_q": command.q.tolist(),
                "last_command_dq": command.dq.tolist(),
                "last_safety": safety_status.to_dict(),
            }
        )
        return True, {"safety": safety_status.to_dict(), "q_cmd": command.q.tolist(), "dq_cmd": command.dq.tolist()}

    def hold_position(self):
        if self.left_arm is not None:
            self.left_arm.hold_position()
        if self.right_arm is not None:
            self.right_arm.hold_position()

    def query_motor_params(self):
        try:
            import openarm_can as oa

            rids = [
                oa.MotorVariable.ACC,
                oa.MotorVariable.DEC,
                oa.MotorVariable.MAX_SPD,
                oa.MotorVariable.PMAX,
                oa.MotorVariable.VMAX,
                oa.MotorVariable.TMAX,
            ]
        except Exception:
            rids = []
        return {
            "left": {} if self.left_arm is None else self.left_arm.query_params(rids),
            "right": {} if self.right_arm is None else self.right_arm.query_params(rids),
        }

    def diagnostics(self):
        return {
            "state": self.last_state.to_dict(),
            "safety": self.safety.last_status.to_dict(),
            "controller": dict(self.debug),
        }
