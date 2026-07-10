"""Low-level OpenArm CAN hardware adapter."""

import time
from typing import Dict, Iterable, Optional

import numpy as np

import openarm_can as oa

from openarm_control.types import ArmState, MITCommand


MOTOR_TYPES = [
    oa.MotorType.DM8009,
    oa.MotorType.DM8009,
    oa.MotorType.DM4340,
    oa.MotorType.DM4340,
    oa.MotorType.DM4310,
    oa.MotorType.DM4310,
    oa.MotorType.DM4310,
]


class OpenArmCanBus:
    """One physical OpenArm side on one CAN interface."""

    def __init__(self, interface: str, can_fd: bool, motor_config: Dict):
        self.interface = str(interface)
        self.can_fd = bool(can_fd)
        self.motor_config = motor_config
        self.arm = oa.OpenArm(self.interface, self.can_fd)
        self.arm.init_arm_motors(
            MOTOR_TYPES,
            motor_config["arm_send_ids"],
            motor_config["arm_recv_ids"],
        )
        self.arm.init_gripper_motor(
            oa.MotorType.DM4310,
            motor_config["gripper_send_id"],
            motor_config["gripper_recv_id"],
        )
        self.arm.set_callback_mode_all(oa.CallbackMode.IGNORE)
        self.arm.enable_all()
        self.arm.recv_all()
        self.arm.set_callback_mode_all(oa.CallbackMode.STATE)
        self.last_state = ArmState.zeros()

    def read_state(self, timeout_us: int = 500) -> ArmState:
        self.arm.refresh_all()
        self.arm.recv_all(timeout_us)
        now = time.time()
        arm_motors = list(self.arm.get_arm().get_motors())
        gripper_motors = list(self.arm.get_gripper().get_motors())
        state = ArmState(
            q=np.array([m.get_position() for m in arm_motors], dtype=float),
            dq=np.array([m.get_velocity() for m in arm_motors], dtype=float),
            tau=np.array([m.get_torque() for m in arm_motors], dtype=float),
            t_mos=np.array([m.get_state_tmos() for m in arm_motors], dtype=float),
            t_rotor=np.array([m.get_state_trotor() for m in arm_motors], dtype=float),
            enabled=np.array([m.is_enabled() for m in arm_motors], dtype=bool),
            gripper_q=np.array([m.get_position() for m in gripper_motors], dtype=float),
            gripper_dq=np.array([m.get_velocity() for m in gripper_motors], dtype=float),
            gripper_tau=np.array([m.get_torque() for m in gripper_motors], dtype=float),
            gripper_t_mos=np.array([m.get_state_tmos() for m in gripper_motors], dtype=float),
            gripper_t_rotor=np.array([m.get_state_trotor() for m in gripper_motors], dtype=float),
            gripper_enabled=np.array([m.is_enabled() for m in gripper_motors], dtype=bool),
            timestamp=now,
        )
        self.last_state = state
        return state

    def send_mit(self, command: MITCommand, gripper_position: Optional[float] = None):
        params = []
        for i in range(len(command.q)):
            params.append(
                oa.MITParam(
                    float(command.kp[i]),
                    float(command.kd[i]),
                    float(command.q[i]),
                    float(command.dq[i]),
                    float(command.tau_ff[i]),
                )
            )
        self.arm.get_arm().mit_control_all(params)
        if gripper_position is not None:
            self.arm.get_gripper().mit_control_all([oa.MITParam(2.0, 0.0, float(gripper_position), 0.0, 0.0)])
        self.arm.recv_all()

    def hold_position(self):
        state = self.last_state if self.last_state.timestamp > 0 else self.read_state()
        zeros = np.zeros_like(state.q)
        cmd = MITCommand(
            q=state.q.copy(),
            dq=zeros.copy(),
            tau_ff=zeros.copy(),
            kp=np.ones_like(state.q) * 5.0,
            kd=np.ones_like(state.q),
        )
        self.send_mit(cmd, gripper_position=float(state.gripper_q[0]) if state.gripper_q.size else None)

    def query_params(self, rids: Iterable) -> Dict[str, list]:
        result = {}
        motors = list(self.arm.get_arm().get_motors()) + list(self.arm.get_gripper().get_motors())
        for rid in rids:
            self.arm.query_param_all(int(rid))
            self.arm.recv_all()
            result[str(rid)] = [float(m.get_param(int(rid))) for m in motors]
        return result
