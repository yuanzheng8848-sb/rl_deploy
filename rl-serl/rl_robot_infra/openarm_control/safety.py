"""Safety monitoring for OpenArm servo control."""

from dataclasses import dataclass

import numpy as np

from openarm_control.types import RobotState, SafetyStatus


@dataclass
class SafetyConfig:
    torque_warning_ratio: float = 0.75
    torque_stop_ratio: float = 1.05
    t_mos_warning: float = 70.0
    t_mos_stop: float = 85.0
    t_rotor_warning: float = 70.0
    t_rotor_stop: float = 85.0
    can_timeout_s: float = 0.2
    velocity_stop_ratio: float = 1.25


class SafetyMonitor:
    def __init__(self, config: SafetyConfig, effort_limits, velocity_limits):
        self.config = config
        self.effort_limits = np.asarray(effort_limits, dtype=float)
        self.velocity_limits = np.asarray(velocity_limits, dtype=float)
        self.last_status = SafetyStatus()

    def evaluate(self, state: RobotState, now: float) -> SafetyStatus:
        reasons = []
        mode = "NORMAL"
        ok_to_send = True
        speed_scale = 1.0

        if state.timestamp <= 0 or now - state.timestamp > self.config.can_timeout_s:
            reasons.append("CAN_TIMEOUT")
            mode = "FAULT_STOP"
            ok_to_send = False

        enabled = np.asarray(state.to_dict()["motor_enabled"], dtype=bool)
        if enabled.size and not np.all(enabled):
            reasons.append("MOTOR_DISABLED")
            mode = "FAULT_STOP"
            ok_to_send = False

        tau_abs = np.abs(state.tau)
        effort_stop = self.effort_limits * self.config.torque_stop_ratio
        effort_warn = self.effort_limits * self.config.torque_warning_ratio
        if np.any(tau_abs > effort_stop):
            reasons.append("TORQUE_STOP")
            mode = "FAULT_STOP"
            ok_to_send = False
        elif np.any(tau_abs > effort_warn) and mode == "NORMAL":
            reasons.append("TORQUE_WARNING")
            mode = "SLOWDOWN"
            speed_scale = min(speed_scale, 0.4)

        dq_abs = np.abs(state.dq)
        if np.any(dq_abs > self.velocity_limits * self.config.velocity_stop_ratio):
            reasons.append("VELOCITY_STOP")
            mode = "FAULT_STOP"
            ok_to_send = False

        t_mos = np.concatenate([state.left.t_mos, state.right.t_mos])
        t_rotor = np.concatenate([state.left.t_rotor, state.right.t_rotor])
        if np.any(t_mos > self.config.t_mos_stop) or np.any(t_rotor > self.config.t_rotor_stop):
            reasons.append("TEMPERATURE_STOP")
            mode = "FAULT_STOP"
            ok_to_send = False
        elif (
            np.any(t_mos > self.config.t_mos_warning)
            or np.any(t_rotor > self.config.t_rotor_warning)
        ) and mode == "NORMAL":
            reasons.append("TEMPERATURE_WARNING")
            mode = "SLOWDOWN"
            speed_scale = min(speed_scale, 0.5)

        status = SafetyStatus(mode=mode, ok_to_send=ok_to_send, speed_scale=speed_scale, reasons=reasons)
        self.last_status = status
        return status
