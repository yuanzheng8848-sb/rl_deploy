"""Typed data containers for OpenArm control.

These classes intentionally stay lightweight and serializable. The hardware
thread, Flask server, and env code pass these objects across boundaries without
depending on openarm_can internals.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


def _list(value):
    return np.asarray(value, dtype=float).reshape(-1).tolist()


@dataclass
class ArmState:
    q: np.ndarray
    dq: np.ndarray
    tau: np.ndarray
    t_mos: np.ndarray
    t_rotor: np.ndarray
    enabled: np.ndarray
    gripper_q: np.ndarray
    gripper_dq: np.ndarray
    gripper_tau: np.ndarray
    gripper_t_mos: np.ndarray
    gripper_t_rotor: np.ndarray
    gripper_enabled: np.ndarray
    timestamp: float

    @classmethod
    def zeros(cls, dof: int = 7):
        return cls(
            q=np.zeros(dof),
            dq=np.zeros(dof),
            tau=np.zeros(dof),
            t_mos=np.zeros(dof),
            t_rotor=np.zeros(dof),
            enabled=np.zeros(dof, dtype=bool),
            gripper_q=np.zeros(1),
            gripper_dq=np.zeros(1),
            gripper_tau=np.zeros(1),
            gripper_t_mos=np.zeros(1),
            gripper_t_rotor=np.zeros(1),
            gripper_enabled=np.zeros(1, dtype=bool),
            timestamp=0.0,
        )

    def to_dict(self) -> Dict:
        return {
            "q": _list(self.q),
            "dq": _list(self.dq),
            "tau": _list(self.tau),
            "t_mos": _list(self.t_mos),
            "t_rotor": _list(self.t_rotor),
            "enabled": np.asarray(self.enabled, dtype=bool).reshape(-1).tolist(),
            "gripper_q": _list(self.gripper_q),
            "gripper_dq": _list(self.gripper_dq),
            "gripper_tau": _list(self.gripper_tau),
            "gripper_t_mos": _list(self.gripper_t_mos),
            "gripper_t_rotor": _list(self.gripper_t_rotor),
            "gripper_enabled": np.asarray(self.gripper_enabled, dtype=bool).reshape(-1).tolist(),
            "timestamp": float(self.timestamp),
        }


@dataclass
class RobotState:
    left: ArmState
    right: ArmState
    timestamp: float
    joint_accel_est: np.ndarray = field(default_factory=lambda: np.zeros(14))

    @classmethod
    def zeros(cls):
        return cls(left=ArmState.zeros(), right=ArmState.zeros(), timestamp=0.0)

    @property
    def q(self):
        return np.concatenate([self.left.q, self.right.q])

    @property
    def dq(self):
        return np.concatenate([self.left.dq, self.right.dq])

    @property
    def tau(self):
        return np.concatenate([self.left.tau, self.right.tau])

    @property
    def gripper_q(self):
        return np.array([self.left.gripper_q[0], self.right.gripper_q[0]], dtype=float)

    def to_dict(self) -> Dict:
        return {
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "q": _list(self.q),
            "dq": _list(self.dq),
            "joint_torque": _list(self.tau),
            "joint_accel_est": _list(self.joint_accel_est),
            "gripper_pos": _list(self.gripper_q),
            "motor_temperature": {
                "t_mos": _list(np.concatenate([self.left.t_mos, self.right.t_mos])),
                "t_rotor": _list(np.concatenate([self.left.t_rotor, self.right.t_rotor])),
            },
            "motor_enabled": np.concatenate([self.left.enabled, self.right.enabled]).astype(bool).tolist(),
            "timestamp": float(self.timestamp),
        }


@dataclass
class MITCommand:
    q: np.ndarray
    dq: np.ndarray
    tau_ff: np.ndarray
    kp: np.ndarray
    kd: np.ndarray


@dataclass
class SafetyStatus:
    mode: str = "NORMAL"
    ok_to_send: bool = True
    speed_scale: float = 1.0
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "mode": self.mode,
            "ok_to_send": bool(self.ok_to_send),
            "speed_scale": float(self.speed_scale),
            "reasons": list(self.reasons),
        }


@dataclass
class CommandTarget:
    q: Optional[np.ndarray] = None
    gripper: Optional[np.ndarray] = None
    active_arms: tuple = (0, 1)
    timestamp: float = 0.0
    source: str = "unknown"
