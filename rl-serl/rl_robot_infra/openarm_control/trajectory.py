"""Trajectory limiting and latest-target tracking for OpenArm servo control."""

import time
from dataclasses import dataclass
from threading import Lock
from typing import Optional

import numpy as np

from openarm_control.types import CommandTarget, MITCommand


@dataclass
class ServoLimits:
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray
    torque: np.ndarray

    @classmethod
    def conservative(cls, dof: int = 14):
        return cls(
            velocity=np.ones(dof) * 1.2,
            acceleration=np.ones(dof) * 4.0,
            jerk=np.ones(dof) * 25.0,
            torque=np.ones(dof) * 6.0,
        )


@dataclass
class ServoGains:
    kp: np.ndarray
    kd: np.ndarray

    @classmethod
    def default(cls, dof: int = 14):
        kp7 = np.array([60, 50, 50, 50, 30, 30, 30], dtype=float)
        kd7 = np.array([2, 2, 2, 2, 0.5, 0.5, 0.5], dtype=float) * 3.0
        if dof == 7:
            return cls(kp=kp7, kd=kd7)
        return cls(kp=np.concatenate([kp7, kp7]), kd=np.concatenate([kd7, kd7]))


class LatestTargetBuffer:
    """Thread-safe latest-only target buffer.

    This avoids stale target queues. If env/VR publishes faster than the servo
    loop, the loop consumes the newest target and drops older ones.
    """

    def __init__(self):
        self._lock = Lock()
        self._target: Optional[CommandTarget] = None

    def update(self, target: CommandTarget):
        with self._lock:
            self._target = target

    def get(self) -> Optional[CommandTarget]:
        with self._lock:
            return self._target

    def clear(self):
        with self._lock:
            self._target = None


class JointTrajectoryLimiter:
    """Generate MIT joint commands with velocity/acceleration/jerk limits."""

    def __init__(self, limits: ServoLimits, gains: ServoGains):
        self.limits = limits
        self.gains = gains
        dof = len(limits.velocity)
        self.last_q_cmd = np.zeros(dof)
        self.last_dq_cmd = np.zeros(dof)
        self.last_accel_cmd = np.zeros(dof)
        self.last_feedback_dq = np.zeros(dof)
        self.last_feedback_ts = 0.0
        self.joint_accel_est = np.zeros(dof)
        self.initialized = False

    def reset(self, q_current, dq_current=None):
        q_current = np.asarray(q_current, dtype=float)
        self.last_q_cmd = q_current.copy()
        self.last_dq_cmd = np.zeros_like(q_current) if dq_current is None else np.asarray(dq_current, dtype=float)
        self.last_accel_cmd = np.zeros_like(q_current)
        self.last_feedback_dq = self.last_dq_cmd.copy()
        self.last_feedback_ts = time.time()
        self.joint_accel_est = np.zeros_like(q_current)
        self.initialized = True

    def estimate_feedback_accel(self, dq_feedback, timestamp):
        dq_feedback = np.asarray(dq_feedback, dtype=float)
        if self.last_feedback_ts > 0:
            dt = max(float(timestamp - self.last_feedback_ts), 1e-4)
            raw = (dq_feedback - self.last_feedback_dq) / dt
            self.joint_accel_est = 0.85 * self.joint_accel_est + 0.15 * raw
        self.last_feedback_dq = dq_feedback.copy()
        self.last_feedback_ts = float(timestamp)
        return self.joint_accel_est.copy()

    def step(self, q_target, q_feedback, dq_feedback, dt, speed_scale=1.0, active_mask=None):
        q_target = np.asarray(q_target, dtype=float)
        q_feedback = np.asarray(q_feedback, dtype=float)
        dq_feedback = np.asarray(dq_feedback, dtype=float)
        dt = max(float(dt), 1e-4)
        if not self.initialized:
            self.reset(q_feedback, dq_feedback)

        if active_mask is None:
            active_mask = np.ones_like(q_target, dtype=bool)
        else:
            active_mask = np.asarray(active_mask, dtype=bool)

        velocity_limit = self.limits.velocity * float(speed_scale)
        acceleration_limit = self.limits.acceleration * float(speed_scale)
        jerk_limit = self.limits.jerk * float(speed_scale)

        desired_delta = q_target - self.last_q_cmd
        desired_dq = np.clip(desired_delta / dt, -velocity_limit, velocity_limit)

        accel = np.clip(
            (desired_dq - self.last_dq_cmd) / dt,
            self.last_accel_cmd - jerk_limit * dt,
            self.last_accel_cmd + jerk_limit * dt,
        )
        accel = np.clip(accel, -acceleration_limit, acceleration_limit)

        dq_cmd = self.last_dq_cmd + accel * dt
        dq_cmd = np.clip(dq_cmd, -velocity_limit, velocity_limit)
        q_cmd = self.last_q_cmd + dq_cmd * dt

        # Prevent overshoot on small remaining errors.
        for i in range(q_target.shape[0]):
            if not active_mask[i]:
                q_cmd[i] = q_feedback[i]
                dq_cmd[i] = 0.0
                accel[i] = 0.0
                continue
            if desired_delta[i] == 0:
                q_cmd[i] = q_target[i]
                dq_cmd[i] = 0.0
            elif np.sign(q_target[i] - self.last_q_cmd[i]) != np.sign(q_target[i] - q_cmd[i]):
                q_cmd[i] = q_target[i]
                dq_cmd[i] = 0.0

        self.last_q_cmd = q_cmd.copy()
        self.last_dq_cmd = dq_cmd.copy()
        self.last_accel_cmd = accel.copy()

        return MITCommand(
            q=q_cmd,
            dq=dq_cmd,
            tau_ff=np.zeros_like(q_cmd),
            kp=self.gains.kp.copy(),
            kd=self.gains.kd.copy(),
        )
