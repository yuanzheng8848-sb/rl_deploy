import time

import numpy as np

from openarm_control.safety import SafetyConfig, SafetyMonitor
from openarm_control.trajectory import JointTrajectoryLimiter, ServoGains, ServoLimits
from openarm_control.types import RobotState, ArmState


def _arm_state(enabled=True, tau=0.0, temp=30.0):
    return ArmState(
        q=np.zeros(7),
        dq=np.zeros(7),
        tau=np.ones(7) * tau,
        t_mos=np.ones(7) * temp,
        t_rotor=np.ones(7) * temp,
        enabled=np.ones(7, dtype=bool) * enabled,
        gripper_q=np.zeros(1),
        gripper_dq=np.zeros(1),
        gripper_tau=np.zeros(1),
        gripper_t_mos=np.ones(1) * temp,
        gripper_t_rotor=np.ones(1) * temp,
        gripper_enabled=np.ones(1, dtype=bool) * enabled,
        timestamp=time.time(),
    )


def test_trajectory_limiter_respects_velocity_limit():
    limits = ServoLimits(
        velocity=np.ones(14) * 0.5,
        acceleration=np.ones(14) * 5.0,
        jerk=np.ones(14) * 50.0,
        torque=np.ones(14) * 5.0,
    )
    limiter = JointTrajectoryLimiter(limits, ServoGains.default(14))
    limiter.reset(np.zeros(14), np.zeros(14))
    cmd = limiter.step(np.ones(14) * 10.0, np.zeros(14), np.zeros(14), dt=0.1)
    assert np.all(np.abs(cmd.dq) <= 0.5 + 1e-9)
    assert np.all(np.abs(cmd.q) <= 0.05 + 1e-9)
    assert np.allclose(cmd.tau_ff, 0.0)


def test_safety_faults_on_disabled_motor():
    state = RobotState(left=_arm_state(enabled=False), right=_arm_state(), timestamp=time.time())
    monitor = SafetyMonitor(SafetyConfig(), effort_limits=np.ones(14) * 5.0, velocity_limits=np.ones(14))
    status = monitor.evaluate(state, time.time())
    assert status.mode == "FAULT_STOP"
    assert not status.ok_to_send
    assert "MOTOR_DISABLED" in status.reasons


def test_safety_slowdown_on_torque_warning():
    state = RobotState(left=_arm_state(tau=4.0), right=_arm_state(), timestamp=time.time())
    monitor = SafetyMonitor(SafetyConfig(), effort_limits=np.ones(14) * 5.0, velocity_limits=np.ones(14))
    status = monitor.evaluate(state, time.time())
    assert status.mode == "SLOWDOWN"
    assert status.ok_to_send
    assert status.speed_scale < 1.0
