import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "rl_robot_infra"), str(ROOT)]

from openarm_control.controller import OpenArmController
from openarm_env.mock_hardware import SimulatedArmBus


def test_simulated_bus_exercises_controller_without_openarm_can():
    controller = OpenArmController(bus_factory=SimulatedArmBus)
    target = np.r_[np.ones(7) * 0.1, np.ones(7) * -0.1]
    for _ in range(20):
        ok, info = controller.command_joint_target(target, [0.02, -0.03], dt=0.01, source="pytest")
        assert ok, info

    state = controller.read_state()
    assert controller.left_arm.command_count == 20
    assert controller.right_arm.command_count == 20
    assert np.all(state.q[:7] > 0)
    assert np.all(state.q[7:] < 0)
    np.testing.assert_allclose(state.gripper_q, [0.02, -0.03])


def test_inactive_arm_is_not_commanded():
    controller = OpenArmController(bus_factory=SimulatedArmBus)
    ok, info = controller.command_joint_target(np.ones(14) * 0.1, active_arms=(0,), dt=0.01)
    assert ok, info
    assert controller.left_arm.command_count == 1
    assert controller.right_arm.command_count == 0


def test_simulated_fault_is_blocked_before_command_send():
    controller = OpenArmController(bus_factory=SimulatedArmBus)
    controller.left_arm.temperature[0] = 90.0
    ok, info = controller.command_joint_target(np.ones(14) * 0.1, dt=0.01)
    assert not ok
    assert "TEMPERATURE_STOP" in info["safety"]["reasons"]
    assert controller.left_arm.command_count == 0
    assert controller.right_arm.command_count == 0
