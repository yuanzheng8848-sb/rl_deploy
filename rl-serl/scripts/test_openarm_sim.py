#!/usr/bin/env python3
"""Run OpenArm simulation checks from low-level primitives to the Gym env."""

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "rl_robot_infra"), str(ROOT)]

from openarm_control.controller import OpenArmController
from openarm_control.safety import SafetyConfig, SafetyMonitor
from openarm_control.trajectory import JointTrajectoryLimiter, ServoGains, ServoLimits
from openarm_control.types import ArmState, RobotState
from openarm_env.mock_hardware import SimulatedArmBus


def stage_trajectory():
    limiter = JointTrajectoryLimiter(ServoLimits.conservative(), ServoGains.default())
    q = np.zeros(14)
    limiter.reset(q)
    command = limiter.step(np.ones(14), q, q, dt=0.01)
    assert np.all(command.q > 0) and np.all(command.q < 1)
    assert np.all(np.abs(command.dq) <= limiter.limits.velocity)
    print("[1/4] trajectory: PASS (bounded MIT command)")


def _healthy_state():
    left = ArmState.zeros()
    right = ArmState.zeros()
    now = __import__("time").time()
    for arm in (left, right):
        arm.enabled[:] = True
        arm.gripper_enabled[:] = True
        arm.t_mos[:] = 25.0
        arm.t_rotor[:] = 25.0
        arm.timestamp = now
    return RobotState(left=left, right=right, timestamp=now)


def stage_safety():
    monitor = SafetyMonitor(SafetyConfig(), np.ones(14) * 6, np.ones(14) * 1.2)
    state = _healthy_state()
    assert monitor.evaluate(state, state.timestamp).ok_to_send
    state.left.t_mos[0] = 90.0
    stopped = monitor.evaluate(state, state.timestamp)
    assert not stopped.ok_to_send and "TEMPERATURE_STOP" in stopped.reasons
    print("[2/4] safety: PASS (healthy send + overtemperature stop)")


def stage_controller(steps=25):
    controller = OpenArmController(bus_factory=SimulatedArmBus)
    target = np.r_[np.ones(7) * 0.08, np.ones(7) * -0.08]
    for _ in range(steps):
        ok, info = controller.command_joint_target(target, [0.03, -0.04], dt=0.01, source="sim-smoke")
        assert ok, info
    state = controller.read_state()
    assert controller.left_arm.command_count == steps
    assert controller.right_arm.command_count == steps
    assert np.linalg.norm(state.q) > 0
    np.testing.assert_allclose(state.gripper_q, [0.03, -0.04])
    print(f"[3/4] controller: PASS ({steps} limited dual-arm commands, q_norm={np.linalg.norm(state.q):.5f})")


def stage_env():
    try:
        from openarm_env.envs.openarm_env import DefaultOpenArmConfig, OpenArmEnv
    except ImportError as exc:
        raise RuntimeError("Gym stage needs rl-serl Python requirements") from exc

    class NoCameraConfig(DefaultOpenArmConfig):
        CAMERAS = {}

    env = OpenArmEnv(env_mode="virtual", hz=1000, config=NoCameraConfig())
    try:
        obs, _ = env.reset()
        action = np.zeros(14, dtype=np.float32)
        action[0] = 1.0
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert env.observation_space.contains(next_obs)
        assert next_obs["state"]["tcp_pose"][0, 0] > obs["state"]["tcp_pose"][0, 0]
        assert (reward, terminated, truncated, info["state_stale"]) == (0.0, False, False, False)
    finally:
        env.close()
    print("[4/4] gym-env: PASS (virtual reset/step/observation contract)")


STAGES = {
    "trajectory": stage_trajectory,
    "safety": stage_safety,
    "controller": stage_controller,
    "env": stage_env,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", *STAGES], default="all")
    parser.add_argument("--steps", type=int, default=25, help="controller commands for the controller stage")
    args = parser.parse_args()
    selected = STAGES if args.stage == "all" else {args.stage: STAGES[args.stage]}
    for name, function in selected.items():
        function(args.steps) if name == "controller" else function()
    print("OpenArm simulation checks completed.")


if __name__ == "__main__":
    main()
