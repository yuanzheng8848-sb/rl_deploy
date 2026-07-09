"""Read-only-ish OpenArm CAN smoke check.

This initializes the OpenArm object and reads motor state using the configured
interface. Do not run it until the robot is powered, supported, and safe.
"""

import argparse
from pathlib import Path

import yaml
import openarm_can as oa


RL_ROBOT_INFRA = Path(__file__).resolve().parents[1] / "rl_robot_infra"
CAN_CONFIG = RL_ROBOT_INFRA / "openarm_configs" / "can.yaml"


def _motor_types():
    return [
        oa.MotorType.DM8009,
        oa.MotorType.DM8009,
        oa.MotorType.DM4340,
        oa.MotorType.DM4340,
        oa.MotorType.DM4310,
        oa.MotorType.DM4310,
        oa.MotorType.DM4310,
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=["left", "right"], required=True)
    args = parser.parse_args()

    with open(CAN_CONFIG) as f:
        cfg = yaml.safe_load(f)

    side_cfg = cfg[args.side]
    motor_cfg = cfg["motors"]
    interface = side_cfg["interface"]
    can_fd = bool(side_cfg.get("can_fd", False))

    print(f"Connecting {args.side} arm on {interface}, can_fd={can_fd}")
    arm = oa.OpenArm(interface, can_fd)
    arm.init_arm_motors(
        _motor_types(),
        motor_cfg["arm_send_ids"],
        motor_cfg["arm_recv_ids"],
    )
    arm.init_gripper_motor(
        oa.MotorType.DM4310,
        motor_cfg["gripper_send_id"],
        motor_cfg["gripper_recv_id"],
    )
    arm.set_callback_mode_all(oa.CallbackMode.STATE)
    arm.refresh_all()
    arm.recv_all()

    positions = [motor.get_position() for motor in arm.get_arm().get_motors()]
    gripper = [motor.get_position() for motor in arm.get_gripper().get_motors()]
    print("arm positions:", positions)
    print("gripper positions:", gripper)


if __name__ == "__main__":
    main()
