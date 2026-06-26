"""
右臂关节自检脚本

逐个关节做小角度往返运动，根据反馈判断每个电机是否正常工作。

使用方法：
    sudo ip link set can1 up
    python3 test_right_arm_joints.py

安全提示：
- 运行前请确认右臂周围无障碍物，机械臂处于安全姿态
- Ctrl+C 可随时中止
- 测试角度默认 ±0.2 rad (~11.5°)，可通过 TEST_ANGLE 调整
"""

import time
import numpy as np
import openarm_can as oa


CAN_INTERFACE = "can1"           # 右臂 can1，左臂 can0
TEST_ANGLE = 0.4                 # 单关节往返测试幅度 (rad)
MOVE_DURATION = 1.5              # 单次运动时长 (s)
CONTROL_HZ = 50                  # 控制频率
# 按关节类型分配增益：DM8009 大力矩(肩部) > DM4340(肘部) > DM4310(腕部)
KP_PER_JOINT = [60, 50, 50, 50, 30, 30, 30]
KD_PER_JOINT = [6,   6,  6,  6, 1.5, 1.5, 1.5]
MOVE_THRESHOLD = 0.05            # 判定电机有响应的最小位移 (rad)


MOTOR_TYPES = [
    oa.MotorType.DM8009,   # joint 0
    oa.MotorType.DM8009,   # joint 1
    oa.MotorType.DM4340,   # joint 2
    oa.MotorType.DM4340,   # joint 3
    oa.MotorType.DM4310,   # joint 4
    oa.MotorType.DM4310,   # joint 5
    oa.MotorType.DM4310,   # joint 6
]
SEND_IDS = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]
RECV_IDS = [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]
GRIPPER_SEND_ID = 0x08
GRIPPER_RECV_ID = 0x18


def init_arm(arm):
    arm.init_arm_motors(MOTOR_TYPES, SEND_IDS, RECV_IDS)
    arm.init_gripper_motor(oa.MotorType.DM4310, GRIPPER_SEND_ID, GRIPPER_RECV_ID)

    # 使能前先读一次，检查CAN通信
    arm.set_callback_mode_all(oa.CallbackMode.STATE)
    for _ in range(5):
        arm.refresh_all()
        arm.recv_all()
        time.sleep(0.05)

    motors = arm.get_arm().get_motors()
    print("\n  [使能前 CAN 通信检查]")
    for i, m in enumerate(motors):
        has_data = (m.get_state_tmos() != 0 or abs(m.get_position()) > 0.0001)
        print(f"    joint {i}: pos={m.get_position():+.4f}, T_mos={m.get_state_tmos()}, "
              f"通信={'OK' if has_data else '无数据'}")

    # 使能
    print("\n  使能电机...")
    arm.set_callback_mode_all(oa.CallbackMode.IGNORE)
    arm.enable_all()
    arm.recv_all()
    time.sleep(0.3)

    # 使能后多次刷新确认
    arm.set_callback_mode_all(oa.CallbackMode.STATE)
    for _ in range(10):
        arm.refresh_all()
        arm.recv_all()
        time.sleep(0.05)

    print("  [使能后状态]")
    for i, m in enumerate(motors):
        print(f"    joint {i}: enabled={m.is_enabled()}, pos={m.get_position():+.4f}, "
              f"tau={m.get_torque():+.4f}, T_mos={m.get_state_tmos()}")
    print()


def read_arm_positions(arm):
    arm.refresh_all()
    arm.recv_all()
    return [m.get_position() for m in arm.get_arm().get_motors()]


def hold_all(arm, target_positions):
    """以分关节增益保持当前姿态，避免重力下滑。"""
    params = [oa.MITParam(KP_PER_JOINT[i], KD_PER_JOINT[i], q, 0, 0)
              for i, q in enumerate(target_positions)]
    arm.get_arm().mit_control_all(params)
    arm.recv_all()


def smooth_move_single_joint(arm, hold_positions, joint_idx, delta, duration):
    """
    保持其他关节位置不变，仅让 joint_idx 关节平滑移动 delta 弧度。
    返回运动结束时该关节实际到达的位置，以及过程中的最大力矩反馈。
    """
    steps = max(1, int(duration * CONTROL_HZ))
    start_q = hold_positions[joint_idx]
    target_q = start_q + delta
    max_tau = 0.0

    for i in range(steps + 1):
        t = i / steps
        smooth_t = t * t * (3.0 - 2.0 * t)
        cur_targets = list(hold_positions)
        cur_targets[joint_idx] = start_q + (target_q - start_q) * smooth_t

        params = [oa.MITParam(KP_PER_JOINT[k], KD_PER_JOINT[k], q, 0, 0)
                  for k, q in enumerate(cur_targets)]
        arm.get_arm().mit_control_all(params)
        arm.recv_all()

        tau = abs(arm.get_arm().get_motors()[joint_idx].get_torque())
        if tau > max_tau:
            max_tau = tau

        time.sleep(1.0 / CONTROL_HZ)

    actual = read_arm_positions(arm)[joint_idx]
    return actual, max_tau


def test_joint(arm, hold_positions, joint_idx):
    """对单个关节做 +TEST_ANGLE → -TEST_ANGLE → 0 的往返测试。"""
    motor_name = MOTOR_TYPES[joint_idx].name
    print(f"\n--- 测试 joint {joint_idx} ({motor_name}, send_id=0x{SEND_IDS[joint_idx]:02x}) ---")

    start_q = hold_positions[joint_idx]
    print(f"  起始位置: {start_q:+.4f} rad")

    # 正向
    pos_pos, tau_pos = smooth_move_single_joint(arm, hold_positions, joint_idx, +TEST_ANGLE, MOVE_DURATION)
    delta_pos = pos_pos - start_q
    ratio_pos = delta_pos / TEST_ANGLE
    print(f"  +{TEST_ANGLE:.2f} rad 指令 → 实际位移 {delta_pos:+.4f} rad (到达率 {ratio_pos*100:+5.1f}%) 最大力矩={tau_pos:.3f}Nm")

    # 反向（从正向位置回到 -TEST_ANGLE 相对起点 = 总移动 -2*TEST_ANGLE）
    hold_at_pos = list(hold_positions)
    hold_at_pos[joint_idx] = pos_pos
    neg_pos, tau_neg = smooth_move_single_joint(arm, hold_at_pos, joint_idx, -2 * TEST_ANGLE, MOVE_DURATION)
    delta_neg = neg_pos - pos_pos
    ratio_neg = delta_neg / (-2 * TEST_ANGLE)
    print(f"  -{2 * TEST_ANGLE:.2f} rad 指令 → 实际位移 {delta_neg:+.4f} rad (到达率 {ratio_neg*100:+5.1f}%) 最大力矩={tau_neg:.3f}Nm")

    # 回到起点
    hold_at_neg = list(hold_positions)
    hold_at_neg[joint_idx] = neg_pos
    smooth_move_single_joint(arm, hold_at_neg, joint_idx, start_q - neg_pos, MOVE_DURATION)
    final_q = read_arm_positions(arm)[joint_idx]
    print(f"  归位完成: {final_q:+.4f} rad")

    # 运动后读温度
    m = arm.get_arm().get_motors()[joint_idx]
    print(f"  测试后: T_mos={m.get_state_tmos()}℃, T_rotor={m.get_state_trotor()}℃, tau={m.get_torque():+.4f}Nm")

    # 判定（基于到达率，到达率正常表示电机能驱动）
    OK_RATIO = 0.30
    PARTIAL_RATIO = 0.15
    has_pos = ratio_pos > PARTIAL_RATIO
    has_neg = ratio_neg > PARTIAL_RATIO
    good_pos = ratio_pos > OK_RATIO
    good_neg = ratio_neg > OK_RATIO

    if good_pos and good_neg:
        verdict = "OK"
    elif has_pos and has_neg:
        verdict = "OK(弱): 双向有响应但到达率偏低（增益/重力/摩擦）"
    elif has_pos or has_neg:
        verdict = "WARN: 仅单向有响应（多半是重力/限位，非电机故障）"
    else:
        verdict = "FAIL: 双向几乎无响应（线缆/电源/CAN ID/电机故障可能性高）"
    print(f"  结论: {verdict}")
    return verdict


def diagnose_all(arm):
    """读取所有关节的诊断信息：使能状态、位置/速度/力矩反馈、MOS温度、转子温度。"""
    arm.refresh_all()
    arm.recv_all()
    motors = arm.get_arm().get_motors()
    print("\n========== 电机诊断 ==========")
    print(f"{'idx':<4}{'type':<10}{'send':<6}{'enabled':<9}{'q(rad)':<10}"
          f"{'dq':<8}{'tau(Nm)':<10}{'T_mos':<7}{'T_rotor':<8}")
    for i, m in enumerate(motors):
        try:
            enabled = m.is_enabled()
        except Exception:
            enabled = "?"
        print(f"{i:<4}{MOTOR_TYPES[i].name:<10}0x{SEND_IDS[i]:02x}  "
              f"{str(enabled):<9}{m.get_position():+.4f}   "
              f"{m.get_velocity():+.3f}  {m.get_torque():+.4f}    "
              f"{m.get_state_tmos():<7}{m.get_state_trotor():<8}")
    print("注意: 库未解析 DM 协议错误码字段，但 T_mos/T_rotor 异常高(>60℃)")
    print("      或 enabled=False / tau 始终接近 0 都可作为故障线索。")
    print("==================================\n")


def main():
    print(f"连接 {CAN_INTERFACE} ...")
    arm = oa.OpenArm(CAN_INTERFACE, False)
    init_arm(arm)

    # 测试前先打印一次诊断信息
    diagnose_all(arm)

    hold_positions = read_arm_positions(arm)
    print(f"初始关节位置: {[f'{q:+.3f}' for q in hold_positions]}")
    print(f"测试参数: ±{TEST_ANGLE} rad, kp={KP_PER_JOINT}, kd={KD_PER_JOINT}, duration={MOVE_DURATION}s")
    print("3 秒后开始，可 Ctrl+C 中止 ...")
    time.sleep(3.0)

    results = {}
    try:
        for j in range(len(SEND_IDS)):
            # 每次重新读取作为基准，防止累计误差
            hold_positions = read_arm_positions(arm)
            results[j] = test_joint(arm, hold_positions, j)
            # 测试间保持当前姿态 0.5s
            for _ in range(int(0.5 * CONTROL_HZ)):
                hold_all(arm, read_arm_positions(arm))
                time.sleep(1.0 / CONTROL_HZ)
    except KeyboardInterrupt:
        print("\n用户中止")

    print("\n========== 自检汇总 ==========")
    for j, verdict in results.items():
        print(f"  joint {j} (0x{SEND_IDS[j]:02x}, {MOTOR_TYPES[j].name}): {verdict}")
    print("=================================\n")

    # 测试后再打印一次诊断信息（看温度变化和反馈扭矩）
    diagnose_all(arm)

    # 末态：所有关节用低增益保持当前位置，避免任何关节被恒力矩推动
    final_positions = read_arm_positions(arm)
    final_params = [oa.MITParam(KP_PER_JOINT[i], KD_PER_JOINT[i], q, 0, 0)
                    for i, q in enumerate(final_positions)]
    arm.get_arm().mit_control_all(final_params)
    arm.get_gripper().mit_control_all([oa.MITParam(0, 0, 0, 0, 0)])
    arm.recv_all()


if __name__ == "__main__":
    main()
