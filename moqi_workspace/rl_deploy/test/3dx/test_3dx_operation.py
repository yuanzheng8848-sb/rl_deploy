#!/usr/bin/env python3
"""
Standalone 3D mouse teleop test aligned with train_pick_place.py.

控制语义与 EvdevSpacemouseIntervention 保持一致：
- 按钮 1（BTN_0 / BTN_LEFT）按下沿切换 intervention mode 开/关。
- 按钮 2（BTN_1 / BTN_RIGHT）按下沿切换 gripper 开/关状态。
- intervention mode 开启后：
  - 3D 鼠标 xyz -> EE 平移增量
  - 3D 鼠标 rx/ry/rz -> EE 姿态增量
  - gripper 使用 toggle 状态映射到归一化动作 ±1
- intervention mode 关闭时，不发送控制指令。
- intervention mode 开启但没有有效输入时，不推进 env.step()，保持与训练一致。

python moqi_workspace/rl_deploy/test/3dx/test_3dx_operation.py   --arm right   --hz 5   --deadzone 0.12   --trans-denom 250   --rot-denom 250   --print-raw --dominant-axis-only

"""

import argparse
import fcntl
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests
from scipy.spatial.transform import Rotation

from evdev import InputDevice, ecodes, list_devices

# Make rl_deploy importable from this test directory.
RL_DEPLOY_DIR = Path(__file__).resolve().parents[2]
if str(RL_DEPLOY_DIR) not in sys.path:
    sys.path.append(str(RL_DEPLOY_DIR))

from openarm_env import OpenArmEnv, DefaultOpenArmConfig


AXIS_CODES = {
    ecodes.ABS_X: "x",
    ecodes.ABS_Y: "y",
    ecodes.ABS_Z: "z",
    ecodes.ABS_RX: "rx",
    ecodes.ABS_RY: "ry",
    ecodes.ABS_RZ: "rz",
    ecodes.REL_X: "x",
    ecodes.REL_Y: "y",
    ecodes.REL_Z: "z",
    ecodes.REL_RX: "rx",
    ecodes.REL_RY: "ry",
    ecodes.REL_RZ: "rz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test SpaceMouse teleop with OpenArmEnv")
    parser.add_argument(
        "--event",
        default="/dev/input/event7",
        help="SpaceMouse event path, or 'auto' to detect like train_pick_place.py",
    )
    parser.add_argument("--server-url", default="http://127.0.0.1:5000/", help="OpenArm server URL")
    parser.add_argument("--arm", default="right", choices=["left", "right", "both"], help="Controlled arm")
    parser.add_argument("--hz", type=float, default=10.0, help="Control frequency")
    parser.add_argument(
        "--control-hz",
        type=float,
        default=80.0,
        help="High-rate 3dx target update frequency used by realtime servo mode",
    )
    parser.add_argument("--trans-denom", type=float, default=420.0, help="Translation (xyz) normalization")
    parser.add_argument("--rot-denom", type=float, default=380.0, help="Rotation (rx,ry,rz) normalization")
    parser.add_argument("--deadzone", type=float, default=0.08, help="Translation axis deadzone in normalized space")
    parser.add_argument(
        "--rot-deadzone",
        type=float,
        default=0.16,
        help="Rotation axis deadzone in normalized space",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="0 means run until Ctrl+C")
    parser.add_argument("--print-every", type=int, default=10, help="Print status every N steps")
    # 末端位移轴映射：设备轴 -> 末端空间 XYZ。取值 x,y,z 或 -x,-y,-z
    parser.add_argument("--ee-x", default="x", help="Device axis for EE X delta (e.g. x, -y)")
    parser.add_argument("--ee-y", default="-y", help="Device axis for EE Y delta")
    parser.add_argument("--ee-z", default="-z", help="Device axis for EE Z delta")
    parser.add_argument("--print-raw", action="store_true", help="Print raw spacemouse values and action")
    parser.add_argument("--realtime-servo", dest="realtime_servo", action="store_true", help="Use realtime servo mode")
    parser.add_argument("--no-realtime-servo", dest="realtime_servo", action="store_false", help="Disable realtime servo mode")
    parser.add_argument(
        "--servo-backend",
        default="analytic",
        choices=["baseik", "analytic"],
        help="Realtime servo backend. 'analytic' mirrors record_demo's Triangle/workspace constraint path",
    )
    parser.add_argument("--servo-hz", type=float, default=100.0, help="Server realtime servo loop frequency")
    parser.add_argument(
        "--servo-trans-step", type=float, default=0.004, help="Max translation step (m) per servo cycle"
    )
    parser.add_argument(
        "--servo-rot-step", type=float, default=0.012, help="Max rotation step (rad) per servo cycle"
    )
    parser.add_argument(
        "--servo-gripper-step",
        type=float,
        default=0.05,
        help="Max gripper step per servo cycle",
    )
    parser.add_argument(
        "--gripper-open-cmd",
        type=float,
        default=-0.95,
        help="Hardware gripper command used for open state in realtime servo mode",
    )
    parser.add_argument(
        "--gripper-close-cmd",
        type=float,
        default=0.0,
        help="Hardware gripper command used for close state in realtime servo mode",
    )
    parser.add_argument(
        "--dominant-axis-only",
        action="store_true",
        help="Only keep the currently strongest SpaceMouse axis for single-axis direction testing",
    )
    parser.set_defaults(realtime_servo=True)
    return parser.parse_args()


def check_server(server_url: str) -> None:
    url = server_url.rstrip("/") + "/getstate"
    try:
        resp = requests.post(url, timeout=2.0)
        resp.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach OpenArm server at {url}. "
            f"Please run openarm_server.py first. Detail: {exc}"
        ) from exc


def get_arm_indices(arm: str) -> list:
    if arm == "both":
        return [0, 1]
    if arm == "left":
        return [0]
    return [1]


def post_json(server_url: str, route: str, payload: dict | None = None, timeout: float = 2.0) -> requests.Response:
    url = server_url.rstrip("/") + route
    return requests.post(url, json=payload or {}, timeout=timeout)


def is_analytic_backend(args: argparse.Namespace) -> bool:
    return args.realtime_servo and args.servo_backend == "analytic"


def gripper_action_to_cmd(args: argparse.Namespace, raw_action: float) -> float:
    return float(args.gripper_close_cmd if raw_action > 0 else args.gripper_open_cmd)


def desired_gripper_cmds(args: argparse.Namespace, env: OpenArmEnv, action: np.ndarray) -> list[float]:
    gripper_cmds = [float(x) for x in env.curr_gripper_pos]
    active_indices = get_arm_indices(args.arm)
    for arm_idx in active_indices:
        arm_action = action if args.arm != "both" else action[arm_idx * 7 : (arm_idx + 1) * 7]
        gripper_cmds[arm_idx] = gripper_action_to_cmd(args, float(arm_action[6]))
    return gripper_cmds


def gripper_needs_keepalive(args: argparse.Namespace, env: OpenArmEnv, action: np.ndarray) -> bool:
    desired = desired_gripper_cmds(args, env, action)
    threshold = max(args.servo_gripper_step, 0.02)
    for arm_idx in get_arm_indices(args.arm):
        if abs(float(env.curr_gripper_pos[arm_idx]) - float(desired[arm_idx])) > threshold:
            return True
    return False


def has_3dx_axes(dev: InputDevice) -> bool:
    caps = dev.capabilities(absinfo=False)
    abs_codes = set(caps.get(ecodes.EV_ABS, []))
    rel_codes = set(caps.get(ecodes.EV_REL, []))
    trans_abs = {ecodes.ABS_X, ecodes.ABS_Y, ecodes.ABS_Z}
    rot_abs = {ecodes.ABS_RX, ecodes.ABS_RY, ecodes.ABS_RZ}
    trans_rel = {ecodes.REL_X, ecodes.REL_Y, ecodes.REL_Z}
    rot_rel = {ecodes.REL_RX, ecodes.REL_RY, ecodes.REL_RZ}
    return (trans_abs.issubset(abs_codes) and rot_abs.issubset(abs_codes)) or (
        trans_rel.issubset(rel_codes) and rot_rel.issubset(rel_codes)
    )


def auto_detect_event_path() -> tuple:
    candidates = []
    for path in list_devices():
        try:
            dev = InputDevice(path)
        except (PermissionError, OSError):
            continue

        name = (dev.name or "").lower()
        name_hit = any(
            key in name
            for key in (
                "3dconnexion",
                "spacemouse",
                "space mouse",
                "spacenavigator",
                "space navigator",
            )
        )
        axes_hit = has_3dx_axes(dev)
        if name_hit or axes_hit:
            score = (100 if name_hit else 0) + (10 if axes_hit else 0)
            candidates.append((score, path, dev.name))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def open_device(event_path: str) -> InputDevice:
    resolved_path = event_path
    if str(event_path).strip().lower() == "auto":
        resolved_path, detected_name = auto_detect_event_path()
        if resolved_path is None:
            raise RuntimeError("SpaceMouse auto-detect failed.")
        print(f"[Spacemouse] auto-detected device: {resolved_path} ({detected_name})")

    try:
        dev = InputDevice(resolved_path)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Event device not found: {resolved_path}") from exc
    except PermissionError as exc:
        raise RuntimeError(f"Permission denied: {resolved_path}. Try sudo.") from exc

    print("=== SpaceMouse Device ===")
    print(f"path    : {dev.path}")
    print(f"name    : {dev.name}")
    print(f"vendor  : 0x{dev.info.vendor:04x}")
    print(f"product : 0x{dev.info.product:04x}")
    print()

    try:
        dev.grab()
    except OSError:
        print("[WARN] Could not grab event device; continuing without exclusive access.")

    flags = fcntl.fcntl(dev.fd, fcntl.F_GETFL)
    fcntl.fcntl(dev.fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    return dev


def apply_deadzone(value: float, threshold: float) -> float:
    return 0.0 if abs(value) < threshold else value


def _parse_axis_spec(spec: str) -> tuple:
    """Parse --ee-x/--ee-y/--ee-z: 'x' -> ('x', 1), '-y' -> ('y', -1)."""
    s = spec.strip().lower()
    if s.startswith("-"):
        return s[1:], -1.0
    return s, 1.0


def device_to_ee_translation(
    axes: dict, trans_denom: float, ee_x_spec: str, ee_y_spec: str, ee_z_spec: str
) -> tuple:
    """
    将 3D 鼠标平移三轴映射为末端空间位移增量（归一化 [-1,1]）。
    语义：鼠标在设备上的位移 = 末端在空间中的位移方向与幅度。
    """
    def get_norm(axis_spec: str) -> float:
        name, sign = _parse_axis_spec(axis_spec)
        raw = axes.get(name, 0.0)
        return sign * np.clip(raw / trans_denom, -1.0, 1.0)

    return get_norm(ee_x_spec), get_norm(ee_y_spec), get_norm(ee_z_spec)


def get_rotation_action(axes: dict, rot_denom: float) -> tuple:
    return (
        np.clip(axes["rx"] / rot_denom, -1.0, 1.0),
        np.clip(-axes["ry"] / rot_denom, -1.0, 1.0),
        np.clip(-axes["rz"] / rot_denom, -1.0, 1.0),
    )


def get_axis_normalized_values(args: argparse.Namespace, axes: dict) -> dict:
    dx, dy, dz = device_to_ee_translation(
        axes, args.trans_denom, args.ee_x, args.ee_y, args.ee_z
    )
    return {
        "x": np.clip(axes["x"] / args.trans_denom, -1.0, 1.0),
        "y": np.clip(axes["y"] / args.trans_denom, -1.0, 1.0),
        "z": np.clip(axes["z"] / args.trans_denom, -1.0, 1.0),
        "rx": np.clip(axes["rx"] / args.rot_denom, -1.0, 1.0),
        "ry": np.clip(-axes["ry"] / args.rot_denom, -1.0, 1.0),
        "rz": np.clip(-axes["rz"] / args.rot_denom, -1.0, 1.0),
        "ee_x": dx,
        "ee_y": dy,
        "ee_z": dz,
    }


def isolate_dominant_axis(args: argparse.Namespace, axes: dict) -> tuple:
    norm = get_axis_normalized_values(args, axes)
    candidates = ("x", "y", "z", "rx", "ry", "rz")
    if all(abs(norm[name]) <= 1e-9 for name in candidates):
        return {k: 0.0 for k in axes}, None, norm
    dominant_axis = max(candidates, key=lambda name: abs(norm[name]))
    isolated = {k: 0.0 for k in axes}
    isolated[dominant_axis] = axes[dominant_axis]
    return isolated, dominant_axis, norm


def describe_device_axis_effect(args: argparse.Namespace, device_axis: str) -> str:
    if device_axis == "rx":
        return "EE_RX (+, rotation)"
    if device_axis == "ry":
        return "EE_RY (-, rotation)"
    if device_axis == "rz":
        return "EE_RZ (-, rotation)"

    effects = []
    for ee_axis, spec in (("X", args.ee_x), ("Y", args.ee_y), ("Z", args.ee_z)):
        mapped_axis, sign = _parse_axis_spec(spec)
        if mapped_axis == device_axis:
            direction = "+" if sign > 0 else "-"
            effects.append(f"EE_{ee_axis} ({direction})")

    if effects:
        return ", ".join(effects)
    return "Unmapped translation axis"


def print_dominant_axis_debug(
    args: argparse.Namespace,
    raw_axes: dict,
    isolated_axes: dict,
    dominant_axis: str,
    action: np.ndarray,
    tcp: np.ndarray,
    tcp_delta: np.ndarray | None = None,
    target_ref_xyz: np.ndarray | None = None,
) -> None:
    norm_raw = get_axis_normalized_values(args, raw_axes)
    norm_iso = get_axis_normalized_values(args, isolated_axes)
    effect = describe_device_axis_effect(args, dominant_axis)
    print(
        f"[DominantAxis] dominant={dominant_axis} effect={effect} "
        f"raw=({raw_axes['x']:+.0f},{raw_axes['y']:+.0f},{raw_axes['z']:+.0f};"
        f"{raw_axes['rx']:+.0f},{raw_axes['ry']:+.0f},{raw_axes['rz']:+.0f}) "
        f"norm_raw=({norm_raw['x']:+.3f},{norm_raw['y']:+.3f},{norm_raw['z']:+.3f};"
        f"{norm_raw['rx']:+.3f},{norm_raw['ry']:+.3f},{norm_raw['rz']:+.3f}) "
        f"isolated=({isolated_axes['x']:+.0f},{isolated_axes['y']:+.0f},{isolated_axes['z']:+.0f};"
        f"{isolated_axes['rx']:+.0f},{isolated_axes['ry']:+.0f},{isolated_axes['rz']:+.0f}) "
        f"ee_cmd=({norm_iso['ee_x']:+.3f},{norm_iso['ee_y']:+.3f},{norm_iso['ee_z']:+.3f}) "
        f"env_action={np.round(action, 4).tolist()} "
        f"target_ref_xyz={None if target_ref_xyz is None else np.round(target_ref_xyz, 4).tolist()} "
        f"tcp_delta={None if tcp_delta is None else np.round(tcp_delta, 4).tolist()} "
        f"tcp_xyz={np.round(tcp, 4).tolist()}"
    )


def update_target_pose_ref(target_pose_ref: np.ndarray, action: np.ndarray, arm: str, action_scale: np.ndarray) -> np.ndarray:
    updated = np.array(target_pose_ref, copy=True)
    for arm_idx in get_arm_indices(arm):
        arm_action = action if arm != "both" else action[arm_idx * 7 : (arm_idx + 1) * 7]
        updated[arm_idx, :3] += arm_action[:3] * action_scale[0]
        rot_curr = Rotation.from_quat(updated[arm_idx, 3:])
        rot_delta = Rotation.from_euler("xyz", arm_action[3:6] * action_scale[1])
        updated[arm_idx, 3:] = (rot_delta * rot_curr).as_quat()
    return updated


def start_realtime_servo(args: argparse.Namespace, env: OpenArmEnv, target_pose_ref: np.ndarray) -> None:
    payload = {
        "arr": np.asarray(target_pose_ref, dtype=np.float32).tolist(),
        "gripper": [float(x) for x in env.curr_gripper_pos],
        "servo_hz": args.servo_hz,
        "trans_step": args.servo_trans_step,
        "rot_step": args.servo_rot_step,
        "gripper_step": args.servo_gripper_step,
        "arm": args.arm,
        "backend": args.servo_backend,
    }
    resp = post_json(args.server_url, "/servo/start", payload=payload, timeout=3.0)
    resp.raise_for_status()


def update_realtime_servo_target(args: argparse.Namespace, env: OpenArmEnv, target_pose_ref: np.ndarray, action: np.ndarray) -> None:
    gripper_cmds = desired_gripper_cmds(args, env, action)

    payload = {
        "arr": np.asarray(target_pose_ref, dtype=np.float32).tolist(),
        "gripper": gripper_cmds,
    }
    resp = post_json(args.server_url, "/servo/target", payload=payload, timeout=2.0)
    resp.raise_for_status()


def stop_realtime_servo(args: argparse.Namespace) -> None:
    try:
        post_json(args.server_url, "/servo/stop", payload={}, timeout=2.0)
    except Exception:
        pass


def poll_events(dev: InputDevice, axes: dict, button_state: dict) -> bool:
    """轮询 3D 鼠标事件，更新 axes 和 toggle 状态。返回本轮是否收到过任意事件。"""
    got_any = False
    button_state["gripper_toggle_changed"] = False
    while True:
        try:
            ev = dev.read_one()
        except BlockingIOError:
            return got_any
        except OSError:
            return got_any

        if ev is None:
            return got_any

        got_any = True
        if ev.type in (ecodes.EV_ABS, ecodes.EV_REL):
            axis = AXIS_CODES.get(ev.code)
            if axis is not None:
                if ev.type == ecodes.EV_REL:
                    axes[axis] += float(ev.value)
                else:
                    axes[axis] = float(ev.value)
        elif ev.type == ecodes.EV_KEY:
            if ev.code in (ecodes.BTN_0, ecodes.BTN_LEFT):
                if int(ev.value) == 1:
                    button_state["intervention_mode"] = not button_state["intervention_mode"]
                    mode = "ON" if button_state["intervention_mode"] else "OFF"
                    print(f"[Spacemouse] intervention mode: {mode}")
            elif ev.code in (ecodes.BTN_1, ecodes.BTN_RIGHT):
                if int(ev.value) == 1:
                    button_state["gripper_close"] = not button_state["gripper_close"]
                    button_state["gripper_toggle_changed"] = True
                    mode = "CLOSE" if button_state["gripper_close"] else "OPEN"
                    print(f"[Spacemouse] gripper mode: {mode}")


def has_motion_input(args: argparse.Namespace, axes: dict) -> bool:
    """是否有有效运动输入：xyz 或 rx,ry,rz 超出死区。"""
    dx, dy, dz = device_to_ee_translation(
        axes, args.trans_denom, args.ee_x, args.ee_y, args.ee_z
    )
    if abs(dx) > args.deadzone or abs(dy) > args.deadzone or abs(dz) > args.deadzone:
        return True
    rx, ry, rz = get_rotation_action(axes, args.rot_denom)
    rx = abs(rx)
    ry = abs(ry)
    rz = abs(rz)
    if rx > args.rot_deadzone or ry > args.rot_deadzone or rz > args.rot_deadzone:
        return True
    return False


def build_action(args: argparse.Namespace, axes: dict, gripper_close: bool) -> np.ndarray:
    """按照训练中的 intervention 逻辑构造 7 维动作。"""
    dx, dy, dz = device_to_ee_translation(
        axes, args.trans_denom, args.ee_x, args.ee_y, args.ee_z
    )
    dx = apply_deadzone(dx, args.deadzone)
    dy = apply_deadzone(dy, args.deadzone)
    dz = apply_deadzone(dz, args.deadzone)
    rx, ry, rz = get_rotation_action(axes, args.rot_denom)
    rx = apply_deadzone(rx, args.rot_deadzone)
    ry = apply_deadzone(ry, args.rot_deadzone)
    rz = apply_deadzone(rz, args.rot_deadzone)
    action = np.array(
        [dx, dy, dz, rx, ry, rz, 1.0 if gripper_close else -1.0],
        dtype=np.float32,
    )
    return action


def sample_env_like_step(
    env: OpenArmEnv,
    prev_obs: dict,
    action: np.ndarray,
    backend: str,
) -> tuple[dict, float, bool, bool, dict, dict]:
    next_obs = env.refresh_obs()
    reward = 0.0
    done = False
    truncated = False
    transition = {
        "observations": prev_obs,
        "actions": np.array(action, copy=True),
        "next_observations": next_obs,
        "rewards": reward,
        "dones": done,
        "truncated": truncated,
    }
    info = {
        "intervene_action": np.array(action, copy=True),
        "control_backend": backend,
        "transition": transition,
    }
    return next_obs, reward, done, truncated, info, transition


def main() -> int:
    args = parse_args()

    check_server(args.server_url)
    dev = open_device(args.event)

    cfg = DefaultOpenArmConfig()
    cfg.SERVER_URL = args.server_url

    env = OpenArmEnv(
        hz=int(args.hz),
        fake_env=False,
        save_video=False,
        config=cfg,
        max_episode_length=10**9,
        arm=args.arm,
    )

    obs, _ = env.reset()

    axes = {"x": 0.0, "y": 0.0, "z": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}
    button_state = {
        "intervention_mode": False,
        "gripper_close": False,
        "gripper_toggle_changed": False,
    }

    print("=== 3D Mouse Teleop (aligned with train_pick_place.py) ===")
    print("CTRL+C to stop.")
    print("Button0/Left: toggle intervention mode ON/OFF")
    print("Button1/Right: toggle gripper CLOSE/OPEN")
    print("When intervention mode is OFF, no command is sent.")
    print("When intervention mode is ON but idle, env.step() is skipped.")
    print(f"Axis mapping: EE X<-{args.ee_x}, Y<-{args.ee_y}, Z<-{args.ee_z}")
    if args.realtime_servo:
        print(
            "Realtime servo mode: ON "
            f"(control_hz={args.control_hz}, sample_hz={args.hz}, "
            f"servo_backend={args.servo_backend}, servo_hz={args.servo_hz}, trans_step={args.servo_trans_step}, "
            f"rot_step={args.servo_rot_step}, gripper_step={args.servo_gripper_step})"
        )
    if args.dominant_axis_only:
        print("Dominant-axis-only mode: ON (only the strongest axis is sent each cycle)")
    print()

    step = 0
    poll_dt = 0.02  # 轮询间隔，无输入时占空比低
    prev_tcp = None
    target_pose_ref = env.currpos.copy()
    last_sample_ts = time.time()
    servo_running = False
    idle_hold_sent = False
    prev_obs_for_transition = obs
    if args.realtime_servo:
        start_realtime_servo(args, env, target_pose_ref)
        servo_running = True

    try:
        while True:
            if args.max_steps > 0 and step >= args.max_steps:
                break

            got_event = poll_events(dev, axes, button_state)
            # REL 设备：无新事件时把轴清零，避免残留导致一直判定为有输入
            if not got_event:
                for k in axes:
                    axes[k] = 0.0

            raw_axes = dict(axes)
            dominant_axis = None
            effective_axes = axes
            if args.dominant_axis_only:
                effective_axes, dominant_axis, _ = isolate_dominant_axis(args, axes)

            if not button_state["intervention_mode"]:
                if args.realtime_servo and servo_running:
                    obs = env.refresh_obs()
                    target_pose_ref = env.currpos.copy()
                    stop_realtime_servo(args)
                    obs = env.refresh_obs()
                    target_pose_ref = env.currpos.copy()
                    servo_running = False
                    idle_hold_sent = False
                time.sleep(poll_dt)
                continue

            has_motion = has_motion_input(args, effective_axes)
            if (not has_motion) and (not button_state["gripper_toggle_changed"]):
                if args.realtime_servo:
                    if is_analytic_backend(args):
                        if not servo_running:
                            obs = env.refresh_obs()
                            target_pose_ref = env.currpos.copy()
                            start_realtime_servo(args, env, target_pose_ref)
                            servo_running = True
                        hold_action = np.array(
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 if button_state["gripper_close"] else -1.0],
                            dtype=np.float32,
                        )
                        if not idle_hold_sent:
                            obs = env.refresh_obs()
                            target_pose_ref = env.currpos.copy()
                            update_realtime_servo_target(args, env, target_pose_ref, hold_action)
                            idle_hold_sent = True
                        else:
                            obs = env.refresh_obs()
                            if gripper_needs_keepalive(args, env, hold_action):
                                update_realtime_servo_target(args, env, target_pose_ref, hold_action)
                    elif servo_running:
                        obs = env.refresh_obs()
                        target_pose_ref = env.currpos.copy()
                        stop_realtime_servo(args)
                        obs = env.refresh_obs()
                        target_pose_ref = env.currpos.copy()
                        servo_running = False
                        idle_hold_sent = False
                    time.sleep(max(0, (1.0 / args.control_hz) - 0.001))
                else:
                    time.sleep(poll_dt)
                continue

            action = build_action(args, effective_axes, button_state["gripper_close"])
            idle_hold_sent = False
            if args.print_raw and not args.dominant_axis_only:
                print(
                    f"[Spacemouse] mode={'ON' if button_state['intervention_mode'] else 'OFF'} "
                    f"raw xyz=({effective_axes['x']:+.0f},{effective_axes['y']:+.0f},{effective_axes['z']:+.0f}) "
                    f"rpy=({effective_axes['rx']:+.0f},{effective_axes['ry']:+.0f},{effective_axes['rz']:+.0f}) "
                    f"btn2_close={int(button_state['gripper_close'])} "
                    f"-> action={np.round(action, 4).tolist()}"
                )

            if args.realtime_servo:
                if not servo_running:
                    obs = env.refresh_obs()
                    target_pose_ref = env.currpos.copy()
                    start_realtime_servo(args, env, target_pose_ref)
                    obs = env.refresh_obs()
                    target_pose_ref = env.currpos.copy()
                    update_realtime_servo_target(
                        args,
                        env,
                        target_pose_ref,
                        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 if button_state["gripper_close"] else -1.0], dtype=np.float32),
                    )
                    servo_running = True
                    idle_hold_sent = False
                target_pose_ref = update_target_pose_ref(target_pose_ref, action, args.arm, env.action_scale)
                update_realtime_servo_target(args, env, target_pose_ref, action)
                now = time.time()
                if now - last_sample_ts >= (1.0 / args.hz):
                    obs, reward, done, truncated, info, _ = sample_env_like_step(
                        env,
                        prev_obs_for_transition,
                        action,
                        args.servo_backend,
                    )
                    prev_obs_for_transition = obs
                    last_sample_ts = now
                    step += 1
                else:
                    time.sleep(max(0, (1.0 / args.control_hz) - 0.001))
                    continue
            else:
                obs, reward, done, truncated, info = env.step(action)
                prev_obs_for_transition = obs
                step += 1

            if args.arm == "both":
                tcp = obs["state"]["tcp_pose"][1, :3]
                target_ref_xyz = target_pose_ref[1, :3]
            else:
                tcp = obs["state"]["tcp_pose"][:3]
                target_ref_xyz = target_pose_ref[get_arm_indices(args.arm)[0], :3]
            tcp_delta = None if prev_tcp is None else (tcp - prev_tcp)
            prev_tcp = np.array(tcp, copy=True)

            if args.dominant_axis_only and dominant_axis is not None:
                print_dominant_axis_debug(
                    args=args,
                    raw_axes=raw_axes,
                    isolated_axes=effective_axes,
                    dominant_axis=dominant_axis,
                    action=action,
                    tcp=tcp,
                    tcp_delta=tcp_delta,
                    target_ref_xyz=target_ref_xyz,
                )

            if step % args.print_every == 0:
                print(
                    f"step={step} action={np.round(action, 3).tolist()} "
                    f"target_ref_xyz={np.round(target_ref_xyz, 3).tolist()} "
                    f"tcp_delta={None if tcp_delta is None else np.round(tcp_delta, 4).tolist()} "
                    f"tcp_xyz={np.round(tcp, 3).tolist()} reward={reward:.3f}"
                )

            if done or truncated:
                obs, _ = env.reset()
                target_pose_ref = env.currpos.copy()
                prev_tcp = None
                if args.realtime_servo:
                    start_realtime_servo(args, env, target_pose_ref)
                    obs = env.refresh_obs()
                    target_pose_ref = env.currpos.copy()
                    servo_running = True
                    idle_hold_sent = False
                prev_obs_for_transition = obs
                last_sample_ts = time.time()

            # 有输入时按控制频率限速，避免单次长按/持续偏转时指令过密
            if args.realtime_servo:
                time.sleep(max(0, (1.0 / args.control_hz) - 0.001))
            else:
                time.sleep(max(0, (1.0 / args.hz) - 0.001))

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        try:
            dev.ungrab()
        except OSError:
            pass
        if args.realtime_servo:
            stop_realtime_servo(args)
            servo_running = False
            idle_hold_sent = False
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
