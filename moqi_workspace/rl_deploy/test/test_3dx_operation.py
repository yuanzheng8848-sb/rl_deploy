#!/usr/bin/env python3
"""
Dual 3D mouse teleop test aligned with the multistage bimanual train pipeline.
"""

import argparse
import fcntl
import os
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import requests
from evdev import InputDevice, ecodes, list_devices
from scipy.spatial.transform import Rotation

RL_DEPLOY_DIR = Path(__file__).resolve().parents[2]
if str(RL_DEPLOY_DIR) not in sys.path:
    sys.path.append(str(RL_DEPLOY_DIR))

from openarm_env import (
    DefaultOpenArmConfig,
    OpenArmEnv,
    apply_binary_gripper_logic,
    binary_gripper_state_to_cmd,
)

DEFAULT_JOINT_TARGET_DIR = RL_DEPLOY_DIR / "joint_targets"
JOINT_TARGET_PATTERN = re.compile(r"joint_target_(\d+)\.npy$")
EE_TARGET_PATTERN = re.compile(r"ee_target_(\d+)\.npy$")


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


def parse_args():
    parser = argparse.ArgumentParser(description="Dual SpaceMouse teleop test with OpenArmEnv")
    parser.add_argument("--server-url", default="http://127.0.0.1:5000/")
    parser.add_argument("--arm", default="both", choices=["left", "right", "both"])
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--control-hz", type=float, default=80.0)
    parser.add_argument("--left-event", default="/dev/input/event17")
    parser.add_argument("--right-event", default="/dev/input/event16")
    parser.add_argument("--trans-denom", type=float, default=420.0)
    parser.add_argument("--rot-denom", type=float, default=380.0)
    parser.add_argument("--deadzone", type=float, default=0.08)
    parser.add_argument("--rot-deadzone", type=float, default=0.16)
    parser.add_argument("--ee-x", default="x")
    parser.add_argument("--ee-y", default="-y")
    parser.add_argument("--ee-z", default="-z")
    parser.add_argument("--servo-hz", type=float, default=100.0)
    parser.add_argument("--servo-backend", default="analytic", choices=["baseik", "analytic"])
    parser.add_argument("--servo-trans-step", type=float, default=0.004)
    parser.add_argument("--servo-rot-step", type=float, default=0.012)
    parser.add_argument("--servo-gripper-step", type=float, default=0.05)
    parser.add_argument("--gripper-open-cmd", type=float, default=DefaultOpenArmConfig.SAFE_GRIPPER_OPEN_CMD)
    parser.add_argument("--gripper-close-cmd", type=float, default=DefaultOpenArmConfig.SAFE_GRIPPER_CLOSE_CMD)
    parser.add_argument("--print-raw", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Only auto-detect and print left/right 3D mouse event paths, then exit.",
    )
    parser.add_argument(
        "--joint-target-dir",
        default=str(DEFAULT_JOINT_TARGET_DIR),
        help="Directory where bimanual joint snapshots (joint_target_NNN.npy) are written.",
    )
    parser.add_argument(
        "--record-key",
        default="r",
        help="Keyboard key (single char) that captures the current joint state to a new .npy file.",
    )
    return parser.parse_args()


def check_server(server_url: str) -> None:
    resp = requests.post(server_url.rstrip("/") + "/getstate", timeout=2.0)
    resp.raise_for_status()


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


def auto_detect_device(exclude_paths=None):
    exclude_paths = set(exclude_paths or [])
    candidates = []
    for path in list_devices():
        if path in exclude_paths:
            continue
        try:
            dev = InputDevice(path)
        except (PermissionError, OSError):
            continue
        name = (dev.name or "").lower()
        name_hit = any(k in name for k in ("3dconnexion", "spacemouse", "space mouse", "spacenavigator"))
        axes_hit = has_3dx_axes(dev)
        if name_hit or axes_hit:
            score = (100 if name_hit else 0) + (10 if axes_hit else 0)
            candidates.append((score, path, dev.name))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def open_device(event_path: str, exclude_paths=None):
    resolved = event_path
    detected_name = None
    if str(event_path).strip().lower() == "auto":
        resolved, detected_name = auto_detect_device(exclude_paths=exclude_paths)
        if resolved is None:
            raise RuntimeError("SpaceMouse auto-detect failed.")
    dev = InputDevice(resolved)
    try:
        dev.grab()
    except OSError:
        pass
    flags = fcntl.fcntl(dev.fd, fcntl.F_GETFL)
    fcntl.fcntl(dev.fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    print(f"[3DX] using {resolved} ({detected_name or dev.name})")
    return dev


def detect_dual_devices(left_event: str, right_event: str):
    left_dev = open_device(left_event)
    right_dev = open_device(right_event, exclude_paths={left_dev.path})
    print("=== Auto-Detected 3D Mouse Ports ===")
    print(f"left_spacemouse_event_path={left_dev.path}")
    print(f"right_spacemouse_event_path={right_dev.path}")
    print("Suggested train flags:")
    print(
        "  "
        f"--left_spacemouse_event_path={left_dev.path} "
        f"--right_spacemouse_event_path={right_dev.path}"
    )
    return left_dev, right_dev


def parse_axis_spec(spec: str):
    value = spec.strip().lower()
    if value.startswith("-"):
        return value[1:], -1.0
    return value, 1.0


def device_to_ee_translation(axes, args):
    def get_norm(axis_spec):
        name, sign = parse_axis_spec(axis_spec)
        raw = axes.get(name, 0.0)
        return sign * np.clip(raw / args.trans_denom, -1.0, 1.0)

    return get_norm(args.ee_x), get_norm(args.ee_y), get_norm(args.ee_z)


def apply_deadzone(value: float, threshold: float) -> float:
    return 0.0 if abs(value) < threshold else value


def get_rotation_action(axes, args):
    return (
        np.clip(axes["rx"] / args.rot_denom, -1.0, 1.0),
        np.clip(-axes["ry"] / args.rot_denom, -1.0, 1.0),
        np.clip(-axes["rz"] / args.rot_denom, -1.0, 1.0),
    )


def has_motion_input(axes, args):
    dx, dy, dz = device_to_ee_translation(axes, args)
    if abs(dx) > args.deadzone or abs(dy) > args.deadzone or abs(dz) > args.deadzone:
        return True
    rx, ry, rz = get_rotation_action(axes, args)
    return abs(rx) > args.rot_deadzone or abs(ry) > args.rot_deadzone or abs(rz) > args.rot_deadzone


def poll_device(dev, state, shared):
    got_any = False
    state["gripper_toggle_changed"] = False
    while True:
        try:
            ev = dev.read_one()
        except (BlockingIOError, OSError):
            return got_any
        if ev is None:
            return got_any
        got_any = True
        if ev.type in (ecodes.EV_ABS, ecodes.EV_REL):
            axis = AXIS_CODES.get(ev.code)
            if axis is not None:
                if ev.type == ecodes.EV_REL:
                    state["axes"][axis] += float(ev.value)
                else:
                    state["axes"][axis] = float(ev.value)
        elif ev.type == ecodes.EV_KEY and int(ev.value) == 1:
            if ev.code in (ecodes.BTN_0, ecodes.BTN_LEFT):
                shared["intervention_mode"] = not shared["intervention_mode"]
                print(f"[3DX:{state['label']}] intervention={'ON' if shared['intervention_mode'] else 'OFF'}")
            elif ev.code in (ecodes.BTN_1, ecodes.BTN_RIGHT):
                state["gripper_close"] = not state["gripper_close"]
                state["gripper_toggle_changed"] = True
                print(f"[3DX:{state['label']}] gripper={'CLOSE' if state['gripper_close'] else 'OPEN'}")


def build_arm_action(state, args):
    dx, dy, dz = device_to_ee_translation(state["axes"], args)
    rx, ry, rz = get_rotation_action(state["axes"], args)
    return np.array(
        [
            apply_deadzone(dx, args.deadzone),
            apply_deadzone(dy, args.deadzone),
            apply_deadzone(dz, args.deadzone),
            apply_deadzone(rx, args.rot_deadzone),
            apply_deadzone(ry, args.rot_deadzone),
            apply_deadzone(rz, args.rot_deadzone),
            1.0 if state["gripper_close"] else -1.0,
        ],
        dtype=np.float32,
    )


def start_servo(args, env, target_pose_ref):
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
    resp = requests.post(args.server_url.rstrip("/") + "/servo/start", json=payload, timeout=3.0)
    resp.raise_for_status()


def stop_servo(args):
    try:
        requests.post(args.server_url.rstrip("/") + "/servo/stop", json={}, timeout=2.0)
    except Exception:
        pass


def update_servo_target(args, env, target_pose_ref, action, gripper_binary_state):
    desired_gripper = [float(x) for x in env.curr_gripper_pos]
    for arm_idx, raw_val in enumerate((float(action[6]), float(action[13]))):
        gripper_binary_state[arm_idx] = apply_binary_gripper_logic(
            raw_val=raw_val,
            prev_binary_state=gripper_binary_state[arm_idx],
            open_threshold=env.gripper_open_threshold,
            close_threshold=env.gripper_close_threshold,
        )
        desired_gripper[arm_idx] = binary_gripper_state_to_cmd(
            gripper_binary_state[arm_idx],
            args.gripper_open_cmd,
            args.gripper_close_cmd,
        )
    payload = {
        "arr": np.asarray(target_pose_ref, dtype=np.float32).tolist(),
        "gripper": desired_gripper,
    }
    resp = requests.post(args.server_url.rstrip("/") + "/servo/target", json=payload, timeout=2.0)
    resp.raise_for_status()


def build_hold_action(left_state, right_state):
    return np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0 if left_state["gripper_close"] else -1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0 if right_state["gripper_close"] else -1.0,
        ],
        dtype=np.float32,
    )


def update_target_pose_ref(target_pose_ref, action, action_scale):
    updated = np.array(target_pose_ref, copy=True)
    for arm_idx in (0, 1):
        arm_action = action[arm_idx * 7 : (arm_idx + 1) * 7]
        updated[arm_idx, :3] += arm_action[:3] * action_scale[0]
        rot_curr = Rotation.from_quat(updated[arm_idx, 3:])
        rot_delta = Rotation.from_euler("xyz", arm_action[3:6] * action_scale[1])
        updated[arm_idx, 3:] = (rot_delta * rot_curr).as_quat()
    return updated


def make_device_state(label):
    return {
        "label": label,
        "axes": {"x": 0.0, "y": 0.0, "z": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        "gripper_close": False,
        "gripper_toggle_changed": False,
    }


def _next_target_index(target_dir: Path) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    next_idx = 1
    for entry in target_dir.iterdir():
        for pattern in (JOINT_TARGET_PATTERN, EE_TARGET_PATTERN):
            match = pattern.match(entry.name)
            if match:
                next_idx = max(next_idx, int(match.group(1)) + 1)
    return next_idx


def _next_joint_target_path(target_dir: Path) -> Path:
    return target_dir / f"joint_target_{_next_target_index(target_dir):03d}.npy"


def start_joint_recorder(env, target_dir: str, record_key: str, env_lock):
    from pynput import keyboard

    record_key = str(record_key).strip().lower()
    if len(record_key) != 1:
        raise ValueError(f"--record-key must be a single character, got: {record_key!r}")

    target_dir = Path(os.path.abspath(os.fspath(target_dir)))
    target_dir.mkdir(parents=True, exist_ok=True)

    def _capture():
        try:
            with env_lock:
                env.refresh_obs()
                joint_state = np.asarray(env.q, dtype=np.float32).copy()
                ee_state = np.asarray(env.currpos, dtype=np.float32).copy()
            if joint_state.shape != (2, 7):
                raise ValueError(f"expected env.q shape (2, 7), got {joint_state.shape}")
            if ee_state.shape != (2, 7):
                raise ValueError(f"expected env.currpos shape (2, 7), got {ee_state.shape}")
            idx = _next_target_index(target_dir)
            joint_path = target_dir / f"joint_target_{idx:03d}.npy"
            ee_path = target_dir / f"ee_target_{idx:03d}.npy"
            np.save(joint_path, joint_state)
            np.save(ee_path, ee_state)
            print(
                f"[JointRecord] saved {joint_path} | "
                f"left={np.round(joint_state[0], 3).tolist()} "
                f"right={np.round(joint_state[1], 3).tolist()}"
            )
            print(
                f"[JointRecord] saved {ee_path} | "
                f"left_ee={np.round(ee_state[0], 3).tolist()} "
                f"right_ee={np.round(ee_state[1], 3).tolist()}"
            )
        except Exception as exc:
            print(f"[JointRecord] failed to record joint/ee state: {exc}")

    def on_press(key):
        try:
            if hasattr(key, "char") and key.char is not None and key.char.lower() == record_key:
                _capture()
        except Exception:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()
    print(
        f"[JointRecord] press <{record_key.upper()}> to snapshot bimanual joint + EE state "
        f"(dir={target_dir})."
    )
    return listener


def main():
    args = parse_args()
    check_server(args.server_url)

    left_dev, right_dev = detect_dual_devices(args.left_event, args.right_event)
    if args.detect_only:
        try:
            left_dev.ungrab()
        except OSError:
            pass
        try:
            right_dev.ungrab()
        except OSError:
            pass
        return 0

    cfg = DefaultOpenArmConfig()
    cfg.SERVER_URL = args.server_url
    cfg.STATION_KEEP_INACTIVE_ARM = False
    env = OpenArmEnv(
        hz=int(args.hz),
        fake_env=False,
        save_video=False,
        config=cfg,
        max_episode_length=10**9,
    )
    env.reset()

    env_lock = threading.Lock()
    start_joint_recorder(
        env,
        target_dir=args.joint_target_dir,
        record_key=args.record_key,
        env_lock=env_lock,
    )

    left_state = make_device_state("left")
    right_state = make_device_state("right")
    shared_state = {"intervention_mode": False}
    target_pose_ref = env.currpos.copy()
    servo_running = False
    idle_hold_sent = False
    gripper_binary_state = np.zeros((2,), dtype=np.int32)
    step = 0

    print("=== Dual 3D Mouse Teleop Test ===")
    print("Left device -> left arm, right device -> right arm")
    print("Button0 on either mouse toggles global intervention, Button1 toggles that mouse gripper")
    print(f"Press <{args.record_key.upper()}> to snapshot bimanual joint state for reward shaping")
    print(f"Configured control_hz={args.control_hz}, servo_hz={args.servo_hz}")

    try:
        while True:
            if args.max_steps > 0 and step >= args.max_steps:
                break

            left_got = poll_device(left_dev, left_state, shared_state)
            right_got = poll_device(right_dev, right_state, shared_state)
            if not left_got:
                for key in left_state["axes"]:
                    left_state["axes"][key] = 0.0
            if not right_got:
                for key in right_state["axes"]:
                    right_state["axes"][key] = 0.0

            any_intervention = shared_state["intervention_mode"]
            left_active = any_intervention and (
                has_motion_input(left_state["axes"], args) or left_state["gripper_toggle_changed"]
            )
            right_active = any_intervention and (
                has_motion_input(right_state["axes"], args) or right_state["gripper_toggle_changed"]
            )

            if not any_intervention:
                if servo_running:
                    with env_lock:
                        env.refresh_obs()
                        target_pose_ref = env.currpos.copy()
                    stop_servo(args)
                    servo_running = False
                    idle_hold_sent = False
                time.sleep(1.0 / args.control_hz)
                continue

            if not servo_running:
                with env_lock:
                    env.refresh_obs()
                    target_pose_ref = env.currpos.copy()
                start_servo(args, env, target_pose_ref)
                servo_running = True
                idle_hold_sent = False

            if not left_active and not right_active:
                hold_action = build_hold_action(left_state, right_state)
                if not idle_hold_sent or left_state["gripper_toggle_changed"] or right_state["gripper_toggle_changed"]:
                    with env_lock:
                        env.refresh_obs()
                        target_pose_ref = env.currpos.copy()
                        update_servo_target(args, env, target_pose_ref, hold_action, gripper_binary_state)
                    idle_hold_sent = True
                time.sleep(1.0 / args.control_hz)
                continue

            left_action = build_arm_action(left_state, args)
            right_action = build_arm_action(right_state, args)
            if not left_active:
                left_action[:6] = 0.0
            if not right_active:
                right_action[:6] = 0.0
            action = np.concatenate([left_action, right_action]).astype(np.float32)

            idle_hold_sent = False
            target_pose_ref = update_target_pose_ref(target_pose_ref, action, env.action_scale)

            with env_lock:
                update_servo_target(args, env, target_pose_ref, action, gripper_binary_state)
                obs = env.refresh_obs()
            if args.print_raw:
                print(
                    f"[Dual3DX] left_axes={left_state['axes']} right_axes={right_state['axes']} "
                    f"action={np.round(action, 4).tolist()} "
                    f"tcp={np.round(obs['state']['tcp_pose'], 4).tolist()} "
                    f"state_shape={np.asarray(obs['state']['tcp_pose']).shape}"
                )

            step += 1
            time.sleep(1.0 / args.control_hz)
    finally:
        stop_servo(args)
        try:
            left_dev.ungrab()
        except OSError:
            pass
        try:
            right_dev.ungrab()
        except OSError:
            pass
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
