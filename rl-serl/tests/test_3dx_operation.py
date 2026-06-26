#!/usr/bin/env python3
"""Drive OpenArm with dual SpaceMouse through the server /servo/* API.

Start the hardware server first:
    cd rl-serl/rl_robot_infra/robot_servers
    python openarm_server.py
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


RL_SERL_ROOT = Path(__file__).resolve().parents[1]
for path in (
    RL_SERL_ROOT / "rl_robot_infra",
    RL_SERL_ROOT / "examples",
):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

from openarm_env.envs.openarm_env import (  # noqa: E402
    DefaultOpenArmConfig,
    OpenArmEnv,
    apply_binary_gripper_logic,
    binary_gripper_state_to_cmd,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Dual SpaceMouse OpenArm servo test.")
    parser.add_argument("--server-url", default="http://127.0.0.1:5000/")
    parser.add_argument("--left-event", default="auto")
    parser.add_argument("--right-event", default="auto")
    parser.add_argument("--detect-only", action="store_true")
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--control-hz", type=float, default=80.0)
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
    parser.add_argument("--print-raw", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0, help="0 means until Ctrl+C.")
    return parser.parse_args()


def import_evdev():
    try:
        from evdev import InputDevice, ecodes, list_devices
    except ImportError as exc:
        raise SystemExit("Missing dependency: evdev. Install it in the runtime env.") from exc
    return InputDevice, ecodes, list_devices


def has_3dx_axes(dev, ecodes):
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
    InputDevice, ecodes, list_devices = import_evdev()
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
        name_hit = any(
            token in name
            for token in ("3dconnexion", "spacemouse", "space mouse", "spacenavigator")
        )
        axes_hit = has_3dx_axes(dev, ecodes)
        if name_hit or axes_hit:
            score = (100 if name_hit else 0) + (10 if axes_hit else 0)
            candidates.append((score, path, dev.name))
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def open_device(event_path, exclude_paths=None):
    InputDevice, _ecodes, _list_devices = import_evdev()
    resolved = event_path
    detected_name = None
    if str(event_path).lower() == "auto":
        resolved, detected_name = auto_detect_device(exclude_paths=exclude_paths)
        if resolved is None:
            raise RuntimeError("SpaceMouse auto-detect failed.")
    dev = InputDevice(resolved)
    try:
        dev.grab()
    except OSError:
        pass
    try:
        flags = fcntl.fcntl(dev.fd, fcntl.F_GETFL)
        fcntl.fcntl(dev.fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    except OSError:
        pass
    print(f"[3DX] using {resolved} ({detected_name or dev.name})")
    return dev


def check_server(server_url):
    resp = requests.post(server_url.rstrip("/") + "/getstate", timeout=2.0)
    resp.raise_for_status()


def make_device_state(label):
    return {
        "label": label,
        "axes": {"x": 0.0, "y": 0.0, "z": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        "gripper_close": False,
        "gripper_toggle_changed": False,
    }


def parse_axis_spec(spec):
    spec = str(spec).strip().lower()
    if spec.startswith("-"):
        return spec[1:], -1.0
    return spec, 1.0


def device_to_ee_translation(axes, args):
    def get_norm(axis_spec):
        name, sign = parse_axis_spec(axis_spec)
        return sign * np.clip(axes.get(name, 0.0) / args.trans_denom, -1.0, 1.0)

    return get_norm(args.ee_x), get_norm(args.ee_y), get_norm(args.ee_z)


def apply_deadzone(value, threshold):
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
    _InputDevice, ecodes, _list_devices = import_evdev()
    axis_codes = {
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
    got_any = False
    state["gripper_toggle_changed"] = False
    while True:
        try:
            event = dev.read_one()
        except BlockingIOError:
            return got_any
        except OSError:
            return got_any
        if event is None:
            return got_any
        got_any = True
        if event.type in (ecodes.EV_ABS, ecodes.EV_REL):
            axis = axis_codes.get(event.code)
            if axis is None:
                continue
            if event.type == ecodes.EV_REL:
                state["axes"][axis] += float(event.value)
            else:
                state["axes"][axis] = float(event.value)
        elif event.type == ecodes.EV_KEY and int(event.value) == 1:
            if event.code in (ecodes.BTN_0, ecodes.BTN_LEFT):
                shared["intervention_mode"] = not shared["intervention_mode"]
                print(
                    f"[3DX:{state['label']}] intervention="
                    f"{'ON' if shared['intervention_mode'] else 'OFF'}"
                )
            elif event.code in (ecodes.BTN_1, ecodes.BTN_RIGHT):
                state["gripper_close"] = not state["gripper_close"]
                state["gripper_toggle_changed"] = True
                print(
                    f"[3DX:{state['label']}] gripper="
                    f"{'CLOSE' if state['gripper_close'] else 'OPEN'}"
                )


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
            env.safe_gripper_open_cmd,
            env.safe_gripper_close_cmd,
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
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0 if left_state["gripper_close"] else -1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
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


def main():
    args = parse_args()
    check_server(args.server_url)
    left_dev = open_device(args.left_event)
    right_dev = open_device(args.right_event, exclude_paths={left_dev.path})

    if args.detect_only:
        print(f"left_spacemouse_event_path={left_dev.path}")
        print(f"right_spacemouse_event_path={right_dev.path}")
        return 0

    cfg = DefaultOpenArmConfig()
    cfg.SERVER_URL = args.server_url
    cfg.REALSENSE_CAMERAS = {}
    env = OpenArmEnv(
        hz=int(args.hz),
        fake_env=False,
        save_video=False,
        config=cfg,
        max_episode_length=10**9,
    )
    env.reset()

    left_state = make_device_state("left")
    right_state = make_device_state("right")
    shared = {"intervention_mode": False}
    target_pose_ref = np.array(env.currpos, copy=True)
    gripper_binary_state = np.zeros((2,), dtype=np.int32)
    servo_running = False
    idle_hold_sent = False
    step = 0

    print("Button0/left button toggles intervention. Button1/right button toggles gripper.")
    print("Ctrl+C exits.")
    try:
        while True:
            if args.max_steps > 0 and step >= args.max_steps:
                break

            left_got = poll_device(left_dev, left_state, shared)
            right_got = poll_device(right_dev, right_state, shared)
            if not left_got:
                for key in left_state["axes"]:
                    left_state["axes"][key] = 0.0
            if not right_got:
                for key in right_state["axes"]:
                    right_state["axes"][key] = 0.0

            if not shared["intervention_mode"]:
                if servo_running:
                    env.refresh_obs()
                    target_pose_ref = np.array(env.currpos, copy=True)
                    stop_servo(args)
                    servo_running = False
                    idle_hold_sent = False
                time.sleep(1.0 / args.control_hz)
                continue

            if not servo_running:
                env.refresh_obs()
                target_pose_ref = np.array(env.currpos, copy=True)
                start_servo(args, env, target_pose_ref)
                servo_running = True

            left_active = has_motion_input(left_state["axes"], args) or left_state["gripper_toggle_changed"]
            right_active = has_motion_input(right_state["axes"], args) or right_state["gripper_toggle_changed"]
            if not left_active and not right_active:
                hold_action = build_hold_action(left_state, right_state)
                if not idle_hold_sent:
                    env.refresh_obs()
                    target_pose_ref = np.array(env.currpos, copy=True)
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
            target_pose_ref = update_target_pose_ref(target_pose_ref, action, env.action_scale)
            update_servo_target(args, env, target_pose_ref, action, gripper_binary_state)
            obs = env.refresh_obs()
            if args.print_raw:
                print(
                    f"action={np.round(action, 4).tolist()} "
                    f"tcp={np.round(obs['state']['tcp_pose'], 4).tolist()}"
                )
            idle_hold_sent = False
            step += 1
            time.sleep(1.0 / args.control_hz)
    finally:
        stop_servo(args)
        for dev in (left_dev, right_dev):
            try:
                dev.ungrab()
            except OSError:
                pass
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
