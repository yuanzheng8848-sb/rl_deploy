#!/usr/bin/env python3
"""
Test 3D mouse connectivity on Linux.

This script checks:
1) Whether the given event device (e.g. /dev/input/event16) is accessible and prints metadata.
2) Reads motion/button events directly from that device via evdev (no spacenavd required).

Note: python-spnav uses PyCObject_AsVoidPtr, which was removed in Python 3, so it fails on
Python 3.12+. This script uses evdev only, which works with spacenavd or without it.

Usage:
    python test_3dx.py
    python test_3dx.py --event /dev/input/event16 --duration 20
"""

import argparse
import fcntl
import os
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D mouse connectivity tester")
    parser.add_argument(
        "--event",
        default="auto",
        help="Input event device path (e.g. /dev/input/event16) or 'auto' (default)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Listen duration in seconds (0 means until Ctrl+C)",
    )
    parser.add_argument(
        "--no-auto-fallback",
        action="store_true",
        help="Disable auto-detection fallback when --event cannot be opened",
    )
    return parser.parse_args()


def _has_3dx_axes(dev) -> bool:
    """Heuristic: device exposes translational + rotational axes (ABS or REL)."""
    try:
        from evdev import ecodes
    except ImportError:
        return False

    caps = dev.capabilities(absinfo=False)
    abs_codes = set(caps.get(ecodes.EV_ABS, []))
    rel_codes = set(caps.get(ecodes.EV_REL, []))

    trans_abs = {ecodes.ABS_X, ecodes.ABS_Y, ecodes.ABS_Z}
    rot_abs = {ecodes.ABS_RX, ecodes.ABS_RY, ecodes.ABS_RZ}
    trans_rel = {ecodes.REL_X, ecodes.REL_Y, ecodes.REL_Z}
    rot_rel = {ecodes.REL_RX, ecodes.REL_RY, ecodes.REL_RZ}

    abs_ok = trans_abs.issubset(abs_codes) and rot_abs.issubset(abs_codes)
    rel_ok = trans_rel.issubset(rel_codes) and rot_rel.issubset(rel_codes)
    return abs_ok or rel_ok


def auto_detect_event_device(verbose: bool = True):
    """Auto-detect likely 3D mouse event device. Returns InputDevice or None."""
    try:
        from evdev import InputDevice, list_devices
    except ImportError:
        print("[ERROR] Missing dependency: evdev")
        print("Install with: pip install evdev")
        return None

    candidates = []
    for path in list_devices():
        try:
            dev = InputDevice(path)
        except PermissionError:
            if verbose:
                print(f"[WARN] Permission denied: {path}")
            continue
        except OSError:
            continue

        name = (dev.name or "").lower()
        keywords = ("3dconnexion", "spacemouse", "space mouse", "spacenavigator", "space navigator")
        name_hit = any(k in name for k in keywords)
        axes_hit = _has_3dx_axes(dev)
        if name_hit or axes_hit:
            score = (100 if name_hit else 0) + (10 if axes_hit else 0)
            candidates.append((score, dev))

    if not candidates:
        if verbose:
            print("[ERROR] Auto-detect failed: no likely 3D mouse event device found.")
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    dev = candidates[0][1]

    if verbose:
        print(f"[INFO] Auto-detected event device: {dev.path} ({dev.name})")
    return dev


def inspect_event_device(event_path: str):
    """Open event device, print info, return the device or None on error."""
    try:
        from evdev import InputDevice
    except ImportError:
        print("[ERROR] Missing dependency: evdev")
        print("Install with: pip install evdev")
        return None

    try:
        dev = InputDevice(event_path)
    except FileNotFoundError:
        print(f"[ERROR] Event device not found: {event_path}")
        return None
    except PermissionError:
        print(f"[ERROR] Permission denied when opening: {event_path}")
        print("Try: sudo python3 <script.py>  (or use your conda python absolute path)")
        return None
    except OSError as exc:
        print(f"[ERROR] Failed to open event device {event_path}: {exc}")
        return None

    print("=== Event Device Info ===")
    print(f"path      : {dev.path}")
    print(f"name      : {dev.name}")
    print(f"phys      : {dev.phys}")
    print(f"uniq      : {dev.uniq}")
    print(f"vendor    : 0x{dev.info.vendor:04x}")
    print(f"product   : 0x{dev.info.product:04x}")
    print(f"version   : 0x{dev.info.version:04x}")
    print()
    return dev


def listen_evdev(dev, duration: float) -> int:
    """Read input events from the given evdev device (3D mouse motion/buttons)."""
    from evdev import ecodes

    # Map (type, code) -> short name for 3D mouse axes (ABS or REL)
    motion_codes = [
        (ecodes.EV_ABS, ecodes.ABS_X, "X"),
        (ecodes.EV_ABS, ecodes.ABS_Y, "Y"),
        (ecodes.EV_ABS, ecodes.ABS_Z, "Z"),
        (ecodes.EV_ABS, ecodes.ABS_RX, "RX"),
        (ecodes.EV_ABS, ecodes.ABS_RY, "RY"),
        (ecodes.EV_ABS, ecodes.ABS_RZ, "RZ"),
    ]
    rel_codes = [
        (ecodes.EV_REL, ecodes.REL_X, "X"),
        (ecodes.EV_REL, ecodes.REL_Y, "Y"),
        (ecodes.EV_REL, ecodes.REL_Z, "Z"),
        (ecodes.EV_REL, ecodes.REL_RX, "RX"),
        (ecodes.EV_REL, ecodes.REL_RY, "RY"),
        (ecodes.EV_REL, ecodes.REL_RZ, "RZ"),
    ]

    print("=== 3D Mouse Event Stream (evdev) ===")
    print("Move or press the 3D mouse to see events. Press Ctrl+C to stop.")
    if duration > 0:
        print(f"Auto-stop after {duration:.1f}s")
    print()

    start = time.time()
    event_count = 0
    last_motion = {}

    try:
        dev.grab()
    except (IOError, OSError):
        print("[WARN] Could not grab device (another process may be using it). Reading without grab.")

    # Non-blocking so we can respect --duration and Ctrl+C
    try:
        flags = fcntl.fcntl(dev.fd, fcntl.F_GETFL)
        fcntl.fcntl(dev.fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    except (IOError, OSError):
        pass

    try:
        while True:
            if duration > 0 and (time.time() - start) >= duration:
                break

            try:
                ev = dev.read_one()
            except BlockingIOError:
                ev = None
            except (IOError, OSError):
                break

            if ev is None:
                time.sleep(0.01)
                continue

            event_count += 1
            ts = time.strftime("%H:%M:%S")

            if ev.type == ecodes.EV_ABS:
                for t, c, name in motion_codes:
                    if ev.code == c:
                        last_motion[name] = ev.value
                        break
                if len(last_motion) >= 6:
                    tx = last_motion.get("X", 0), last_motion.get("Y", 0), last_motion.get("Z", 0)
                    rot = last_motion.get("RX", 0), last_motion.get("RY", 0), last_motion.get("RZ", 0)
                    print(f"[{ts}] MOTION trans=({tx[0]:+5d}, {tx[1]:+5d}, {tx[2]:+5d}) rot=({rot[0]:+5d}, {rot[1]:+5d}, {rot[2]:+5d})")
                    last_motion.clear()
            elif ev.type == ecodes.EV_REL:
                for t, c, name in rel_codes:
                    if ev.code == c:
                        last_motion[name] = last_motion.get(name, 0) + ev.value
                        break
                if len(last_motion) >= 6:
                    tx = last_motion.get("X", 0), last_motion.get("Y", 0), last_motion.get("Z", 0)
                    rot = last_motion.get("RX", 0), last_motion.get("RY", 0), last_motion.get("RZ", 0)
                    print(f"[{ts}] MOTION trans=({tx[0]:+5d}, {tx[1]:+5d}, {tx[2]:+5d}) rot=({rot[0]:+5d}, {rot[1]:+5d}, {rot[2]:+5d})")
                    last_motion.clear()
            elif ev.type == ecodes.EV_KEY:
                key_name = ecodes.KEY.get(ev.code, ev.code)
                state = "PRESS" if ev.value else "RELEASE"
                print(f"[{ts}] BUTTON {state} {key_name}")
            else:
                print(f"[{ts}] type={ev.type} code={ev.code} value={ev.value}")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        try:
            dev.ungrab()
        except (IOError, OSError):
            pass

    print()
    print(f"Total events: {event_count}")
    if event_count == 0:
        print("No events captured. Move or press the 3D mouse, or check permissions.")
    return 0


def main() -> int:
    args = parse_args()

    if args.event.strip().lower() == "auto":
        dev = auto_detect_event_device(verbose=True)
    else:
        dev = inspect_event_device(args.event)
        if dev is None and not args.no_auto_fallback:
            print("[INFO] Falling back to auto-detect...")
            dev = auto_detect_event_device(verbose=True)

    if dev is None:
        return 1

    return listen_evdev(dev, args.duration)


if __name__ == "__main__":
    sys.exit(main())
