#!/usr/bin/env python3
"""Inspect and stream SpaceMouse/3Dconnexion evdev events."""
import argparse
import fcntl
import os
import time


def parse_args():
    parser = argparse.ArgumentParser(description="3D mouse evdev connectivity test.")
    parser.add_argument("--event", default="auto", help="Event path or 'auto'.")
    parser.add_argument("--duration", type=float, default=0.0, help="0 means until Ctrl+C.")
    parser.add_argument("--dual", action="store_true", help="Auto-detect two devices.")
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


def detect_devices(count):
    InputDevice, ecodes, list_devices = import_evdev()
    candidates = []
    for path in list_devices():
        try:
            dev = InputDevice(path)
        except (PermissionError, OSError):
            continue
        name = (dev.name or "").lower()
        name_hit = any(
            token in name
            for token in (
                "3dconnexion",
                "spacemouse",
                "space mouse",
                "spacenavigator",
                "space navigator",
            )
        )
        axes_hit = has_3dx_axes(dev, ecodes)
        if name_hit or axes_hit:
            score = (100 if name_hit else 0) + (10 if axes_hit else 0)
            candidates.append((score, path, dev.name))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = []
    for _score, path, _name in candidates:
        if path not in selected:
            selected.append(path)
        if len(selected) == count:
            break
    return selected


def open_device(event_path):
    InputDevice, _ecodes, _list_devices = import_evdev()
    dev = InputDevice(event_path)
    try:
        flags = fcntl.fcntl(dev.fd, fcntl.F_GETFL)
        fcntl.fcntl(dev.fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    except OSError:
        pass
    print("=== Event Device Info ===")
    print(f"path    : {dev.path}")
    print(f"name    : {dev.name}")
    print(f"phys    : {dev.phys}")
    print(f"uniq    : {dev.uniq}")
    print(f"vendor  : 0x{dev.info.vendor:04x}")
    print(f"product : 0x{dev.info.product:04x}")
    print()
    return dev


def event_name(event, ecodes):
    axis_names = {
        (ecodes.EV_ABS, ecodes.ABS_X): "ABS_X",
        (ecodes.EV_ABS, ecodes.ABS_Y): "ABS_Y",
        (ecodes.EV_ABS, ecodes.ABS_Z): "ABS_Z",
        (ecodes.EV_ABS, ecodes.ABS_RX): "ABS_RX",
        (ecodes.EV_ABS, ecodes.ABS_RY): "ABS_RY",
        (ecodes.EV_ABS, ecodes.ABS_RZ): "ABS_RZ",
        (ecodes.EV_REL, ecodes.REL_X): "REL_X",
        (ecodes.EV_REL, ecodes.REL_Y): "REL_Y",
        (ecodes.EV_REL, ecodes.REL_Z): "REL_Z",
        (ecodes.EV_REL, ecodes.REL_RX): "REL_RX",
        (ecodes.EV_REL, ecodes.REL_RY): "REL_RY",
        (ecodes.EV_REL, ecodes.REL_RZ): "REL_RZ",
        (ecodes.EV_KEY, ecodes.BTN_0): "BTN_0",
        (ecodes.EV_KEY, ecodes.BTN_1): "BTN_1",
        (ecodes.EV_KEY, ecodes.BTN_LEFT): "BTN_LEFT",
        (ecodes.EV_KEY, ecodes.BTN_RIGHT): "BTN_RIGHT",
    }
    return axis_names.get((event.type, event.code), f"type={event.type} code={event.code}")


def listen(devices, duration):
    _InputDevice, ecodes, _list_devices = import_evdev()
    start = time.time()
    print("Move the 3D mouse or press buttons. Ctrl+C exits.")
    for dev in devices:
        try:
            dev.grab()
        except OSError:
            print(f"[WARN] Could not grab {dev.path}; reading without exclusive grab.")

    try:
        while True:
            if duration > 0 and time.time() - start >= duration:
                break
            got_event = False
            for dev in devices:
                try:
                    event = dev.read_one()
                except BlockingIOError:
                    event = None
                except OSError:
                    event = None
                if event is None:
                    continue
                got_event = True
                if event.type in (ecodes.EV_ABS, ecodes.EV_REL, ecodes.EV_KEY):
                    print(
                        f"{time.strftime('%H:%M:%S')} {dev.path} "
                        f"{event_name(event, ecodes)} value={event.value}"
                    )
            if not got_event:
                time.sleep(0.01)
    finally:
        for dev in devices:
            try:
                dev.ungrab()
            except OSError:
                pass


def main():
    args = parse_args()
    if args.dual:
        paths = detect_devices(2)
        if len(paths) < 2:
            raise SystemExit(f"Expected two 3D mouse devices, detected: {paths}")
    elif args.event == "auto":
        paths = detect_devices(1)
        if not paths:
            raise SystemExit("No likely 3D mouse event device detected.")
    else:
        paths = [args.event]

    print("Detected paths:", ", ".join(paths))
    devices = [open_device(path) for path in paths]
    listen(devices, args.duration)


if __name__ == "__main__":
    main()
