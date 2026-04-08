#!/usr/bin/env python3
"""
USB camera connection test for record_demo_rl.py camera config.

Aligned with moqi_workspace/pyroki/record_demo_rl.py:
- CAMERA_CONFIGS contains head USB camera device path /dev/video10
- Expected setup: width=640, height=480, fps=30

This script checks:
1) Device open status
2) Frame grabbing stability
3) Actual camera properties reported by OpenCV
"""

import argparse
import glob
import os
import time

import cv2


# From moqi_workspace/pyroki/record_demo_rl.py CAMERA_CONFIGS:
# ("/dev/video10", 640, 480, 30)  # head (USB Camera Device Path)
HEAD_USB_CAMERA_DEVICE = "/dev/video10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test USB camera connectivity (record_demo_rl style)")
    parser.add_argument("--device", default=HEAD_USB_CAMERA_DEVICE, help=f"USB camera device index/path (default head: {HEAD_USB_CAMERA_DEVICE})")
    parser.add_argument("--head-only", action="store_true", help=f"Only test head USB camera device {HEAD_USB_CAMERA_DEVICE}")
    parser.add_argument("--auto-scan", action="store_true", help="Auto scan camera index and pick first valid device")
    parser.add_argument("--scan-all", action="store_true", help="Scan and list all valid camera indices, then exit")
    parser.add_argument("--scan-start", type=int, default=0, help="Auto-scan start index")
    parser.add_argument("--scan-end", type=int, default=30, help="Auto-scan end index (inclusive)")
    parser.add_argument("--width", type=int, default=640, help="Requested width")
    parser.add_argument("--height", type=int, default=480, help="Requested height")
    parser.add_argument("--fps", type=int, default=30, help="Requested FPS")
    parser.add_argument("--duration", type=float, default=5.0, help="Test duration in seconds")
    parser.add_argument("--show", action="store_true", help="Show live preview window")
    parser.add_argument(
        "--auto-head-port",
        action="store_true",
        help="Auto-detect top USB camera from /dev/video* (recommended when /dev/video10 is missing)",
    )
    return parser.parse_args()


def find_first_valid_camera(start: int, end: int) -> int | None:
    """
    Scan [start, end] and return the first camera index that:
    1) opens successfully
    2) can read at least one frame
    """
    print(f"[SCAN] Scanning camera index from {start} to {end} ...")
    for idx in range(start, end + 1):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue

        ok = False
        for _ in range(10):
            ret, frame = cap.read()
            if ret and frame is not None:
                ok = True
                break
            time.sleep(0.02)
        cap.release()

        if ok:
            print(f"[SCAN] Found valid camera at index {idx}")
            return idx

    print("[SCAN] No valid camera found in scan range.")
    return None


def find_all_valid_cameras(start: int, end: int) -> list[int]:
    """
    Scan [start, end] and return all camera indices that:
    1) open successfully
    2) can read at least one frame
    """
    print(f"[SCAN] Scanning all camera index from {start} to {end} ...")
    found = []
    for idx in range(start, end + 1):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue

        ok = False
        for _ in range(10):
            ret, frame = cap.read()
            if ret and frame is not None:
                ok = True
                break
            time.sleep(0.02)
        cap.release()

        if ok:
            found.append(idx)
            print(f"[SCAN] valid camera index: {idx}")

    if not found:
        print("[SCAN] No valid camera found in scan range.")
    return found


def _read_v4l2_name(dev_path: str) -> str:
    """Read Linux v4l2 device name from sysfs if possible."""
    try:
        base = os.path.basename(dev_path)  # videoX
        name_path = f"/sys/class/video4linux/{base}/name"
        with open(name_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def list_video_device_paths() -> list[str]:
    """List /dev/video* in numeric order."""
    paths = glob.glob("/dev/video*")
    def sort_key(p: str):
        b = os.path.basename(p)
        try:
            return int(b.replace("video", ""))
        except Exception:
            return 10**9
    return sorted(paths, key=sort_key)


def probe_camera_device(dev_path: str, width: int, height: int, fps: int) -> tuple[bool, int, int, float]:
    """Open + read a few frames; return (ok, actual_w, actual_h, actual_fps)."""
    cap = cv2.VideoCapture(dev_path)
    if not cap.isOpened():
        cap.release()
        return False, 0, 0, 0.0

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS))

    ok = False
    for _ in range(15):
        ret, frame = cap.read()
        if ret and frame is not None:
            ok = True
            break
        time.sleep(0.02)

    cap.release()
    return ok, actual_w, actual_h, actual_fps


def find_head_usb_camera_device(width: int, height: int, fps: int) -> str | None:
    """
    Auto-detect a likely head USB camera device path.
    Heuristics:
    1) must be readable and can stream frames
    2) prefer names containing usb/camera/uvc/webcam
    3) prefer closer resolution match to requested width/height
    """
    paths = list_video_device_paths()
    if not paths:
        print("[AUTO] No /dev/video* found.")
        return None

    candidates = []
    print("[AUTO] Scanning /dev/video* for top USB camera ...")
    for p in paths:
        if not os.access(p, os.R_OK | os.W_OK):
            continue

        ok, aw, ah, afps = probe_camera_device(p, width, height, fps)
        if not ok:
            continue

        name = _read_v4l2_name(p).lower()
        name_score = 0
        for kw in ("usb", "camera", "uvc", "webcam", "hd"):
            if kw in name:
                name_score += 10

        # Smaller gap to requested resolution gets higher score
        res_gap = abs(aw - width) + abs(ah - height)
        res_score = max(0, 2000 - res_gap) / 100.0
        score = name_score + res_score

        candidates.append((score, p, name, aw, ah, afps))
        print(f"[AUTO] candidate {p} name='{name or 'unknown'}' actual={aw}x{ah}@{afps:.1f} score={score:.2f}")

    if not candidates:
        print("[AUTO] No readable streaming /dev/video* candidates found.")
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0]
    print(f"[AUTO] selected head camera: {best[1]} (name='{best[2] or 'unknown'}', actual={best[3]}x{best[4]}@{best[5]:.1f})")
    return best[1]


def main() -> int:
    args = parse_args()

    if args.head_only:
        # Force head camera index from record_demo_rl.py
        args.device = HEAD_USB_CAMERA_DEVICE
        # In head-only mode, scan options are ignored.
        args.auto_scan = False
        args.scan_all = False

    if args.scan_all:
        all_found = find_all_valid_cameras(args.scan_start, args.scan_end)
        if all_found:
            print(f"[SCAN] All valid indices: {all_found}")
            return 0
        return 3

    device = args.device
    if args.auto_scan:
        found = find_first_valid_camera(args.scan_start, args.scan_end)
        if found is None:
            print("[FAIL] Auto-scan did not find any readable USB camera.")
            return 3
        device = found

    # Auto-detect top camera path explicitly, or fallback when configured path is missing.
    if args.auto_head_port:
        auto_dev = find_head_usb_camera_device(args.width, args.height, args.fps)
        if auto_dev is None:
            print("[FAIL] Auto head camera detection failed.")
            return 6
        device = auto_dev

    print("=== USB Camera Connection Test ===")
    print(
        f"Request: device={device}, width={args.width}, "
        f"height={args.height}, fps={args.fps}, duration={args.duration}s"
    )

    try:
        # If user passes numeric string (e.g. "2"), treat as index
        cap_target = int(device) if isinstance(device, str) and device.isdigit() else device
    except Exception:
        cap_target = device

    if isinstance(cap_target, str) and cap_target.startswith("/dev/video"):
        if not os.path.exists(cap_target):
            print(f"[WARN] Device path not found: {cap_target}")
            print("[INFO] Trying auto-detect head camera from /dev/video* ...")
            auto_dev = find_head_usb_camera_device(args.width, args.height, args.fps)
            if auto_dev is None:
                print(f"[FAIL] Device path not found and auto-detect failed: {cap_target}")
                return 4
            cap_target = auto_dev
            device = auto_dev
            print(f"[INFO] Fallback to auto-detected device: {device}")
        if not os.access(cap_target, os.R_OK | os.W_OK):
            print(f"[FAIL] No permission to access {cap_target}")
            print("Tips:")
            print("- Add current user to 'video' group: sudo usermod -aG video $USER")
            print("- Re-login (or reboot) to refresh groups")
            print("- Or run test with elevated permission")
            return 5

    cap = cv2.VideoCapture(cap_target)
    if not cap.isOpened():
        print(f"[FAIL] Could not open USB camera index {device}")
        print("Tips:")
        print("- Check device index (try 0/1/2/... or v4l2-ctl --list-devices)")
        print("- Check permissions for /dev/video*")
        return 1

    # Mirror record_demo_rl.py configuration.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Opened device {device}")
    print(f"[INFO] Actual properties: width={actual_w}, height={actual_h}, fps={actual_fps:.2f}")

    start = time.time()
    last_print = start
    frame_count = 0
    fail_count = 0

    try:
        while time.time() - start < args.duration:
            ret, frame = cap.read()
            if not ret or frame is None:
                fail_count += 1
                time.sleep(0.01)
                continue

            frame_count += 1
            if args.show:
                cv2.imshow(f"USB Camera {device}", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            now = time.time()
            if now - last_print >= 1.0:
                elapsed = now - start
                fps_now = frame_count / max(elapsed, 1e-6)
                print(f"[RUN] frames={frame_count}, read_fail={fail_count}, avg_fps={fps_now:.2f}")
                last_print = now
    finally:
        cap.release()
        cv2.destroyAllWindows()

    elapsed = time.time() - start
    avg_fps = frame_count / max(elapsed, 1e-6)
    print("\n=== Result ===")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Frames: {frame_count}")
    print(f"Read fail count: {fail_count}")
    print(f"Average FPS: {avg_fps:.2f}")

    if frame_count <= 0:
        print("[FAIL] No valid frames received.")
        return 2

    print("[PASS] USB camera is connected and streaming frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
