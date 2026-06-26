#!/usr/bin/env python3
"""Show all local OpenArm cameras without constructing env/JAX objects."""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


RL_SERL_ROOT = Path(__file__).resolve().parents[1]
for path in (
    RL_SERL_ROOT / "rl_robot_infra",
    RL_SERL_ROOT / "examples",
):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

from openarm_env.camera.local_camera import MODEL_IMAGE_SIZE, build_cameras  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Preview local OpenArm cameras.")
    parser.add_argument("--fake-env", action="store_true", help="Use MockCamera objects.")
    parser.add_argument("--window", default="rl-serl local cameras")
    parser.add_argument("--width", type=int, default=384, help="Display width per camera.")
    parser.add_argument("--height", type=int, default=384, help="Display height per camera.")
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def normalize_frame(frame):
    if isinstance(frame, (list, tuple)):
        frame = frame[0]
    if frame is None:
        return None
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        return None
    return frame


def to_display_bgr(name, frame):
    frame = np.asarray(frame)
    if frame.shape[:2] != MODEL_IMAGE_SIZE[::-1]:
        frame = cv2.resize(frame, MODEL_IMAGE_SIZE)
    if name in ("image_left", "image_right"):
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame


def main():
    args = parse_args()
    cameras = build_cameras(fake_env=args.fake_env)
    if not cameras:
        raise SystemExit("No cameras initialized.")

    latest = {
        name: np.zeros((args.height, args.width, 3), dtype=np.uint8)
        for name, _ in cameras
    }
    delay = max(1, int(1000 / max(args.fps, 1.0)))
    print("Press q or ESC to quit.")
    print("Cameras:", ", ".join(name for name, _ in cameras))

    try:
        while True:
            for name, cam in cameras:
                try:
                    frame = normalize_frame(cam.get_data(viz=False))
                except TypeError:
                    frame = normalize_frame(cam.get_data())
                except Exception as exc:
                    print(f"[WARN] {name}: {exc}")
                    continue
                if frame is None:
                    continue
                display = to_display_bgr(name, frame)
                display = cv2.resize(display, (args.width, args.height))
                cv2.putText(
                    display,
                    name,
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                latest[name] = display

            panel = np.concatenate([latest[name] for name, _ in cameras], axis=1)
            cv2.imshow(args.window, panel)
            key = cv2.waitKey(delay) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        for _name, cam in cameras:
            close = getattr(cam, "close", None)
            if callable(close):
                close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
