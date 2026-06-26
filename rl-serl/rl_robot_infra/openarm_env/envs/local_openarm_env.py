"""LocalOpenArmEnv: OpenArm env variant that reads cameras locally.

Migrated from rl_deploy/train.py (LocalOpenArmEnv, lines ~211-328).

Unlike OpenArmEnv (which receives images base64-encoded from the flask server),
this variant runs its own background capture thread reading the three cameras
directly on the client side (head USB + two wrist RealSense). It is used by the
training/actor entrypoints so that image latency does not go through the server.
"""
import time
import cv2
import numpy as np
from threading import Event, Thread

from openarm_env.envs.openarm_env import OpenArmEnv
from openarm_env.camera.local_camera import (
    MODEL_IMAGE_SIZE,
    build_cameras,
)


class LocalOpenArmEnv(OpenArmEnv):
    """OpenArm env variant that always reads images locally, even in fake mode."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.fake_env:
            print("[LocalOpenArmEnv] forcing local camera init in fake mode.")
            self.init_cameras(None)

    def init_cameras(self, _config):
        self.cameras = build_cameras(fake_env=self.fake_env)
        self.latest_images_raw = {}

        self.stop_event = Event()
        self.capture_thread = Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _capture_loop(self):
        while not self.stop_event.is_set():
            for name, cam in self.cameras:
                try:
                    frame = cam.get_data(viz=False)
                    is_realsense = isinstance(frame, (list, tuple))
                    if is_realsense:
                        frame = frame[0]
                    if frame is None:
                        continue

                    if is_realsense:
                        full_rgb = frame
                    else:
                        full_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Keep raw frames as the camera produced them. Network-view
                    # cropping is owned by NetworkPrimaryImageCropWrapper.
                    self.latest_images_raw[name] = full_rgb

                    resized = cv2.resize(frame, MODEL_IMAGE_SIZE)
                    if is_realsense:
                        rgb = resized
                    else:
                        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    self.latest_images[name] = rgb
                except Exception as exc:
                    print(f"[Capture Error:{name}] {exc}")
            time.sleep(0.01)

    def close(self):
        if hasattr(self, "stop_event"):
            self.stop_event.set()
        if hasattr(self, "capture_thread"):
            self.capture_thread.join(timeout=1.0)
        super().close()
