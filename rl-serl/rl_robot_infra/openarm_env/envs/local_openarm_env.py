"""LocalOpenArmEnv: OpenArm env variant that reads cameras locally."""
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
    """OpenArm env variant that reads real or virtual cameras locally."""

    def init_cameras(self, camera_config):
        self.cameras = build_cameras(camera_config, virtual=self.is_virtual)
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
        for _, camera in getattr(self, "cameras", []):
            close = getattr(camera, "close", None)
            if callable(close):
                close()
                continue
            stop = getattr(camera, "stop", None)
            if callable(stop):
                stop()
        super().close()
