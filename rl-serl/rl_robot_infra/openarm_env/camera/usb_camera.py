"""Configurable USB camera backend for OpenArm deployments.

The top/head camera may vary between robots. Keep the public camera contract
small: get_data() returns one BGR frame, matching cv2.VideoCapture.
"""

import os

import cv2


class USBCamera:
    def __init__(
        self,
        device_id=None,
        width=640,
        height=480,
        fps=30,
        exposure=None,
        backend="opencv",
        env_device_id="OPENARM_HEAD_CAMERA_DEVICE",
    ):
        if backend != "opencv":
            raise ValueError(
                f"Unsupported USB camera backend '{backend}'. "
                "Add a backend adapter in openarm_env.camera.camera_factory."
            )

        if device_id is None:
            device_id = os.getenv(env_device_id)
        if device_id is None:
            device_id = 0

        self.device_id = device_id
        self.cap = cv2.VideoCapture(device_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        if exposure is not None:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open USB camera device {device_id}")

    def get_data(self, viz=False):
        ret, frame = self.cap.read()
        if not ret:
            return None

        if viz:
            cv2.imshow("USB Camera", frame)
            cv2.waitKey(1)

        return frame

    def __del__(self):
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
