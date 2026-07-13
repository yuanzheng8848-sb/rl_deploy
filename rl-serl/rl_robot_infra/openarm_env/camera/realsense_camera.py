import time

import cv2
import numpy as np
import pyrealsense2 as rs


class RealsenseCamera:
    def __init__(self, device_id=None, enable_depth=True, width=640, height=480, fps=30):
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("No Realsense device connected")

        device = None
        if device_id is not None:
            for dev in devices:
                if dev.get_info(rs.camera_info.serial_number) == device_id:
                    device = dev
                    break
            if device is None:
                raise RuntimeError(f"Device with serial number {device_id} not found")
        else:
            device = devices[0]

        self.pipeline = rs.pipeline()
        config = rs.config()
        self.serial_number = device.get_info(rs.camera_info.serial_number)
        config.enable_device(self.serial_number)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        self.enable_depth = bool(enable_depth)
        if self.enable_depth:
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        self.pipeline.start(config)
        self.rate = 0.0
        self.k = 0
        self.color_image = None
        self.depth_image = None

    def _read_data(self, viz=False):
        start = time.perf_counter()
        data = [None, None]
        frames = self.pipeline.wait_for_frames()

        rgb_frame = frames.get_color_frame()
        if rgb_frame:
            color_image = np.asanyarray(rgb_frame.get_data())
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            data[0] = color_image
            self.color_image = color_image
            if viz:
                cv2.imshow(
                    f"RGB Image-{self.serial_number}",
                    cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR),
                )
                cv2.waitKey(1)
        else:
            self.color_image = None

        if self.enable_depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth_np = np.asanyarray(depth_frame.get_data())
                data[1] = depth_np
                self.depth_image = depth_np
                if viz:
                    depth_color = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_np, alpha=0.03),
                        cv2.COLORMAP_JET,
                    )
                    cv2.imshow(f"depth Image-{self.serial_number}", depth_color)
                    cv2.waitKey(1)
            else:
                self.depth_image = None
        else:
            self.depth_image = None

        elapsed = time.perf_counter() - start
        fps = 1 / elapsed if elapsed > 0 else 0.0
        self.rate = (self.k * self.rate + fps) / (self.k + 1)
        self.k += 1
        return data

    def read_rgb(self, viz=False):
        return self._read_data(viz=viz)[0]

    def close(self):
        if hasattr(self, "pipeline"):
            self.pipeline.stop()

    def stop(self):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def get_rate(self):
        return self.rate
