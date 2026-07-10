"""Virtual camera sources for offline OpenArm environments."""

import numpy as np


class MockCamera:
    """Camera source that returns black RGB frames."""

    def __init__(self, width=640, height=480, fps=30):
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        print(f"[MockCamera] Initialized {self.width}x{self.height}@{self.fps}fps")

    def get_data(self, viz=False):
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return [img, None]
