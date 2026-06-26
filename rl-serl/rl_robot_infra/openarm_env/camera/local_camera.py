"""Local camera setup for OpenArm RL deployment.

Migrated from rl_deploy/train.py (LocalOpenArmEnv.init_cameras / _capture_loop).
Provides the device constants and a helper to construct the three cameras used
during local (client-side) image capture:
  - image_left  : RealSense (serial 150622074105)
  - image_right : RealSense (serial 236422072385)
  - image_primary (head) : USB global-shutter camera (OpenCVCamera)

The actual capture thread lives on LocalOpenArmEnv (envs/local_openarm_env.py),
which owns self.cameras / self.latest_images; this module only centralizes the
hardware constants and camera construction so they are reusable and testable.

RealsenseCamera / OpenCVCamera come from moqi_workspace/pyroki (added to sys.path
by examples/compat.py). MockCamera comes from the migrated mock_hardware.
"""
import sys
from pathlib import Path

# Bottom-out path injection for pyroki (realsense_camera), in case this module
# is imported without examples.compat having run.
# This file: rl-serl/rl_robot_infra/openarm_env/camera/local_camera.py
# parents: [0]=camera [1]=openarm_env [2]=rl_robot_infra [3]=rl-serl [4]=zy
_ZY_ROOT = Path(__file__).resolve().parents[4]
_PYROKI_DIR = _ZY_ROOT / "moqi_workspace" / "pyroki"
if _PYROKI_DIR.exists() and str(_PYROKI_DIR) not in sys.path:
    sys.path.insert(0, str(_PYROKI_DIR))

try:
    from realsense_camera import OpenCVCamera, RealsenseCamera
except Exception as exc:  # pragma: no cover - hardware/runtime dependent
    print(f"[local_camera] Failed to import camera modules from pyroki: {exc}")
    RealsenseCamera = None
    OpenCVCamera = None

# --- Hardware constants (migrated from train.py) ---
HEAD_CAMERA_DEVICE = "/dev/v4l/by-id/usb-Global_Shutter_Camera_Global_Shutter_Camera_01.00.00-video-index0"
HEAD_CAMERA_WIDTH = 640
HEAD_CAMERA_HEIGHT = 480
MODEL_IMAGE_SIZE = (128, 128)

LEFT_CAMERA_SERIAL = "150622074105"
RIGHT_CAMERA_SERIAL = "236422072385"


def build_cameras(fake_env=False):
    """Construct the (key, camera) list used by LocalOpenArmEnv.

    Returns a list of (image_key, camera_obj). Any camera that fails to
    initialize is skipped with a warning (matches original behavior).
    """
    try:
        from openarm_env.mock_hardware import MockCamera
    except Exception:
        MockCamera = None

    cameras = []

    # Left wrist RealSense
    try:
        if fake_env and MockCamera is not None:
            cam_left = MockCamera(width=640, height=480, fps=30)
        else:
            cam_left = RealsenseCamera(
                device_id=LEFT_CAMERA_SERIAL,
                enable_depth=False,
                width=640,
                height=480,
                fps=30,
            )
        cameras.append(("image_left", cam_left))
        print(f"Initialized Left Camera ({LEFT_CAMERA_SERIAL})")
    except Exception as exc:
        print(f"Failed to init left camera: {exc}")

    # Right wrist RealSense
    try:
        if fake_env and MockCamera is not None:
            cam_right = MockCamera(width=640, height=480, fps=30)
        else:
            cam_right = RealsenseCamera(
                device_id=RIGHT_CAMERA_SERIAL,
                enable_depth=False,
                width=640,
                height=480,
                fps=30,
            )
        cameras.append(("image_right", cam_right))
        print(f"Initialized Right Camera ({RIGHT_CAMERA_SERIAL})")
    except Exception as exc:
        print(f"Failed to init right camera: {exc}")

    # Head USB camera (primary)
    try:
        if fake_env and MockCamera is not None:
            cam_head = MockCamera(
                width=HEAD_CAMERA_WIDTH,
                height=HEAD_CAMERA_HEIGHT,
                fps=30,
            )
        else:
            cam_head = OpenCVCamera(
                HEAD_CAMERA_DEVICE,
                width=HEAD_CAMERA_WIDTH,
                height=HEAD_CAMERA_HEIGHT,
                fps=30,
                exposure=150,
            )
        cameras.append(("image_primary", cam_head))
        print(f"Initialized Head Camera ({HEAD_CAMERA_DEVICE})")
    except Exception as exc:
        print(f"Failed to init head camera: {exc}")

    return cameras
