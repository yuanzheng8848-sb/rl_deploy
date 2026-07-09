"""Local camera setup for OpenArm RL deployment.

The actual capture thread lives on LocalOpenArmEnv (envs/local_openarm_env.py),
which owns self.cameras / self.latest_images; this module only centralizes the
hardware constants and camera construction so they are reusable and testable.
"""
from pathlib import Path

import yaml

from openarm_env.camera.camera_factory import build_camera


MODEL_IMAGE_SIZE = (128, 128)
INFRA_ROOT = Path(__file__).resolve().parents[2]
CAMERA_CONFIG_PATH = INFRA_ROOT / "openarm_configs" / "cameras.yaml"


_IMAGE_KEYS = {
    "head": "image_primary",
    "left": "image_left",
    "right": "image_right",
}


def load_camera_configs():
    with open(CAMERA_CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


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
    for name, cfg in load_camera_configs().items():
        try:
            cam = build_camera(name, cfg, fake_env=fake_env, mock_camera_cls=MockCamera)
            image_key = _IMAGE_KEYS.get(name, f"image_{name}")
            cameras.append((image_key, cam))
            print(f"Initialized {name} camera ({cfg.get('type')})")
        except Exception as exc:
            print(f"Failed to init {name} camera: {exc}")

    return cameras
