"""Local camera construction for OpenArm environments."""

from pathlib import Path

import yaml

from openarm_env.camera.camera_factory import build_camera
from openarm_env.mock_hardware import MockCamera


MODEL_IMAGE_SIZE = (128, 128)

DEPLOYMENT_IMAGE_KEYS = {
    "head": "image_primary",
    "left": "image_left",
    "right": "image_right",
}


def resolve_camera_config(camera_ref):
    if isinstance(camera_ref, dict):
        cfg = dict(camera_ref)
        cfg.setdefault("name", cfg.get("id", "camera"))
        return cfg
    raise TypeError(f"Camera config must be an explicit mapping, got {camera_ref!r}")


def load_deployment_camera_config(path=None):
    config_path = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[2] / "openarm_configs" / "cameras.yaml"
    )
    with open(config_path, encoding="utf-8") as handle:
        hardware_config = yaml.safe_load(handle) or {}
    return {
        image_key: {"name": hardware_name, **dict(hardware_config[hardware_name])}
        for hardware_name, image_key in DEPLOYMENT_IMAGE_KEYS.items()
    }


def build_cameras(camera_config, virtual=False):
    """Construct explicit (image_key, camera) pairs from deployment config."""
    if not isinstance(camera_config, dict):
        raise TypeError("camera_config must be a dict mapping image keys to camera configs")

    cameras = []
    seen = set()
    for image_key, camera_ref in camera_config.items():
        if image_key in seen:
            raise ValueError(f"Duplicate image key {image_key!r}")
        seen.add(image_key)
        cfg = resolve_camera_config(camera_ref)
        camera_name = cfg.get("name", image_key)
        cam = build_camera(camera_name, cfg, virtual=virtual, mock_camera_cls=MockCamera)
        cameras.append((image_key, cam))
        mode = "virtual" if virtual else cfg.get("type")
        print(f"Initialized {image_key} camera ({mode})")
    return cameras
