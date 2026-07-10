"""Local camera construction for OpenArm environments."""

from openarm_env.camera.camera_factory import build_camera


MODEL_IMAGE_SIZE = (128, 128)

_DEFAULT_CAMERA_CONFIGS = {
    "local-head": {
        "name": "head",
        "type": "usb",
        "width": 640,
        "height": 480,
        "fps": 30,
        "env_device_id": "OPENARM_HEAD_CAMERA_DEVICE",
    },
    "local-left": {
        "name": "left",
        "type": "realsense",
        "width": 640,
        "height": 480,
        "fps": 30,
    },
    "local-right": {
        "name": "right",
        "type": "realsense",
        "width": 640,
        "height": 480,
        "fps": 30,
    },
}


def resolve_camera_config(camera_ref):
    if isinstance(camera_ref, dict):
        cfg = dict(camera_ref)
        cfg.setdefault("name", cfg.get("id", "camera"))
        return cfg
    if isinstance(camera_ref, str):
        if camera_ref not in _DEFAULT_CAMERA_CONFIGS:
            raise ValueError(f"Unknown camera ref {camera_ref!r}")
        return dict(_DEFAULT_CAMERA_CONFIGS[camera_ref])
    raise TypeError(f"Camera config must be a dict or known camera ref, got {camera_ref!r}")


def build_cameras(camera_config, virtual=False):
    """Construct (image_key, camera) pairs from EnvConfig.REALSENSE_CAMERAS only."""
    if not isinstance(camera_config, dict):
        raise TypeError("camera_config must be a dict mapping image keys to camera configs")

    try:
        from openarm_env.mock_hardware import MockCamera
    except Exception:
        MockCamera = None

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
