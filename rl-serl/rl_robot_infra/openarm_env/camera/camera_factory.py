"""Camera factory for OpenArm hardware variants."""


def build_camera(name, cfg, virtual=False, mock_camera_cls=None):
    camera_type = cfg.get("type", "").lower()

    if virtual:
        if mock_camera_cls is None:
            raise ValueError("virtual camera mode requires a mock_camera_cls")
        return mock_camera_cls(
            width=cfg.get("width", 640),
            height=cfg.get("height", 480),
            fps=cfg.get("fps", 30),
        )

    if camera_type in {"usb", "opencv"}:
        from openarm_env.camera.usb_camera import USBCamera

        return USBCamera(
            device_id=cfg.get("device_id"),
            width=cfg.get("width", 640),
            height=cfg.get("height", 480),
            fps=cfg.get("fps", 30),
            exposure=cfg.get("exposure"),
            backend=cfg.get("backend", "opencv"),
            env_device_id=cfg.get("env_device_id", "OPENARM_HEAD_CAMERA_DEVICE"),
        )

    if camera_type == "realsense":
        from openarm_env.camera.realsense_camera import RealsenseCamera

        return RealsenseCamera(
            device_id=cfg.get("serial") or cfg.get("device_id"),
            enable_depth=cfg.get("enable_depth", False),
            width=cfg.get("width", 640),
            height=cfg.get("height", 480),
            fps=cfg.get("fps", 30),
        )

    raise ValueError(f"Unsupported camera type for {name}: {camera_type}")


def camera_frame(data):
    if isinstance(data, (list, tuple)):
        return data[0] if data else None
    return data
