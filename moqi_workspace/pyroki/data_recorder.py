# 文件名：data_recorder.py
import cv2
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from queue import Queue
import threading

"""
数据采集模块，用于记录机械臂训练数据（LeRobot 兼容格式）
moqi_workspace/
├── record_data/
│   ├── images/
│   │   ├── cam_1_rgb/
│   │   ├── cam_1_depth/
│   │   ├── cam_2_rgb/
│   │   ├── cam_2_depth/
│   ├── trajectory.npy
│   ├── trajectory.json
"""

class DataRecorder:
    """
    用于采集机械臂训练数据(LeRobot 兼容格式）
    每帧记录：
        - 左右机械臂关节角
        - 左右末端执行器位姿
        - 相机 RGB / Depth 图像
    """

    def __init__(self, save_dir="dataset", use_cameras=True, save_depth=True):
        self.save_dir = Path(save_dir)
        self.use_cameras = use_cameras
        self.save_depth = save_depth

        # 图像保存路径
        if use_cameras:
            self.image_dir = self.save_dir / "images"
            subs = ["cam_0_rgb", "cam_1_rgb", "cam_2_rgb"]
            if self.save_depth:
                subs.extend(["cam_0_depth", "cam_1_depth", "cam_2_depth"])
            for sub in subs:
                (self.image_dir / sub).mkdir(parents=True, exist_ok=True)

        # 数据缓存
        self.data_list = []
        self.frame_idx = 0
        # 异步图像写入
        self._img_queue: Queue | None = None
        self._img_worker: threading.Thread | None = None
        self._stop_event = threading.Event()

        # 确保根目录存在
        self.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DataRecorder] Recording to {self.save_dir.resolve()}")

        if self.use_cameras:
            self._img_queue = Queue(maxsize=1024)
            self._img_worker = threading.Thread(target=self._image_worker, daemon=True)
            self._img_worker.start()

    def _image_worker(self):
        assert self._img_queue is not None
        while not self._stop_event.is_set() or not self._img_queue.empty():
            try:
                task = self._img_queue.get(timeout=0.1)
            except Exception:
                continue
            try:
                kind, path, array = task
                if kind == "rgb":
                    # cv2.imwrite 期望 BGR 格式，所以需要将 RGB 转回 BGR
                    bgr_image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
                    # 适度降低 JPEG 质量以提升速度
                    cv2.imwrite(str(path), bgr_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                elif kind == "depth":
                    # 直接保存为 .npy（可选：切换为 np.savez_compressed 节省空间但更耗 CPU）
                    np.save(str(path), array)
            finally:
                self._img_queue.task_done()

    def record(self, obs_joints, obs_ee_poses, obs_gripper_joints, action_joints, action_ee_poses, action_gripper_joints, cameras=None):
        """保存一帧数据"""
        timestamp = datetime.now().isoformat()
        frame_id = f"{self.frame_idx:06d}"

        # 保存相机图像
        rgb_paths, depth_paths = [], []
        if self.use_cameras and cameras is not None:
            for i, cam in enumerate(cameras):
                rgb, depth = cam.color_image, cam.depth_image
                rgb_path = self.image_dir / f"cam_{i}_rgb" / f"{frame_id}.jpg"
                
                # 立即记录路径，异步写入磁盘
                rgb_paths.append(str(rgb_path.relative_to(self.save_dir)))
                if self._img_queue is not None and rgb is not None:
                    self._img_queue.put(("rgb", rgb_path, rgb))

                if self.save_depth:
                    depth_path = self.image_dir / f"cam_{i}_depth" / f"{frame_id}.npy"
                    depth_paths.append(str(depth_path.relative_to(self.save_dir)))
                    if self._img_queue is not None and depth is not None:
                        self._img_queue.put(("depth", depth_path, depth))

        # 组织一帧数据结构（LeRobot 格式）
        record = {
            "frame_id": self.frame_idx,
            "timestamp": timestamp,
            "observations": {
                "qpos": np.array(obs_joints).flatten().tolist(),
                "ee_pose": np.array(obs_ee_poses).flatten().tolist(),
                "gripper_joints":np.array(obs_gripper_joints).flatten().tolist(),
                "rgb_paths": rgb_paths,
                "depth_paths": depth_paths,
            },
            "action": {
                "qpos": np.array(action_joints).flatten().tolist(),
                "ee_pose": np.array(action_ee_poses).flatten().tolist(),
                "gripper_joints":np.array(action_gripper_joints).flatten().tolist(),
            },
        }
        # print(record)

        self.data_list.append(record)
        self.frame_idx += 1

    def save(self):
        """保存所有采集数据"""
        # 等待图像队列写入完成
        if self._img_queue is not None:
            self._img_queue.join()
            self._stop_event.set()
            if self._img_worker is not None and self._img_worker.is_alive():
                self._img_worker.join(timeout=1.0)
        np.save(self.save_dir / "trajectory.npy", self.data_list)
        with open(self.save_dir / "trajectory.json", "w") as f:
            json.dump(self.data_list, f, indent=2)
        print(f"[DataRecorder] Saved {self.frame_idx} frames to {self.save_dir}")
