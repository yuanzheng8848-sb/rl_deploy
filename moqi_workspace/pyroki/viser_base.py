"""Main function for bimanual IK with optional self-collision avoidance."""

import queue
import threading
from typing import Callable

import numpy as np
import numpy.typing as npt
import viser
import yourdfpy
from loguru import logger
from viser.extras import ViserUrdf

from pyroki.collision import CollGeom, Sphere, HalfSpace, RobotCollision

import pdb

class ViserBase:
    def __init__(
        self,
        config: dict,
        urdf: yourdfpy.URDF,
        joint_order: tuple[str, ...],
        target_link_idxs: tuple[int, ...],
        fk_func: Callable[[npt.NDArray], npt.NDArray] | None = None,
        use_sim: bool = False,
        use_teleop: bool = False,
    ) -> None:
        assert not use_sim or (use_sim and fk_func is not None), (
            "Forward kinematic function must be provided for initialize teleoperation pose for simulation."
        )
        self._use_sim = use_sim
        self._use_teleop = use_teleop

        # setup kinematic simulation
        self._target_link_idxs = target_link_idxs
        self._sim_initial_joints = np.zeros(len(joint_order))
        self._fk_func: Callable[[npt.NDArray], npt.NDArray] | None = None
        self._joint_current: npt.NDArray | None = None
        if use_sim:
            self._fk_func = fk_func
            self._setup_kinematic_sim(
                config["sim_init_joint_deg"],
                None,
                joint_order,
            )

        # setup visualization
        self.nb_vis_frames = config["nb_vis_frames"]
        self._setup_visualization(urdf, config)

        # thread for update joint configuration
        self._shutdown_event = threading.Event()
        self._queue_joint: queue.Queue[npt.NDArray[np.float64]] = queue.Queue(
            maxsize=1
        )
        self._thread_joint_update: threading.Thread | None = None
        self._start()

        self.object_handle_list = []

    def _setup_kinematic_sim(
        self,
        init_joint_deg: dict,
        remove_joint_name: tuple[str] | None,
        joint_order: tuple[str, ...],
    ) -> None:
        init_joints = [0.0 for _ in joint_order]
        if remove_joint_name:
            for _joint_name in remove_joint_name:
                init_joint_deg.pop(_joint_name)
        
        for k, v in init_joint_deg.items():
            if k in joint_order:
                init_joints[joint_order.index(k)] = v
        self._sim_initial_joints = np.array(init_joints)
        self._joint_current = np.array(init_joints)

    def get_init_joints_for_sim(self) -> npt.NDArray:
        """Get the initial joint configuration."""
        return self._sim_initial_joints

    def _setup_visualization(self, urdf: yourdfpy.URDF, config: dict) -> None:
        # setup visualization
        self._server = viser.ViserServer()
        self._server.scene.add_grid("/ground", width=2, height=2)
        self._urdf_vis = ViserUrdf(
            self._server,
            urdf,
            root_node_name="/base",
            load_collision_meshes=config["visualize_collision"],
            collision_mesh_color_override=(0.0, 0.0, 0.3, 0.3),
        )
        self._urdf_vis.update_cfg(self._sim_initial_joints)

        # setup initial frames in ui
        self._setup_frames()

        # timing display
        self._timing_handle = self._server.gui.add_number(
            "Elapsed (ms)", 0.001, disabled=True
        )
        self._joint_speed_handle = self._server.gui.add_number(
            "max speed of all joints (rad/s)", 0.001, disabled=True
        )

    def _setup_frames(self) -> None:
        # Create origin
        o_trans = (0, 0, 0)
        o_wxyz = (1, 0, 0, 0)
        self._server.scene.add_frame(
            "/origin", position=o_trans, wxyz=o_wxyz, axes_length=0.5
        )
        # Get initial target poses
        target_poses = []
        if self._use_sim:
            assert self._fk_func is not None, (
                "Forward kinematic function must be provided for initialize teleoperation pose for simulation."
            )
            target_wxyz_xyz = self._fk_func(self._sim_initial_joints)[
                list(self._target_link_idxs)
            ]
            for idx in range(len(self._target_link_idxs)):
                target_poses.append(
                    (
                        tuple(target_wxyz_xyz[idx, 4:]),
                        tuple(target_wxyz_xyz[idx, :4]),
                    )
                )
        else:
            for _ in range(len(self._target_link_idxs)):
                target_poses.append((o_trans, o_wxyz))
        
        # Create visualization handles for target poses
        names_ctrl = ["left", "right"]
        if len(self._target_link_idxs) == 3:
            names_ctrl.append("chest")
        
        # 控制设置
        ctrl_params = []
        for name, (position, wxyz) in zip(names_ctrl, target_poses):
            ctrl_params.append(
                {
                    "name": f"/target_{name}",
                    "scale": 0.2,
                    "position": position,
                    "wxyz": wxyz,
                }
            )
        self._ctrl_frame_handles: list[
            viser.FrameHandle | viser.TransformControlsHandle
        ] = []

        if self._use_teleop:
            # Explicitly unpack parameters for add_transform_controls
            for param in ctrl_params:
                scale = param.get("scale", 0.1)
                self._ctrl_frame_handles.append(
                    self._server.scene.add_transform_controls(
                        str(param["name"]) + 'ctrl',
                        position=np.array(param["position"]),
                        wxyz=np.array(param["wxyz"]),
                        scale=scale if isinstance(scale, float) else 0.2,
                    )
                )

        # 显示设置

        self._vis_frame_handles = []
        # Explicitly unpack parameters for add_frame
        for i in range(self.nb_vis_frames):
            scale = 0.2
            self._vis_frame_handles.append(
                self._server.scene.add_frame(
                    'vis' + str(i),
                    position=np.array([0.0, 0.0, 0.0]),
                    wxyz=np.array([1,0,0,1]),
                    axes_length=scale if isinstance(scale, float) else 0.2,
                    axes_radius=0.01,
                )
            )

    def _start(self):
        if self._thread_joint_update is None:
            self._thread_joint_update = threading.Thread(
                target=self._joints_loop,
                daemon=True,
            )
            self._thread_joint_update.start()
            logger.info("Viser joint updating thread started")

    def stop(self) -> None:
        self._shutdown_event.set()
        if self._thread_joint_update is not None:
            logger.info("Stop viser update thread ...")
            self._thread_joint_update.join()
            logger.info("Viser update thread stopped.")

    def _joints_loop(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                joints = self._queue_joint.get(timeout=1.0)
                self._urdf_vis.update_cfg(joints)
            except queue.Empty:
                continue

    def get_target_pose(self) -> npt.NDArray:
        """Get target pose in (num_target, 7), the sequence is rwxyz-xyz"""
        target_poses = []
        for handle in self._ctrl_frame_handles:
            target_poses.append(np.concatenate([handle.wxyz, handle.position]))
        return np.array(target_poses).reshape(-1, 7)

    def get_target_pose_sim(self) -> npt.NDArray:
        assert (
            self._use_sim
            and self._fk_func is not None
            and self._joint_current is not None
        ), "Only use this method in kinematic simulation."
        target_wxyz_xyz = self._fk_func(self._joint_current)[
            list(self._target_link_idxs)
        ]
        target_poses = []
        for wxyz_xyz in target_wxyz_xyz:
            target_poses.append(
                np.concat([wxyz_xyz[4:], wxyz_xyz[1:4], [wxyz_xyz[0]]])
            )
        return np.array(target_poses)

    def update_joints(self, joints: npt.NDArray):
        # self._urdf_vis.update_cfg(joints)
        try:
            self._queue_joint.put_nowait(joints)
        except queue.Full:
            self._queue_joint.get_nowait()
            self._queue_joint.put_nowait(joints)
        if self._use_sim:
            self._joint_current = joints

    def update_results(
        self,
        solution: npt.NDArray,
        elapsed_time: float,
        joint_speed: float | None = None,
    ) -> None:
        """Update the visualizer with the new joint configuration."""
        self._timing_handle.value = 0.99 * self._timing_handle.value + 0.01 * (
            elapsed_time * 1000
        )
        if joint_speed is not None:
            self._joint_speed_handle.value = joint_speed
        self.update_joints(solution)

    def update_vis_frame(self, frame_poses: npt.NDArray) -> None:
        """Update the target poses in the visualizer."""
        for i, handle in enumerate(self._vis_frame_handles):
            handle.wxyz = frame_poses[i, :4]
            handle.position = frame_poses[i, 4:]


    def update_env_collision(self, sphere_center: np.ndarray, sphere_radius: float):
        # sphere_centers size: (n x 3)

        for i in range(sphere_center.shape[0]):
            sphere_coll = Sphere.from_center_and_radius(
                    center=sphere_center[i],
                    radius=sphere_radius,
                )
            object_handle = self._server.scene.add_mesh_trimesh("/obstacle/sphere" + str(i), mesh=sphere_coll.to_trimesh())
        
            self.object_handle_list.append(object_handle)

    def remove(self):
        for object_handle in self.object_handle_list:
            object_handle.remove()

if __name__ == "__main__":
    import yaml
    import pdb
    from pathlib import Path
    
    # Dynamic path resolution
    current_dir = Path(__file__).resolve().parent
    workspace_root = current_dir.parent
    
    cfgs_path = current_dir / "config"
    cfg_viser = yaml.safe_load( (cfgs_path / "viser.yaml").read_text())

    urdf_path = (workspace_root / "openarm/openarm_description/urdf/robot/openarm_bimanual.urdf").as_posix()
    
    # Define filename handler for package://
    package_path = (workspace_root / "openarm").as_posix()
    def handler(fname):
        return fname.replace("package://", package_path + "/")

    joint_order = [
        "openarm_left_joint1",
        "openarm_left_joint2",
        "openarm_left_joint3",
        "openarm_left_joint4",
        "openarm_left_joint5",
        "openarm_left_joint6",
        "openarm_left_joint7",

        # arm
        "openarm_right_joint1",
        "openarm_right_joint2",
        "openarm_right_joint3",
        "openarm_right_joint4",
        "openarm_right_joint5",
        "openarm_right_joint6",
        "openarm_right_joint7",
    ]

    urdf = yourdfpy.URDF.load(
            urdf_path,
            filename_handler=handler,
            build_collision_scene_graph=False,
    )

    def test():

        return

    viser = ViserBase(cfg_viser, urdf, joint_order, (6, 16), test, True, True)

