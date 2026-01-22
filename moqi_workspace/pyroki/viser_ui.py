"""Main function for bimanual IK with optional self-collision avoidance."""

import queue
import threading
from typing import Callable

import numpy as onp
import numpy.typing as npt
import viser
import yourdfpy
from loguru import logger
from viser.extras import ViserUrdf

from pyroki.collision import CollGeom, Sphere, HalfSpace, RobotCollision

class ViserUpperBodyUI:
    def __init__(
        self,
        config: dict,
        urdf: yourdfpy.URDF,
        joint_order: tuple[str, ...],
        target_link_idxs: tuple[int, ...],
        elbow_angle_deg: npt.NDArray | None = None,
        fk_func: Callable[[npt.NDArray], npt.NDArray] | None = None,
        use_sim: bool = False,
        use_teleop: bool = False,
    ) -> None:
        assert not use_sim or (use_sim and fk_func is not None), (
            "Forward kinematic function must be provided for initialize teleoperation pose for simulation."
        )
        self._use_sim = use_sim
        self._use_teleop = use_teleop
        # lift link config
        lift_joint_name = config["lift_joint_name"]
        lift_enable = lift_joint_name in joint_order

        # setup kinematic simulation
        self._target_link_idxs = target_link_idxs
        self._sim_initial_joints = onp.zeros(len(joint_order))
        self._fk_func: Callable[[npt.NDArray], npt.NDArray] | None = None
        self._joint_current: npt.NDArray | None = None
        if use_sim:
            self._fk_func = fk_func
            self._setup_kinamatic_sim(
                config["sim_init_joint_deg"],
                lift_joint_name if lift_enable else None,
                joint_order,
            )

        # setup visualization
        self._setup_visualization(urdf, config)

        # Create GUI controls for elbow angle control
        self._enable_elbow_angle = False
        if elbow_angle_deg:
            self._enable_elbow_angle = True
            self._setup_elbow_sliders(elbow_angle_deg)

        # thread for update joint configuration
        self._shutdown_event = threading.Event()
        self._queue_joint: queue.Queue[npt.NDArray[onp.float64]] = queue.Queue(
            maxsize=1
        )
        self._thread_joint_update: threading.Thread | None = None
        self._start()

        self.object_handle_list = []

    def _setup_kinamatic_sim(
        self,
        init_joint_deg: dict,
        remove_lift_name: str | None,
        joint_order: tuple[str, ...],
    ) -> None:
        init_joints = [0.0 for _ in joint_order]
        if remove_lift_name:
            init_joint_deg.pop(remove_lift_name)
        for k, v in init_joint_deg.items():
            if k in joint_order:
                init_joints[joint_order.index(k)] = v * onp.pi / 180.0
        self._sim_initial_joints = onp.array(init_joints)
        self._joint_current = onp.array(init_joints)

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
        names_vis = ["left", "right"]
        if len(self._target_link_idxs) == 3:
            names_vis.append("chest")
        vis_params = []
        for name, (position, wxyz) in zip(names_vis, target_poses):
            vis_params.append(
                {
                    "name": f"/target_{name}",
                    "scale": 0.2,
                    "position": position,
                    "wxyz": wxyz,
                }
            )
        self._vis_frame_handles: list[
            viser.FrameHandle | viser.TransformControlsHandle
        ] = []
        if self._use_teleop:
            # Explicitly unpack parameters for add_transform_controls
            for param in vis_params:
                scale = param.get("scale", 0.2)
                self._vis_frame_handles.append(
                    self._server.scene.add_transform_controls(
                        str(param["name"]),
                        position=onp.array(param["position"]),
                        wxyz=onp.array(param["wxyz"]),
                        scale=scale if isinstance(scale, float) else 0.2,
                    )
                )
        else:
            # Explicitly unpack parameters for add_frame
            for param in vis_params:
                scale = param.get("scale", 0.2)
                self._vis_frame_handles.append(
                    self._server.scene.add_frame(
                        str(param["name"]),
                        position=onp.array(param["position"]),
                        wxyz=onp.array(param["wxyz"]),
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

    def is_elbow_angle_enabled(self) -> bool:
        """Check if elbow angle control is enabled."""
        return self._enable_elbow_angle

    def _setup_elbow_sliders(self, elbow_angle_deg: npt.NDArray) -> None:
        """Setup elbow angle sliders in the UI."""
        elbow_angle_left = elbow_angle_deg[0]
        elbow_angle_right = elbow_angle_deg[1]
        with self._server.gui.add_folder("Elbow Angle"):
            self._elbow_angle_left_slider = self._server.gui.add_slider(
                "Left Elbow Angle (deg)",
                min=-180.0,
                max=0.0,
                step=0.1,
                initial_value=elbow_angle_left,
            )
            self._elbow_angle_right_slider = self._server.gui.add_slider(
                "Right Elbow Angle (deg)",
                min=0.0,
                max=180.0,
                step=0.1,
                initial_value=elbow_angle_right,
            )

    def get_target_pose(self) -> npt.NDArray:
        """Get target pose in (num_target, 7), the sequence is rwxyz-xyz"""
        target_poses = []
        for handle in self._vis_frame_handles:
            target_poses.append(onp.concatenate([handle.wxyz, handle.position]))
        return onp.array(target_poses).reshape(-1, 7)

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
                onp.concat([wxyz_xyz[4:], wxyz_xyz[1:4], [wxyz_xyz[0]]])
            )
        return onp.array(target_poses)

    def get_elbow_angle_deg(self) -> npt.NDArray | None:
        if not self._enable_elbow_angle:
            return None
        return onp.array(
            [self._elbow_angle_left_slider.value, self._elbow_angle_right_slider.value]
        )

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

    def update_target_pose(self, target_poses: npt.NDArray) -> None:
        """Update the target poses in the visualizer."""
        for i, handle in enumerate(self._vis_frame_handles):
            handle.wxyz = target_poses[i, :4]
            handle.position = target_poses[i, 4:]


    def update_env_collision(self, sphere_center: onp.ndarray, sphere_radius: float):
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