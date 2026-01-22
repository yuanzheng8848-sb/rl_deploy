import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import numpy.typing as npt
import pyroki as pk
import yourdfpy
from pyroki.collision import CollGeom, Sphere, HalfSpace, RobotCollision

from .elbow_cost import elbow_angle_cost


class BaseIKSolver:
    """
    A controller class for upper body inverse kinematics with configurable weights and collision settings.

    This class encapsulates robot loading, IK solving, and collision detection for upper body control.
    """

    @dataclass
    class BaseIKSolverConfig:
        """Configuration for IK cost function weights."""

        use_initial: bool = True
        use_manipulability: bool = False
        pose_position_weight: tuple[float, ...] = (30.0, 30.0)
        pose_orientation_weight: tuple[float, ...] = (15.0, 15.0)
        rest_weight: float = 25.0
        limit_weight: float = 100.0
        elbow_weight: float = 5.0
        manipulability_weight: float = 2.0
        collision_weight: float = 80.0
        collision_margin: float = 0.02

    def __init__(
        self,
        config_solver: dict,
        config_robot: dict,
        reference_path: str,
        visualize_collision: bool = False,
    ):
        """
        Initialize the IK solver from configuration.
        """

        # Set up solver configuration
        self._setup_solver(config_solver)
        self._setup_robot(config_robot, reference_path, visualize_collision)

    def _setup_solver(self, config: dict) -> None:
        self._track_chest = config_robot["chest"]["tracking"]

        cfg_weight = config["weight"]
        position_weight = [cfg_weight["arm_position"]] * 2
        orientation_weight = [cfg_weight["arm_orientation"]] * 2
        if self._track_chest:
            position_weight.append(cfg_weight["chest_position"])
            orientation_weight.append(cfg_weight["chest_orientation"])
        
        self._solver_cfg = self.UpperBodyIKSolverConfig(
            use_initial=config["use_initial"],
            use_manipulability=config["use_manipulability"],
            pose_position_weight=tuple(position_weight),
            pose_orientation_weight=tuple(orientation_weight),
            rest_weight=cfg_weight["rest"],
            limit_weight=cfg_weight["limit"],
            elbow_weight=cfg_weight["elbow"],
            manipulability_weight=cfg_weight["manipulability"],
            collision_weight=cfg_weight["collision"],
            collision_margin=config["collision_margin"],
        )

    def _setup_robot(
        self,
        config_robot: dict,
        reference_path: str,
        visualize_collision: bool = False,
    ) -> None:
        # Load description
        cfg_desc = config_robot["description"]
        path_to_desc: Path = (Path(reference_path) / cfg_desc["package_path"]).resolve()
        urdf_path = (path_to_desc / cfg_desc["urdf_relative_path"]).as_posix()
        self.urdf = yourdfpy.URDF.load(
            urdf_path,
            filename_handler=lambda fname: fname.replace(
                "package://", path_to_desc.as_posix() + "/"
            ),
            build_collision_scene_graph=visualize_collision,
        )
        self._robot = pk.Robot.from_urdf(self.urdf)
        # Set up links indices for target links
        self._target_link_names: list = cfg_desc["end_effector_link_names"]
        cfg_chest = config["chest"]
        if self._track_chest:
            self._target_link_names.append(cfg_chest["link_name"])
        self._target_pose_desired_shape = (len(self._target_link_names), 7)
        self._target_link_idxs = jnp.array(
            [self._robot.links.names.index(name) for name in self._target_link_names]
        )

        # Load disabled collision pairs from SRDF if available
        cfg_coll = config["collision"]
        disabled_pairs: tuple[tuple[str, str], ...] = ()
        if cfg_coll["disabled_pairs"]["enable"]:
            srdf_path = (
                path_to_desc / cfg_coll["disabled_pairs"]["srdf_relative_path"]
            ).as_posix()
            disabled_pairs = self._load_disabled_pairs(srdf_path)
        # Create robot collision model
        sphere_and_capsule_only = cfg_coll["sphere_and_capsule_only"]
        if sphere_and_capsule_only:
            self._robot_coll = RobotCollision.from_urdf_with_only_sphere_and_capsule(
                self.urdf, disabled_pairs
            )
        else:
            self._robot_coll = RobotCollision.from_urdf(self.urdf, disabled_pairs)
        # Set up collision with desktop if specified
        self._world_coll: list[CollGeom] | None = None
        if cfg_coll["world"]["enable"]:
            cfg_wcoll = cfg_coll["world"]
            self._world_coll = []
            if cfg_wcoll["desktop"]["enable"]:
                self._world_coll.append(
                    HalfSpace.from_point_and_normal(
                        point=onp.array(
                            [0.0, 0.0, cfg_wcoll["desktop"]["height_offset"]]
                        ),
                        normal=onp.array([0.0, 0.0, 1.0]),
                    )
                )
            if cfg_wcoll["side_walls"]["enable"]:
                side_offset = cfg_wcoll["side_walls"]["side_offset"]
                for point_y, dir_y in zip([side_offset, -side_offset], [-1.0, 1.0]):
                    self._world_coll.append(
                        HalfSpace.from_point_and_normal(
                            point=onp.array([0.0, point_y, 0.0]),
                            normal=onp.array([0.0, dir_y, 0.0]),
                        )
                    )

        # Elbow angle config
        cfg_elbow = config["elbow_angle"]
        self._track_elbow = True if cfg_elbow["tracking"] else None
        self._shoulder_offset = onp.array(cfg_elbow["shoulder_offset"])
        self._elbow_angle_targets = cfg_elbow["target"]
        for k, v in self._elbow_angle_targets.items():
            self._elbow_angle_targets[k] = v * onp.pi / 180.0
        self._elbow_links_idxs: Dict[str, List[int]] = {"left": [], "right": []}
        elbow_links = cfg_elbow["link_names"]
        for k in self._elbow_links_idxs.keys():
            for link_name in elbow_links[k]:
                self._elbow_links_idxs[k].append(
                    self._robot.links.names.index(link_name)
                )

    def _load_disabled_pairs(self, srdf_path: str) -> tuple[tuple[str, str], ...]:
        """Load disabled collision pairs from SRDF file."""
        try:
            tree = ET.parse(srdf_path)
            root = tree.getroot()

            disabled_pairs = []
            for disable_collision in root.findall("disable_collisions"):
                link1 = disable_collision.get("link1")
                link2 = disable_collision.get("link2")
                if link1 and link2:
                    disabled_pairs.append((link1, link2))

            return tuple(disabled_pairs)
        except (ET.ParseError, FileNotFoundError) as e:
            print(f"Warning: Could not parse SRDF file {srdf_path}: {e}")
            return ()

    def solve_ik(
        self,
        target_pose: npt.NDArray,
        current_joints: npt.NDArray,
    ) -> npt.NDArray:
        # Use provided collision settings or defaults
        num_targets = self._target_link_idxs.shape[0]
        assert target_pose.shape == (num_targets, 7), (
            f"Target pose of {target_pose.shape} mismatches with desired ({num_targets}, 7)"
        )

        # Prepare initial values outside of JIT function
        initial_vals = None
        if self._solver_cfg.use_initial:
            initial_vals = current_joints
        manipulability_weight = None
        if self._solver_cfg.use_manipulability:
            manipulability_weight = self._solver_cfg.manipulability_weight
        cfg = self._solve_ik_jax(
            self._robot,
            jnp.array(target_pose[:, :4]),
            jnp.array(target_pose[:, 4:]),
            jnp.array(self._target_link_idxs),
            self._elbow_links_idxs["left"],
            self._elbow_links_idxs["right"],
            self._elbow_angle_targets["left"],
            self._elbow_angle_targets["right"],
            self._shoulder_offset,
            jnp.array(current_joints),
            self._robot_coll,
            jnp.array(self._solver_cfg.pose_position_weight),
            jnp.array(self._solver_cfg.pose_orientation_weight),
            self._solver_cfg.rest_weight,
            self._solver_cfg.limit_weight,
            self._solver_cfg.elbow_weight,
            self._solver_cfg.collision_margin,
            self._solver_cfg.collision_weight,
            world_coll=self._world_coll,
            elbow_cost_flag=self._track_elbow,
            manipulability_weight=manipulability_weight,
            initial_vals=initial_vals,
        )
        assert cfg.shape == (self._robot.joints.num_actuated_joints,)

        return onp.array(cfg)

    @staticmethod
    @jdc.jit
    def _solve_ik_jax(
        robot: pk.Robot,
        target_wxyz: jax.Array,
        target_position: jax.Array,
        target_joint_idxs: jax.Array,
        elbow_idxs_left: list[int],
        elbow_idxs_right: list[int],
        elbow_angle_left: float,
        elbow_angle_right: float,
        base_offset: onp.typing.NDArray,
        current_joints: jax.Array,
        robot_coll: RobotCollision,
        pose_position_weight: jax.Array,
        pose_orientation_weight: jax.Array,
        rest_weight: float,
        limit_weight: float,
        elbow_weight: float,
        collision_margin: float,
        collision_weight: float,
        # velocity_weight: float,
        # velocity_limit_weight: float,
        world_coll: list[CollGeom] | None = None,
        elbow_cost_flag: bool | None = None,
        manipulability_weight: float | None = None,
        initial_vals: onp.ndarray | None = None,
    ) -> jax.Array:
        joint_var_cls = robot.joint_var_cls

        # Get the batch axes for the variable through the target pose.
        # Batch axes for the variables and cost terms (e.g., target pose) should be broadcastable!
        target_pose = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(target_wxyz), target_position
        )
        batch_axes = target_pose.get_batch_axes()

        factors = [
            pk.costs.pose_cost_analytic_jac(
                jax.tree.map(lambda x: x[None], robot),
                joint_var_cls(jnp.full(batch_axes, 0)),
                target_pose,
                target_joint_idxs,
                pos_weight=pose_position_weight,
                ori_weight=pose_orientation_weight,
            ),
            pk.costs.rest_cost(
                joint_var_cls(0),
                current_joints,
                weight=rest_weight,
            ),
            pk.costs.limit_cost(
                robot,
                joint_var_cls(0),
                weight=limit_weight,
            ),
            pk.costs.self_collision_cost(
                robot,
                robot_coll,
                joint_var_cls(0),
                margin=collision_margin,
                weight=collision_weight,
            ),
        ]
        if world_coll is not None:
            factors.extend(
                [
                    pk.costs.world_collision_cost(
                        robot,
                        robot_coll,
                        joint_var_cls(0),
                        coll,
                        margin=collision_margin,
                        weight=collision_weight,
                    )
                    for coll in world_coll
                ]
            )
        if elbow_cost_flag is not None:
            factors.append(
                elbow_angle_cost(
                    robot,
                    joint_var_cls(0),
                    elbow_idxs_left,
                    base_offset,
                    elbow_angle_left,
                    weight=elbow_weight,
                )
            )
            factors.append(
                elbow_angle_cost(
                    robot,
                    joint_var_cls(0),
                    elbow_idxs_right,
                    base_offset,
                    elbow_angle_right,
                    weight=elbow_weight,
                )
            )
        if manipulability_weight is not None:
            # only use manipulability for arms, not chest
            factors.append(
                pk.costs.manipulability_cost(
                    robot,
                    joint_var_cls(0),
                    target_joint_idxs[:2],
                    manipulability_weight,
                ),
            )

        # Prepare initial values if provided
        if initial_vals is not None:
            initial_values = jaxls.VarValues.make(
                [joint_var_cls(0).with_value(jnp.array(initial_vals))]
            )
        else:
            initial_values = None
        sol = (
            jaxls.LeastSquaresProblem(factors, [joint_var_cls(0)])
            .analyze()
            .solve(
                initial_vals=initial_values,
                verbose=False,
                linear_solver="dense_cholesky",
                # linear_solver="conjugate_gradient",
                trust_region=jaxls.TrustRegionConfig(lambda_initial=10.0),
            )
        )
        return sol[joint_var_cls(0)]

    def forward_kinematics(self, joints: npt.NDArray) -> npt.NDArray:
        """
        Compute forward kinematics.

        Args:
            joints: Joint configuration.

        Returns:
            Link poses
        """
        return onp.array(self._robot.forward_kinematics(jnp.array(joints)))

    def get_desired_target_pose_shape(self) -> tuple[int, int]:
        """Get the desired shape of the target pose array."""
        return self._target_pose_desired_shape

    def set_elbow_angle_targets(self, angle_deg: npt.NDArray) -> None:
        """Set elbow angle targets in degrees."""
        self._elbow_angle_targets["left"] = angle_deg[0] * onp.pi / 180.0
        self._elbow_angle_targets["right"] = angle_deg[1] * onp.pi / 180.0

    def get_elbow_angle_targets(self) -> npt.NDArray | None:
        """Get current elbow angle targets in degrees."""
        if not self._track_elbow:
            return None
        elbow_angle = [
            self._elbow_angle_targets["left"],
            self._elbow_angle_targets["right"],
        ]
        return onp.array(elbow_angle) * 180.0 / onp.pi

    def get_actuated_joint_order(self) -> tuple[str, ...]:
        return self._robot.joints.actuated_names

    def get_target_link_indices(self) -> tuple[int, ...]:
        """Get the names of the target links."""
        return tuple(map(int, self._target_link_idxs))

    def get_target_error(
        self, solution: npt.NDArray, target_pose: npt.NDArray
    ) -> npt.NDArray:
        current_pose = self.forward_kinematics(solution)
        ee_pose = current_pose[self._target_link_idxs]
        trans_error = onp.linalg.norm(ee_pose[:, 4:7] - target_pose[:, 4:7], axis=1)
        ee_xyzw = jnp.concatenate([ee_pose[:, 1:4], ee_pose[:, :1]], axis=1)
        target_xyzw = jnp.concatenate([target_pose[:, 1:4], target_pose[:, :1]], axis=1)
        quat_error = jaxlie.SO3.from_quaternion_xyzw(
            ee_xyzw
        ).inverse() @ jaxlie.SO3.from_quaternion_xyzw(target_xyzw)
        # ensure the w component of quaternion is positive, choose the shorter rotation path
        w_component = quat_error.wxyz[:, 0]
        w_component = jnp.where(w_component < 0, -w_component, w_component)
        w_clipped = jnp.clip(w_component, -1.0, 1.0)
        rot_error = 2 * jnp.acos(w_clipped)
        return onp.array([trans_error, rot_error])

    def update_world_collision(self, sphere_center: onp.ndarray, sphere_radius: float) -> None:
        # sphere_centers size: (n x 3)

        self._world_coll = []
        for i in range(sphere_center.shape[0]):
            self._world_coll.append(
                Sphere.from_center_and_radius(
                    center=sphere_center[i],
                    radius=sphere_radius,
                )
            )
    
    def reset_world_collision(self):
        self._world_coll = []
