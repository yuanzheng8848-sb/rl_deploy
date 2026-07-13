"""Transformation helpers used by OpenArm wrappers."""

import numpy as np
from scipy.spatial.transform import Rotation as R


def construct_twist_rotation_matrix(tcp_pose):
    """Rotate a TCP-local ``[linear, angular]`` twist into the world frame.

    Both parts of the twist are velocities at the TCP itself, so changing their
    coordinate basis only requires the pose rotation.  The translation term of
    an SE(3) adjoint would change the reference point of the twist and must not
    be applied here.
    """
    rotation = R.from_quat(tcp_pose[3:]).as_matrix()
    transform = np.zeros((6, 6), dtype=rotation.dtype)
    transform[:3, :3] = rotation
    transform[3:, 3:] = rotation
    return transform


def construct_homogeneous_matrix(tcp_pose):
    """Construct the homogeneous transformation matrix from a pose."""
    rotation = R.from_quat(tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    transform = np.zeros((4, 4))
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    transform[3, 3] = 1
    return transform
