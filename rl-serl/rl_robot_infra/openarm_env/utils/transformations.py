"""Transformation helpers used by OpenArm wrappers."""

import numpy as np
from scipy.spatial.transform import Rotation as R


def construct_adjoint_matrix(tcp_pose):
    """Construct the adjoint matrix for a spatial velocity vector."""
    rotation = R.from_quat(tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    skew_matrix = np.array(
        [
            [0, -translation[2], translation[1]],
            [translation[2], 0, -translation[0]],
            [-translation[1], translation[0], 0],
        ]
    )
    adjoint_matrix = np.zeros((6, 6))
    adjoint_matrix[:3, :3] = rotation
    adjoint_matrix[3:, 3:] = rotation
    adjoint_matrix[3:, :3] = skew_matrix @ rotation
    return adjoint_matrix


def construct_transform_matrix(tcp_pose):
    """Construct the block transform matrix from a pose."""
    rotation = R.from_quat(tcp_pose[3:]).as_matrix()
    transform_matrix = np.zeros((6, 6))
    transform_matrix[:3, :3] = rotation
    transform_matrix[3:, 3:] = rotation
    return transform_matrix


def construct_homogeneous_matrix(tcp_pose):
    """Construct the homogeneous transformation matrix from a pose."""
    rotation = R.from_quat(tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    transform = np.zeros((4, 4))
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    transform[3, 3] = 1
    return transform


def construct_adjoint_matrix_from_euler(tcp_pose):
    """Construct the adjoint matrix for a pose with xyz Euler angles."""
    rotation = R.from_euler("xyz", tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    skew_matrix = np.array(
        [
            [0, -translation[2], translation[1]],
            [translation[2], 0, -translation[0]],
            [-translation[1], translation[0], 0],
        ]
    )
    adjoint_matrix = np.zeros((6, 6))
    adjoint_matrix[:3, :3] = rotation
    adjoint_matrix[3:, 3:] = rotation
    adjoint_matrix[3:, :3] = skew_matrix @ rotation
    return adjoint_matrix


def construct_homogeneous_matrix_from_euler(tcp_pose):
    """Construct the homogeneous transformation matrix from xyz Euler pose."""
    rotation = R.from_euler("xyz", tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    transform = np.zeros((4, 4))
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    transform[3, 3] = 1
    return transform
