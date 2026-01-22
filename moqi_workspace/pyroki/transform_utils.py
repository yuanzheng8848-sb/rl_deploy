from typing import Dict, List

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation, Slerp


def trans_quat_dict2trans_quat(pose_dict: Dict[str, Dict[str, float]]) -> npt.NDArray:
    pose = []
    for key in ["x", "y", "z"]:
        pose.append(pose_dict["position"][key])
    for key in ["x", "y", "z", "w"]:
        pose.append(pose_dict["orientation"][key])
    return np.array(pose)


def trans_quat2mat(trans_quat: npt.NDArray) -> npt.NDArray:
    tf = np.eye(4)
    tf[:3, :3] = Rotation.from_quat(trans_quat[3:]).as_matrix()
    tf[:3, 3] = trans_quat[:3]
    return tf


def mat2trans_quat(tf: npt.NDArray) -> npt.NDArray:
    trans_quat = np.zeros(7)
    trans_quat[:3] = tf[:3, 3]
    trans_quat[3:] = Rotation.from_matrix(tf[:3, :3]).as_quat()
    return trans_quat


def frame_unity2righthand(xyzqwqxqyqz: List[float]) -> npt.NDArray:
    """
    Note: The input is in the order of (x, y, z, qw, qx, qy, qz) in unity flavor,
    the output is in the order of (x, y, z, qx, qy, qz, qw) in right hand flavor.
    """
    x = xyzqwqxqyqz[2]
    y = -xyzqwqxqyqz[0]
    z = xyzqwqxqyqz[1]
    qx = -xyzqwqxqyqz[6]
    qy = xyzqwqxqyqz[4]
    qz = -xyzqwqxqyqz[5]
    qw = xyzqwqxqyqz[3]
    return np.array([x, y, z, qx, qy, qz, qw])


def orientation_dist(quat1: npt.NDArray, quat2: npt.NDArray) -> float:
    """
    Calculate the distance between two quaternions.
    The distance is defined as the angle between the two quaternions.
    """
    return (Rotation.from_quat(quat1).inv() * Rotation.from_quat(quat2)).magnitude()


def inv_tf(tf: npt.NDArray) -> npt.NDArray:
    """
    Invert a transformation matrix.
    """
    inv_tf = np.eye(4)
    inv_tf[:3, :3] = tf[:3, :3].T
    inv_tf[:3, 3] = -inv_tf[:3, :3] @ tf[:3, 3]
    return inv_tf


def xyz_rxyzw2rwxyz_xyz(xyz_rxyzw: npt.NDArray | list[float]) -> npt.NDArray:
    xyz_rxyzw = np.asarray(xyz_rxyzw)
    return np.array(xyz_rxyzw[..., [6, 3, 4, 5, 0, 1, 2]])


def rwxyz_xyz2xyz_rxyzw(rwxyz_xyz: npt.NDArray | list[float]) -> npt.NDArray:
    rwxyz_xyz = np.asarray(rwxyz_xyz)
    return np.array(rwxyz_xyz[..., [4, 5, 6, 1, 2, 3, 0]])


def rwxyz_xyz2mat(rwxyz_xyz: npt.NDArray) -> npt.NDArray:
    return trans_quat2mat(rwxyz_xyz2xyz_rxyzw(rwxyz_xyz))


def interp_transform(tf_start, tf_end, num) -> list[npt.NDArray]:
    rot_mat_start_end = [tf_start[:3, :3], tf_end[:3, :3]]
    rot_interp = Slerp([0, num - 1], Rotation.from_matrix(rot_mat_start_end))
    trans_interp = np.linspace(tf_start[:3, 3], tf_end[:3, 3], num)
    tf_interp = []
    for idx in range(num):
        transform = np.eye(4)
        transform[:3, :3] = rot_interp(idx).as_matrix()
        transform[:3, 3] = trans_interp[idx]
        tf_interp.append(transform)
    return tf_interp
