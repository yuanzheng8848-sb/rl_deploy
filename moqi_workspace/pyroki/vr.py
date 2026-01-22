import queue
import socket
import struct
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
import numpy.typing as npt
import yaml  # type: ignore
from loguru import logger
from scipy.spatial.transform import Rotation

import transform_utils as tf_utils


class VRUpperBodyTeleop:
    SEQUENCE = ["left", "right", "chest"]

    @dataclass
    class Quest3State:
        button_top: list[bool]
        button_bottom: list[bool]
        thumb_stick: list[int]
        trigger_index: list[float]
        trigger_hand: list[float]
        pose_hand: list[npt.NDArray]
        pose_abs_hand: list[npt.NDArray]
        pose_abs_head: npt.NDArray

    class EEState(Enum):
        STATE_IDLE = 1
        STATE_MOVE = 2
        STATE_RESET = 3

    def __init__(
        self,
        config: dict,
        ip_vr: str,
        reference_pose_func: Callable[[], npt.NDArray | None],
        reference_path: str | None = None,
    ) -> None:
        # Declare ROS parameters
        self._load_config(config, reference_path)
        self._get_reference_pose = reference_pose_func
        self._vr_ip = ip_vr
        self._ee_states = [self.EEState.STATE_IDLE] * 3
        self._reset_traj: list[list[npt.NDArray] | None] = [None] * 3
        self._tf_base_ee_ref: list[npt.NDArray | None] = [None] * 3
        self._tf_ctrl_vr_sta: list[npt.NDArray | None] = [None] * 3
        self._pose_target_cur: list[npt.NDArray | None] = [None] * 3

        self._queue_vr_cmd: queue.Queue[tuple[npt.NDArray, npt.NDArray]] = queue.Queue(
            maxsize=1
        )
        self._shutdown_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._queue_debug: queue.Queue[npt.NDArray] = queue.Queue(maxsize=1)

        self._start()

    def _load_config(self, config: dict, reference_path: str | None = None) -> None:
        config_udp = config["udp"]
        self._host_port = config_udp["port"]["host"]
        self._vr_port = config_udp["port"]["vr"]
        self._broadcast_msg = config_udp["broadcast_msg"].encode()
        self._max_retries = config_udp["max_retries"]
        self._load_reset_config(config["reset"])
        self._load_calib_data(config["calib_data_path"], reference_path)
        self._load_bounding(config["bounding"])
        self._gripper_cmd_limit = config["gripper_command_limit"]

    def _load_reset_config(self, config: dict) -> None:
        self._tf_reset = []
        self._lift_height = config["lift_height"]
        for name in self.SEQUENCE:
            reset_pose = tf_utils.trans_quat_dict2trans_quat(config["poses"][name])
            reset_pose[2] += self._lift_height
            self._tf_reset.append(tf_utils.trans_quat2mat(reset_pose))
        self._reset_tolerance = config["tolerance"]
        self._reset_sample_num = config["samples"]
        self._reset_time_step = config["time"] / self._reset_sample_num
        self._reset_start_time: float | None = None

    def _load_calib_data(
        self, calib_data_path: str, reference_path: str | None = None
    ) -> None:
        self._tf_ctrl_ee_calib = []
        if not calib_data_path.startswith("/"):
            assert reference_path is not None, (
                "reference_path must be provided when use relative path for calibration data"
            )
            if not reference_path.endswith("/"):
                reference_path += "/"
            calib_data_path = reference_path + calib_data_path
        for name in self.SEQUENCE:
            calib_path = Path(calib_data_path) / f"{name}.yaml"
            if not calib_path.exists():
                raise FileNotFoundError(
                    f"Calibration file {calib_path} does not exist."
                )
            with open(calib_path, "r") as f:
                calib_data = yaml.safe_load(f)
            self._tf_ctrl_ee_calib.append(
                tf_utils.trans_quat2mat(tf_utils.trans_quat_dict2trans_quat(calib_data))
            )
        # x-forwward & z-up to z-forward & x-up
        self._rot_mat_frame_align = np.array(
            [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]
        )

    def _load_bounding(self, config: dict) -> None:
        self._trans_min = np.array(config["translation"]["min"])
        self._trans_max = np.array(config["translation"]["max"])
        self._orient_range = config["orientation"]["range"]
        self._orient_ref = Rotation.from_euler(
            "xyz", config["orientation"]["reference"]
        )

    def _decode(self, data: npt.NDArray) -> Quest3State:
        # 48 * 7 (x, y, z, qw, qx, qy, qz), the first 3 are commands
        # sender is a unity app from Open Loong
        return self.Quest3State(
            button_top=[bool(data[1, 6] == 1), bool(data[1, 1] == 1)],
            button_bottom=[bool(data[1, 4] == 1), bool(data[1, 5] == 1)],
            thumb_stick=[int(data[2, 1]), int(data[2, 2])],
            trigger_index=[data[1, 0], data[2, 2]],
            trigger_hand=[data[1, 2], data[1, 3]],
            pose_hand=[
                tf_utils.frame_unity2righthand(data[0]),
                tf_utils.frame_unity2righthand(data[24]),
            ],
            pose_abs_hand=[
                tf_utils.frame_unity2righthand(data[25]),
                tf_utils.frame_unity2righthand(data[26]),
            ],
            pose_abs_head=tf_utils.frame_unity2righthand(data[27]),
        )

    def _calibrate(
        self,
        tf_vr_ctrl: npt.NDArray,
        idx: int,
    ) -> npt.NDArray:
        """
        Apply calibration and seperate rotation and translation.
        """
        tf_vr_wrist = tf_vr_ctrl @ self._tf_ctrl_ee_calib[idx]
        if idx != 2:
            tf_vr_wrist[:3, :3] = tf_vr_wrist[:3, :3] @ self._rot_mat_frame_align
        return tf_vr_wrist

    def _pose_bounding(self, tf: npt.NDArray) -> npt.NDArray:
        """
        ! fixme: can leads to rotation jerk
        Clip the pose to be within the safety range.
        """
        raise NotImplementedError()
        tf = tf.copy()  # type: ignore
        tf[:3, 3] = np.clip(tf[:3, 3], self._trans_min, self._trans_max)
        orient = Rotation.from_matrix(tf[:3, :3])
        orient_diff = self._orient_ref.inv() * orient
        if orient_diff.magnitude() > self._orient_range:
            orient_clip = Rotation.from_rotvec(
                orient_diff.as_rotvec() * self._orient_range / orient_diff.magnitude()
            )
            tf[:3, :3] = (self._orient_ref * orient_clip).as_matrix()
        return tf

    def _state_polling(
        self,
        quest3state: Quest3State,
        ee_poses_cur: npt.NDArray,
        tf_vr_ctrl_uncal: list[npt.NDArray],
    ) -> None:
        # reset
        if np.any(quest3state.button_top):
            # if y or b is pressed, go to reset
            if self._ee_states[0] != self.EEState.STATE_RESET:
                self._reset_start_time = time.time()
                for idx in range(3):
                    self._ee_states[idx] = self.EEState.STATE_RESET
                    # interpolate ee pose from current to reset
                    traj = tf_utils.interp_transform(
                        tf_utils.trans_quat2mat(ee_poses_cur[idx]),
                        self._tf_reset[idx],
                        self._reset_sample_num + 1,
                    )
                    self._reset_traj[idx] = traj[1:]
                    self._tf_ctrl_vr_sta[idx] = None
        if self._ee_states[0] == self.EEState.STATE_RESET:
            assert (
                self._ee_states[1] == self.EEState.STATE_RESET
                and self._ee_states[2] == self.EEState.STATE_RESET
            ), "All end effectors should be in reset state at the same time."
            done_reset = True
            for idx in range(3):
                tf_diff = tf_utils.trans_quat2mat(ee_poses_cur[idx]) @ tf_utils.inv_tf(
                    self._tf_reset[idx]
                )
                rot_diff = Rotation.from_matrix(tf_diff[:3, :3]).magnitude()
                trans_diff = tf_diff[:3, 3]
                # trans_diff[2] += self._lift_height
                trans_diff = np.linalg.norm(trans_diff)
                logger.warning(f"{self.SEQUENCE[idx]} - {trans_diff}, {rot_diff}")
                if (
                    rot_diff > self._reset_tolerance["rotation"]
                    or trans_diff > self._reset_tolerance["translation"]
                ):
                    done_reset = False
                    break
            if done_reset:
                logger.info("Reset done.")
                self._reset_start_time = None
                for idx in range(3):
                    self._reset_traj[idx] = None
                    self._ee_states[idx] = self.EEState.STATE_IDLE
                return

        # idle and move
        for idx in range(3):
            if self._ee_states[idx] == self.EEState.STATE_IDLE:
                # set target pose as current pose when started
                if self._pose_target_cur[idx] is None:
                    self._pose_target_cur[idx] = np.array(ee_poses_cur[idx])
                flag_to_move = False
                if idx < 2:
                    # press trigger to control corresponding arm
                    if quest3state.trigger_hand[idx]:
                        flag_to_move = True
                else:
                    # press both triggers to control chest
                    if quest3state.trigger_hand[0] and quest3state.trigger_hand[1]:
                        flag_to_move = True
                if flag_to_move:
                    pose_target_last = np.array(self._pose_target_cur[idx])
                    self._ee_states[idx] = self.EEState.STATE_MOVE
                    self._tf_base_ee_ref[idx] = tf_utils.trans_quat2mat(
                        pose_target_last
                    )
                    self._tf_ctrl_vr_sta[idx] = tf_utils.inv_tf(
                        self._calibrate(tf_vr_ctrl_uncal[idx], idx)
                    )
            elif self._ee_states[idx] == self.EEState.STATE_MOVE:
                flag_to_idle = False
                if idx < 2:
                    if not quest3state.trigger_hand[idx]:
                        flag_to_idle = True
                else:
                    if (
                        not quest3state.trigger_hand[0]
                        or not quest3state.trigger_hand[1]
                    ):
                        flag_to_idle = True
                if flag_to_idle:
                    self._ee_states[idx] = self.EEState.STATE_IDLE
                    self._tf_base_ee_ref[idx] = None
                    self._tf_ctrl_vr_sta[idx] = None

    def _action_polling(
        self, quest3state: Quest3State, tf_vr_ctrl_uncal: list[npt.NDArray]
    ) -> tuple[list[npt.NDArray], npt.NDArray]:
        # reset interpolation index
        idx_reset = None
        if self._ee_states[0] == self.EEState.STATE_RESET:
            assert (
                self._ee_states[1] == self.EEState.STATE_RESET
                and self._ee_states[2] == self.EEState.STATE_RESET
            ), "All end effectors should be in reset state at the same time."
            assert self._reset_start_time is not None
            idx_reset = (time.time() - self._reset_start_time) // self._reset_time_step
            idx_reset = int(np.clip(idx_reset, 0, self._reset_sample_num - 1))
            logger.warning(idx_reset)

        # ee pose
        poses_tgt = []
        for idx in range(3):
            pose_target: npt.NDArray | None = None
            if self._ee_states[idx] == self.EEState.STATE_RESET:
                traj = self._reset_traj[idx]
                assert traj is not None and idx_reset is not None
                pose_target = tf_utils.mat2trans_quat(traj[idx_reset])
                self._pose_target_cur[idx] = np.array(pose_target)
            elif self._ee_states[idx] == self.EEState.STATE_IDLE:
                pose_target = self._pose_target_cur[idx]
            elif self._ee_states[idx] == self.EEState.STATE_MOVE:
                tf_base_ee_ref = np.array(self._tf_base_ee_ref[idx])
                tf_vr_ctrl = self._calibrate(tf_vr_ctrl_uncal[idx], idx)
                tf_diff = self._tf_ctrl_vr_sta[idx] @ tf_vr_ctrl
                # if idx < 2:
                #     tf_base_ee_tgt = self._pose_bounding(tf_base_ee_ref @ tf_diff)
                # else:
                #     tf_base_ee_tgt = tf_base_ee_ref @ tf_diff
                tf_base_ee_tgt = tf_base_ee_ref @ tf_diff
                pose_target = tf_utils.mat2trans_quat(tf_base_ee_tgt)
                self._pose_target_cur[idx] = np.array(pose_target)
            assert pose_target is not None, (
                f"Target pose is None for {self.SEQUENCE[idx]}."
            )
            poses_tgt.append(pose_target)

        # gripper, raw data from vr: release is 0, pressed is 1
        if self._ee_states[0] == self.EEState.STATE_RESET:
            gripper_width = np.array([self._gripper_cmd_limit["open"]] * 2)
        else:
            gripper_width = (
                np.array(quest3state.trigger_index)
                * (self._gripper_cmd_limit["close"] - self._gripper_cmd_limit["open"])
                + self._gripper_cmd_limit["open"]
            )
        return poses_tgt, gripper_width

    def _process(
        self, quest3state: Quest3State
    ) -> tuple[npt.NDArray, npt.NDArray] | None:
        # x, y, z, qx, qy, qz, qw
        ee_poses_cur = self._get_reference_pose()
        if ee_poses_cur is None:
            logger.debug("Reference pose is None, waiting for initialization...")
            return None
        # vr_poses = np.array(quest3state.pose_abs_hand + [quest3state.pose_abs_head])
        tf_vr_ctrl_uncal = []
        for trans_quat_vr_ee in quest3state.pose_abs_hand + [quest3state.pose_abs_head]:
            tf_vr_ctrl_uncal.append(tf_utils.trans_quat2mat(trans_quat_vr_ee))
        # state machine
        self._state_polling(quest3state, ee_poses_cur, tf_vr_ctrl_uncal)
        # action according to state
        poses_target, gripper_width = self._action_polling(
            quest3state, tf_vr_ctrl_uncal
        )
        # convert from xyz-rxyzw to rwxyz-xyz
        rwxyz_xyz_tgt = []
        for pose in poses_target:
            rwxyz_xyz_tgt.append(tf_utils.xyz_rxyzw2rwxyz_xyz(pose))
        return np.array(rwxyz_xyz_tgt), gripper_width

    def _vr_loop(self) -> None:
        class ConnectState(Enum):
            CONNECT_STATE = 1
            DISCONNECT_STATE = 2
            ERROR_STATE = 3

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.bind(("0.0.0.0", self._host_port))
        # sock.bind(("10.255.8.41", self._host_port))
        sock.settimeout(1.0)  # in seconds
        connect_state = ConnectState.DISCONNECT_STATE
        retry_count = 0
        logger.info(f"VR socket bound to port {self._host_port}, waiting for data from {self._vr_ip}:{self._vr_port}")
        while not self._shutdown_event.is_set():
            try:
                if connect_state == ConnectState.DISCONNECT_STATE:
                    logger.info(f"Sending broadcast message to {self._vr_ip}:{self._vr_port}")
                    sock.sendto(self._broadcast_msg, (self._vr_ip, self._vr_port))
                    retry_count = 0
                    logger.info("Waiting for initial response from Quest...")
                    data, addr = sock.recvfrom(2048)
                    logger.info(f"Received message: {len(data)} bytes from {addr}")
                    connect_state = ConnectState.CONNECT_STATE
                else:
                    data_bytes, addr = sock.recvfrom(8192)
                    data_length = len(data_bytes)
                    if data_length == 4 * 48 * 7:
                        float_array = struct.unpack(f"{48 * 7}f", data_bytes)
                        raw_data = np.array(float_array).reshape((48, 7))
                        quest3state = self._decode(raw_data)
                        # self._debug_put(quest3state)
                        result = self._process(quest3state)
                        if result is not None:
                            try:
                                self._queue_vr_cmd.put_nowait(result)
                            except queue.Full:
                                self._queue_vr_cmd.get_nowait()
                                self._queue_vr_cmd.put_nowait(result)
                        else:
                            logger.debug("VR result is None (reference pose not ready)")
                retry_count = 0
            except socket.timeout:
                retry_count += 1
                if retry_count >= self._max_retries:
                    logger.warning(f"Data not received for {retry_count} times.")
                    connect_state = ConnectState.DISCONNECT_STATE
                continue
            except socket.error as e:
                logger.error(f"Socket error occurred: {e}")
                break
        sock.close()

    def _debug_put(self, quest3state: Quest3State) -> None:
        poses: list[npt.NDArray] = []
        stamp = time.time()
        for pose in [
            quest3state.pose_abs_hand[0],
            quest3state.pose_abs_hand[1],
            quest3state.pose_abs_head,
        ]:
            poses.append(np.concatenate([np.array(pose), [stamp]]))
        pose_arr = np.array(poses)
        try:
            self._queue_debug.put_nowait(pose_arr)
        except queue.Full:
            self._queue_debug.get_nowait()
            self._queue_debug.put_nowait(pose_arr)

    def db_get_raw_data(self) -> npt.NDArray | None:
        try:
            return self._queue_debug.get_nowait()
        except queue.Empty:
            return None

    def _start(self) -> None:
        # self._zmq_interface.start()
        if self._thread is None:
            self._thread = threading.Thread(target=self._vr_loop, daemon=True)
            self._thread.start()
            logger.info("VR teleoperation thread started.")
        else:
            logger.info("VR teleoperation thread is already running.")

    def stop(self) -> None:
        self._shutdown_event.set()
        if self._thread:
            logger.info("Shutdown VR thread...")
            self._thread.join()
            logger.info("VR thread exited.")
        # self._zmq_interface.stop()

    def get_vr_command(self) -> tuple[npt.NDArray, npt.NDArray] | None:
        try:
            return self._queue_vr_cmd.get_nowait()
        except queue.Empty:
            return None

    def wait_for_initial_states(self) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Wait for the reference pose to be available.
        """
        command: tuple[npt.NDArray, npt.NDArray] | None = None
        cnt = 0
        while True:
            command = self.get_vr_command()
            if command is not None:
                break
            time.sleep(0.1)
            if cnt % 10 == 0:
                logger.info("Waiting for reference pose...")
            cnt += 1
        logger.info("Reference pose received.")
        return command
