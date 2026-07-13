"""Deterministic camera and arm hardware doubles for offline testing."""

import time

import numpy as np

from openarm_control.types import ArmState


class SimulatedArmBus:
    """In-process replacement for one :class:`OpenArmCanBus`.

    It deliberately models only the adapter contract, not motor physics. Each MIT
    command advances the joints by a configurable fraction of the requested move,
    which is enough to exercise controller limiting, inactive-arm handling, gripper
    commands, diagnostics, and safety without importing ``openarm_can``.
    """

    response = 0.35

    def __init__(self, interface="sim", can_fd=False, motor_config=None):
        self.interface = str(interface)
        self.can_fd = bool(can_fd)
        self.motor_config = motor_config or {}
        self.q = np.zeros(7, dtype=float)
        self.dq = np.zeros(7, dtype=float)
        self.tau = np.zeros(7, dtype=float)
        self.gripper_q = np.zeros(1, dtype=float)
        self.enabled = np.ones(7, dtype=bool)
        self.gripper_enabled = np.ones(1, dtype=bool)
        self.temperature = np.ones(7, dtype=float) * 25.0
        self.last_command = None
        self.command_count = 0
        self._last_read = time.time()

    def read_state(self, timeout_us=500):
        del timeout_us
        now = time.time()
        return ArmState(
            q=self.q.copy(),
            dq=self.dq.copy(),
            tau=self.tau.copy(),
            t_mos=self.temperature.copy(),
            t_rotor=self.temperature.copy(),
            enabled=self.enabled.copy(),
            gripper_q=self.gripper_q.copy(),
            gripper_dq=np.zeros(1),
            gripper_tau=np.zeros(1),
            gripper_t_mos=np.ones(1) * 25.0,
            gripper_t_rotor=np.ones(1) * 25.0,
            gripper_enabled=self.gripper_enabled.copy(),
            timestamp=now,
        )

    def send_mit(self, command, gripper_position=None):
        previous_q = self.q.copy()
        self.q += self.response * (np.asarray(command.q, dtype=float) - self.q)
        self.dq = np.asarray(command.dq, dtype=float).copy()
        self.tau = np.asarray(command.tau_ff, dtype=float).copy()
        if gripper_position is not None:
            self.gripper_q[0] = float(gripper_position)
        self.last_command = command
        self.command_count += 1
        self._last_read = time.time()
        if not np.all(np.isfinite(self.q)):
            self.q = previous_q
            raise ValueError("non-finite simulated joint command")

    def hold_position(self):
        self.dq.fill(0.0)

    def query_params(self, rids):
        return {str(rid): [0.0] * 8 for rid in rids}


class MockCamera:
    """Camera source that returns black RGB frames."""

    def __init__(self, width=640, height=480, fps=30):
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        print(f"[MockCamera] Initialized {self.width}x{self.height}@{self.fps}fps")

    def read_rgb(self, viz=False):
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return img
