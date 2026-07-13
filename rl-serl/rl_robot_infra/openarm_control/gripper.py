"""Calibrated mapping between network gripper state and raw motor positions."""

import numpy as np


class GripperCalibration:
    """Own raw gripper units at the hardware boundary and expose boolean state."""

    def __init__(self, open_position, closed_position, open_threshold, close_threshold):
        self.open_position = np.asarray(open_position, dtype=np.float64).reshape(2)
        self.closed_position = np.asarray(closed_position, dtype=np.float64).reshape(2)
        self.open_threshold = np.asarray(open_threshold, dtype=np.float64).reshape(2)
        self.close_threshold = np.asarray(close_threshold, dtype=np.float64).reshape(2)
        self.closed = np.zeros(2, dtype=bool)

        closing_positive = self.closed_position > self.open_position
        valid = np.where(
            closing_positive,
            self.open_threshold < self.close_threshold,
            self.open_threshold > self.close_threshold,
        )
        if not np.all(valid):
            raise ValueError("gripper hysteresis thresholds do not follow motor direction")

    @classmethod
    def from_config(cls, config):
        return cls(
            open_position=config["open_position"],
            closed_position=config["closed_position"],
            open_threshold=config["open_threshold"],
            close_threshold=config["close_threshold"],
        )

    def target_from_closed(self, closed):
        closed = np.asarray(closed, dtype=bool).reshape(2)
        return np.where(closed, self.closed_position, self.open_position)

    def update_from_position(self, position):
        position = np.asarray(position, dtype=np.float64).reshape(2)
        closing_positive = self.closed_position > self.open_position
        close_reached = np.where(
            closing_positive,
            position >= self.close_threshold,
            position <= self.close_threshold,
        )
        open_reached = np.where(
            closing_positive,
            position <= self.open_threshold,
            position >= self.open_threshold,
        )
        self.closed = np.where(close_reached, True, np.where(open_reached, False, self.closed))
        return self.closed.copy()
