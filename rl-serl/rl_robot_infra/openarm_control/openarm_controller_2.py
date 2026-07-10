"""Compatibility import for the refactored OpenArm controller.

The implementation now lives in openarm_control.controller. Keep this module so
older server imports continue to work while the main control path moves to the
new /control/* API.
"""

from openarm_control.controller import OpenArmController


__all__ = ["OpenArmController"]
