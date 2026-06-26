"""Task-agnostic gym wrappers — forwarded from serl_launcher.

These are the generic wrappers that hil-serl keeps in serl_launcher (not robot
specific). OpenArm-specific wrappers (DualRelativeFrame, DualSpacemouseIntervention,
etc.) live in rl_robot_infra.openarm_env.envs.wrappers instead.
"""
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

__all__ = ["ChunkingWrapper", "SERLObsWrapper"]
