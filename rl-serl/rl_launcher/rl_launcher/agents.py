"""Agent factories — forwarded from serl_launcher.

rl-serl reuses serl_launcher's algorithm implementations verbatim. This module
re-exports the agent factory functions so that examples import from
`rl_launcher.agents` instead of reaching into serl_launcher directly. If the
algorithm backend ever changes, only this layer needs to be updated.
"""
from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
    make_wandb_logger,
)

__all__ = [
    "make_sac_pixel_agent",
    "make_sac_pixel_agent_hybrid_single_arm",
    "make_sac_pixel_agent_hybrid_dual_arm",
    "make_trainer_config",
    "make_wandb_logger",
]
