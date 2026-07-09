from .continuous.bc import BCAgent
from .continuous.sac import SACAgent
from .continuous.sac_hybrid_single import SACAgentHybridSingleArm
from .continuous.sac_hybrid_dual import SACAgentHybridDualArm
from rl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
    make_wandb_logger,
)

agents = {
    "bc": BCAgent,
    "sac": SACAgent,
    "sac_hybrid_single": SACAgentHybridSingleArm,
    "sac_hybrid_dual": SACAgentHybridDualArm,
}

__all__ = [
    "BCAgent",
    "SACAgent",
    "SACAgentHybridSingleArm",
    "SACAgentHybridDualArm",
    "agents",
    "make_sac_pixel_agent",
    "make_sac_pixel_agent_hybrid_single_arm",
    "make_sac_pixel_agent_hybrid_dual_arm",
    "make_trainer_config",
    "make_wandb_logger",
]
