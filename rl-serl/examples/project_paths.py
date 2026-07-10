"""Local package paths for running examples without installation."""

import sys
from pathlib import Path


RL_SERL_ROOT = Path(__file__).resolve().parents[1]

for path in (RL_SERL_ROOT / "rl_launcher", RL_SERL_ROOT / "rl_robot_infra"):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)
