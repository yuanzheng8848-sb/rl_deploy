"""Task-scoped runtime artifact paths for rl-serl experiments.

All default runtime artifacts live under:

    examples/experiments/<exp_name>/

Entry points may still expose explicit path overrides, but their no-override
behavior should go through this module so demo collection, classifier training
and evaluation, and RLPD training all agree on the task directory layout.
"""
from pathlib import Path


EXPERIMENTS_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = EXPERIMENTS_DIR.parent
RL_SERL_ROOT = EXAMPLES_DIR.parent


def task_dir(exp_name: str) -> Path:
    """Return the root artifact directory for a task."""
    return EXPERIMENTS_DIR / str(exp_name)


def task_demo_root(exp_name: str) -> Path:
    return task_dir(exp_name) / "demo" / "collected"


def task_success_dir(exp_name: str) -> Path:
    return task_demo_root(exp_name) / "success"


def task_failure_dir(exp_name: str) -> Path:
    return task_demo_root(exp_name) / "failure"


def task_raw_image_root(exp_name: str) -> Path:
    return task_demo_root(exp_name) / "raw_images"


def task_raw_success_dir(exp_name: str) -> Path:
    return task_raw_image_root(exp_name) / "success"


def task_raw_failure_dir(exp_name: str) -> Path:
    return task_raw_image_root(exp_name) / "failure"


def task_classifier_ckpt_dir(exp_name: str) -> Path:
    return task_dir(exp_name) / "classifier_ckpt"


def task_bc_checkpoint_dir(exp_name: str) -> Path:
    return task_dir(exp_name) / "checkpoints_bc"


def task_bc_eval_dir(exp_name: str) -> Path:
    return task_dir(exp_name) / "bc_eval"


def task_rlpd_checkpoint_dir(exp_name: str) -> Path:
    return task_dir(exp_name) / "checkpoints_rlpd"
