#!/usr/bin/env python3
"""Train an RLPD-compatible behavior cloning checkpoint for rl-serl."""
import compat  # noqa: F401  (sys.path + CUDA/JAX patches; must be first)

import csv
import os

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints

from rl_launcher.agents import make_sac_pixel_agent_hybrid_dual_arm, make_wandb_logger
from rl_launcher.data import MemoryEfficientReplayBufferDataStore

from bc_utils import load_demo_dir_into_buffer
from experiments.artifacts import task_bc_checkpoint_dir, task_success_dir
from experiments.mappings import CONFIG_MAPPING


FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", "openarm_pickplace", "Experiment name in CONFIG_MAPPING.")
flags.DEFINE_string("bc_checkpoint_path", None, "Defaults to task checkpoints_bc.")
flags.DEFINE_string("success_dir", None, "Defaults to task success demo directory.")
flags.DEFINE_integer("train_steps", 20000, "Number of BC update steps.")
flags.DEFINE_integer("batch_size", None, "Defaults to task config batch_size.")
flags.DEFINE_integer("checkpoint_period", 1000, "Checkpoint save period.")
flags.DEFINE_integer("plot_period", 500, "Period for generating training plots.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("debug", False, "Disable wandb when true.")
flags.DEFINE_boolean("skip_zero_actions", True, "Skip zero-action demo transitions.")
flags.DEFINE_float(
    "bc_active_xyz_threshold",
    0.05,
    "XYZ action norm above which a sample is treated as active for BC weighting.",
)
flags.DEFINE_float(
    "bc_active_xyz_weight",
    3.0,
    "Extra BC loss weight for active XYZ action samples.",
)
flags.DEFINE_float(
    "bc_xyz_norm_loss_weight",
    0.50,
    "Weight for absolute XYZ action magnitude loss.",
)
flags.DEFINE_float(
    "bc_xyz_relative_norm_loss_weight",
    0.15,
    "Weight for active-only relative XYZ action magnitude loss.",
)
flags.DEFINE_float(
    "bc_xyz_cosine_loss_weight",
    0.02,
    "Small auxiliary direction loss weight for active XYZ actions.",
)


devices = jax.local_devices()
if hasattr(jax.sharding, "PositionalSharding"):
    sharding = jax.sharding.PositionalSharding(devices).replicate()
else:
    sharding = jax.sharding.SingleDeviceSharding(devices[0])


def print_green(text):
    print(f"\033[92m{text}\033[00m")


def create_agent(config, env):
    agent = make_sac_pixel_agent_hybrid_dual_arm(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
        discount=config.discount,
        bc_active_xyz_threshold=FLAGS.bc_active_xyz_threshold,
        bc_active_xyz_weight=FLAGS.bc_active_xyz_weight,
        bc_xyz_norm_loss_weight=FLAGS.bc_xyz_norm_loss_weight,
        bc_xyz_relative_norm_loss_weight=FLAGS.bc_xyz_relative_norm_loss_weight,
        bc_xyz_cosine_loss_weight=FLAGS.bc_xyz_cosine_loss_weight,
    )
    return jax.device_put(jax.tree_util.tree_map(jnp.array, agent), sharding)


def write_log_row(csv_path, row):
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def scalarize_info(info):
    """Extract scalar metrics from update info dict, flattening nested dicts."""
    result = {}

    def flatten_dict(d, prefix=""):
        """Recursively flatten nested dictionaries."""
        for key, value in d.items():
            new_key = f"{prefix}/{key}" if prefix else key

            # If value is a dict, recurse
            if isinstance(value, dict):
                flatten_dict(value, new_key)
            else:
                # Try to extract scalar
                try:
                    arr = np.asarray(jax.device_get(value))
                    if arr.shape == ():
                        result[new_key] = float(arr)
                    elif arr.shape == (1,):
                        result[new_key] = float(arr[0])
                except Exception:
                    continue

    flatten_dict(info)
    return result


def plot_training_curves(csv_path, checkpoint_path, current_step):
    """Generate detailed training plots from CSV log."""
    if not os.path.exists(csv_path):
        return

    # Read training log
    data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if key not in data:
                    data[key] = []
                try:
                    data[key].append(float(value))
                except ValueError:
                    data[key].append(value)

    if 'step' not in data or len(data['step']) == 0:
        return

    steps = np.array(data['step'])

    # Prepare plot directory
    plot_dir = os.path.join(checkpoint_path, "training_plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Remove duplicate metrics (keep the cleaner name)
    duplicate_map = {
        'actor/actor_loss': 'actor/bc_loss',  # Keep bc_loss
        'grasp_critic/grasp_critic_loss': 'grasp_critic/bc_grasp_loss',  # Keep bc_grasp_loss
    }

    # Filter out duplicates
    filtered_data = {}
    for key, values in data.items():
        if key in duplicate_map:
            continue  # Skip the duplicate
        filtered_data[key] = values

    data = filtered_data

    # Find all numeric columns (excluding step)
    numeric_columns = []
    for key in data.keys():
        if key != 'step' and len(data[key]) > 0:
            try:
                float(data[key][0])
                numeric_columns.append(key)
            except (ValueError, TypeError):
                pass

    if not numeric_columns:
        print("[BC] No numeric metrics found in training log to plot.")
        return

    # Group metrics by type (more specific grouping for BC training)
    metric_groups = {
        'BC Loss': [k for k in numeric_columns if 'bc_loss' in k.lower() or 'loss' in k.lower()],
        'BC Error (MSE)': [k for k in numeric_columns if 'mse' in k.lower() or 'mae' in k.lower()],
        'Gripper Accuracy': [k for k in numeric_columns if 'accuracy' in k.lower()],
        'Learning Rates': [k for k in numeric_columns if 'lr' in k.lower()],
    }

    # Remove empty groups
    metric_groups = {k: v for k, v in metric_groups.items() if v}

    # Determine subplot layout
    n_groups = len(metric_groups)
    if n_groups == 0:
        return

    n_cols = min(2, n_groups)
    n_rows = (n_groups + n_cols - 1) // n_cols

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    if n_groups == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    fig.suptitle(f'BC Training Progress (Step {current_step})', fontsize=16, fontweight='bold')

    # Plot each group
    for idx, (group_name, metrics) in enumerate(metric_groups.items()):
        ax = axes[idx]
        for metric in metrics:
            # Clean up label for display
            label = metric.replace('actor/', '').replace('grasp_critic/', 'gripper_').replace('_', ' ').title()
            ax.plot(steps, data[metric], label=label, linewidth=2, alpha=0.8)

        ax.set_xlabel('Training Step')
        ax.set_ylabel('Value')
        ax.set_title(group_name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_groups, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(plot_dir, f'training_curves_step_{current_step}.png')
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()

    # Also create a summary plot for the last step
    if current_step == steps[-1]:
        create_training_summary_plot(data, steps, plot_dir)

    print(f"[BC] Saved training plot: {plot_path}")


def create_training_summary_plot(data, steps, plot_dir):
    """Create a comprehensive summary plot at the end of training."""
    # Remove duplicate metrics
    duplicate_keys = ['actor/actor_loss', 'grasp_critic/grasp_critic_loss']
    data = {k: v for k, v in data.items() if k not in duplicate_keys}

    # Find key metrics
    loss_metrics = [k for k in data.keys() if 'loss' in k.lower() and k != 'step']
    accuracy_metrics = [k for k in data.keys() if 'acc' in k.lower() or 'accuracy' in k.lower()]
    error_metrics = [k for k in data.keys() if 'mse' in k.lower() or 'mae' in k.lower()]

    # Count non-empty groups
    groups = []
    if loss_metrics:
        groups.append(('Training Losses', loss_metrics))
    if error_metrics:
        groups.append(('Prediction Errors', error_metrics))
    if accuracy_metrics:
        groups.append(('Gripper Accuracy', accuracy_metrics))

    if not groups:
        # If no typical training metrics, just show all numeric metrics
        all_metrics = [k for k in data.keys() if k != 'step' and isinstance(data[k][0], (int, float))]
        if all_metrics:
            groups = [('Training Metrics', all_metrics)]

    if not groups:
        return

    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 5))
    if n_groups == 1:
        axes = [axes]

    fig.suptitle('BC Training Summary', fontsize=16, fontweight='bold')

    for idx, (group_name, metrics) in enumerate(groups):
        ax = axes[idx]
        for metric in metrics:
            # Clean up label
            label = metric.replace('actor/', '').replace('grasp_critic/', 'gripper_').replace('_', ' ').title()
            ax.plot(steps, data[metric], label=label, linewidth=2, alpha=0.8)

        ax.set_xlabel('Training Step')
        ax.set_ylabel('Value')
        ax.set_title(group_name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    summary_path = os.path.join(plot_dir, 'training_summary.png')
    plt.savefig(summary_path, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"[BC] Saved training summary: {summary_path}")



def main(_):
    if FLAGS.exp_name not in CONFIG_MAPPING:
        raise ValueError(f"Experiment {FLAGS.exp_name!r} not found in CONFIG_MAPPING.")

    config = CONFIG_MAPPING[FLAGS.exp_name]()
    checkpoint_path = os.path.abspath(
        FLAGS.bc_checkpoint_path or task_bc_checkpoint_dir(FLAGS.exp_name)
    )
    success_dir = os.path.abspath(FLAGS.success_dir or task_success_dir(FLAGS.exp_name))
    batch_size = FLAGS.batch_size or config.batch_size
    os.makedirs(checkpoint_path, exist_ok=True)

    print_green(f"[BC] exp={FLAGS.exp_name}")
    print_green(f"[BC] success demos={success_dir}")
    print_green(f"[BC] checkpoint_path={checkpoint_path}")

    env = config.get_environment(fake_env=True, classifier=False)
    agent = create_agent(config, env)

    buffer = MemoryEfficientReplayBufferDataStore(
        env.observation_space,
        env.action_space,
        capacity=config.replay_buffer_capacity,
        image_keys=config.image_keys,
        include_grasp_penalty=True,
    )
    files, transitions, skipped = load_demo_dir_into_buffer(
        success_dir,
        buffer,
        skip_zero_action=FLAGS.skip_zero_actions,
    )
    print_green(
        f"[BC] loaded success demos: {files} files, {transitions} transitions, {skipped} skipped"
    )
    if len(buffer) == 0:
        raise ValueError(f"No usable success demo transitions found in {success_dir}")

    latest = checkpoints.latest_checkpoint(checkpoint_path)
    start_step = 0
    if latest:
        agent = agent.replace(
            state=checkpoints.restore_checkpoint(checkpoint_path, agent.state)
        )
        start_step = int(os.path.basename(latest).replace("checkpoint_", "")) + 1
        print_green(f"[BC] resumed checkpoint {latest}")

    iterator = buffer.get_iterator(
        sample_args={"batch_size": batch_size, "pack_obs_and_next_obs": True},
        device=sharding,
    )
    wandb_logger = make_wandb_logger(
        project="rl-serl-bc", description=FLAGS.exp_name, debug=FLAGS.debug
    )
    csv_path = os.path.join(checkpoint_path, "training_log.csv")

    for step in tqdm.tqdm(range(start_step, FLAGS.train_steps), desc="BC training"):
        batch = next(iterator)
        agent, info = agent.update(
            batch,
            networks_to_update=frozenset({"actor", "grasp_critic"}),
            bc_mode=True,
        )
        if step % 10 == 0:
            row = {"step": step, **scalarize_info(info)}
            write_log_row(csv_path, row)
            if wandb_logger is not None:
                wandb_logger.log(row, step=step)

        # Generate training plots periodically
        if step > 0 and step % FLAGS.plot_period == 0:
            plot_training_curves(csv_path, checkpoint_path, step)

        if step > 0 and step % FLAGS.checkpoint_period == 0:
            checkpoints.save_checkpoint(
                checkpoint_path, agent.state, step=step, keep=5, overwrite=True
            )
            print_green(f"[BC] saved checkpoint at step {step}")

    checkpoints.save_checkpoint(
        checkpoint_path, agent.state, step=FLAGS.train_steps, keep=5, overwrite=True
    )

    # Generate final training plots
    plot_training_curves(csv_path, checkpoint_path, FLAGS.train_steps)

    print_green("[BC] training complete")
    print_green(f"[BC] Training plots saved in: {os.path.join(checkpoint_path, 'training_plots')}")
    env.close()


if __name__ == "__main__":
    app.run(main)
