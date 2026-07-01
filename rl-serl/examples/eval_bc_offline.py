#!/usr/bin/env python3
"""Offline BC checkpoint evaluation against recorded demo transitions."""
import compat  # noqa: F401  (sys.path + CUDA/JAX patches; must be first)

import csv
import json
import os
from pathlib import Path

import cv2
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from absl import app, flags
from flax.training import checkpoints
from matplotlib.backends.backend_pdf import PdfPages

from rl_launcher.agents import make_sac_pixel_agent_hybrid_dual_arm

from bc_utils import (
    continuous_action,
    gripper_classes,
    load_trajectories,
    prepare_transition,
    summarize,
)
from experiments.artifacts import (
    task_bc_checkpoint_dir,
    task_bc_eval_dir,
    task_success_dir,
)
from experiments.mappings import CONFIG_MAPPING


FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", "openarm_pickplace", "Experiment name in CONFIG_MAPPING.")
flags.DEFINE_string("bc_checkpoint_path", None, "Defaults to task checkpoints_bc.")
flags.DEFINE_integer("checkpoint_step", 0, "Checkpoint step, 0 means latest.")
flags.DEFINE_string("success_dir", None, "Defaults to task success demo directory.")
flags.DEFINE_string("output_dir", None, "Defaults to task bc_eval/offline.")
flags.DEFINE_integer("max_trajs", 0, "0 means all trajectories per split.")
flags.DEFINE_integer("high_error_count", 50, "Number of high-error frames to export.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("argmax", True, "Use deterministic action mode.")
flags.DEFINE_boolean("generate_plots", True, "Generate visualization plots in PDF.")
flags.DEFINE_boolean(
    "plot_trajectories", True, "Plot predicted vs real end-effector trajectory comparison (PNG)."
)
flags.DEFINE_integer(
    "traj_count", 0, "Number of trajectories to plot (0 = all). Worst-MSE first."
)
flags.DEFINE_float(
    "moving_step_threshold",
    0.001,
    "TCP real-step norm threshold in meters for relative/direction moving-step metrics.",
)
flags.DEFINE_float(
    "active_action_threshold",
    0.1,
    "Demo action xyz norm threshold for relative/direction action diagnostics.",
)
# ---- State / action xyz layout ----
# SERLObsWrapper flattens a gym.spaces.Dict, whose keys are ordered by space key.
# For proprio keys {gripper_pose, tcp_pose, tcp_vel}, flattened state is:
#   gripper_pose=(2,), tcp_pose=(2, 6), tcp_vel=(2, 6)
# so tcp_pose left=[2:8] and right=[8:14].
STATE_LEFT_POSE = slice(2, 8)
STATE_RIGHT_POSE = slice(8, 14)
# Raw 14-d action: left ee twist=[0:6], left gripper=6, right ee twist=[7:13], right gripper=13.
ACT_LEFT_XYZ = slice(0, 3)
ACT_RIGHT_XYZ = slice(7, 10)


def print_green(text):
    print(f"\033[92m{text}\033[00m")


def create_agent(config, env):
    return make_sac_pixel_agent_hybrid_dual_arm(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
        discount=config.discount,
    )


def restore_agent(agent, checkpoint_path):
    if FLAGS.checkpoint_step:
        state = checkpoints.restore_checkpoint(
            checkpoint_path, agent.state, step=FLAGS.checkpoint_step
        )
    else:
        latest = checkpoints.latest_checkpoint(checkpoint_path)
        if not latest:
            raise FileNotFoundError(f"No BC checkpoint found in {checkpoint_path}")
        state = checkpoints.restore_checkpoint(checkpoint_path, agent.state)
    return agent.replace(state=state)


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_frame(out_dir, rank, row, obs):
    image = obs.get("image_primary") if isinstance(obs, dict) else None
    if image is None:
        return
    image = np.asarray(image)
    if image.shape == (1, 128, 128, 3):
        image = image[0]
    if image.shape[-1] != 3:
        return
    target = out_dir / f"{rank:04d}_{row['split']}_{Path(row['trajectory']).stem}_{row['frame']:06d}.png"
    cv2.imwrite(str(target), cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR))


def generate_visualizations(output_dir, frame_rows, traj_rows, summary):
    """Generate comprehensive visualization report as PDF."""
    pdf_path = output_dir / "evaluation_report.pdf"

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    with PdfPages(pdf_path) as pdf:
        # Page 1: Error Distribution
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Error Distribution Analysis', fontsize=16, fontweight='bold')

        # MSE histogram
        mse_vals = [r['continuous_mse'] for r in frame_rows]
        axes[0, 0].hist(mse_vals, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0, 0].axvline(np.mean(mse_vals), color='red', linestyle='--',
                          label=f'Mean: {np.mean(mse_vals):.4f}')
        axes[0, 0].axvline(np.median(mse_vals), color='green', linestyle='--',
                          label=f'Median: {np.median(mse_vals):.4f}')
        axes[0, 0].set_xlabel('Continuous MSE')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('MSE Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # MAE histogram
        mae_vals = [r['continuous_mae'] for r in frame_rows]
        axes[0, 1].hist(mae_vals, bins=50, edgecolor='black', alpha=0.7, color='lightcoral')
        axes[0, 1].axvline(np.mean(mae_vals), color='red', linestyle='--',
                          label=f'Mean: {np.mean(mae_vals):.4f}')
        axes[0, 1].axvline(np.median(mae_vals), color='green', linestyle='--',
                          label=f'Median: {np.median(mae_vals):.4f}')
        axes[0, 1].set_xlabel('Continuous MAE')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('MAE Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Box plots
        error_data = [mse_vals, mae_vals,
                     [r['smoothness_mse'] for r in frame_rows],
                     [r['magnitude_error'] for r in frame_rows]]
        axes[1, 0].boxplot(error_data, tick_labels=['MSE', 'MAE', 'Smoothness', 'Magnitude'])
        axes[1, 0].set_ylabel('Error Value')
        axes[1, 0].set_title('Error Metrics Box Plot')
        axes[1, 0].grid(True, alpha=0.3)

        # Cosine similarity
        cos_vals = [r['cosine_similarity'] for r in frame_rows]
        axes[1, 1].hist(cos_vals, bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
        axes[1, 1].axvline(np.mean(cos_vals), color='red', linestyle='--',
                          label=f'Mean: {np.mean(cos_vals):.4f}')
        axes[1, 1].set_xlabel('Cosine Similarity')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Action Cosine Similarity')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # Page 2: Time Series Analysis
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Temporal Error Analysis', fontsize=16, fontweight='bold')

        # Group by trajectory for visualization
        traj_groups = {}
        for row in frame_rows:
            traj_name = Path(row['trajectory']).stem
            if traj_name not in traj_groups:
                traj_groups[traj_name] = []
            traj_groups[traj_name].append(row)

        # Plot first 5 trajectories' MSE over time
        for i, (traj_name, frames) in enumerate(list(traj_groups.items())[:5]):
            frames_sorted = sorted(frames, key=lambda x: x['frame'])
            frame_ids = [f['frame'] for f in frames_sorted]
            mse_over_time = [f['continuous_mse'] for f in frames_sorted]
            axes[0, 0].plot(frame_ids, mse_over_time, marker='o', markersize=2,
                           label=traj_name[:20], alpha=0.7)
        axes[0, 0].set_xlabel('Frame Index')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].set_title('MSE Over Time (First 5 Trajectories)')
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)

        # Average MSE across all trajectories
        max_frames = max(len(frames) for frames in traj_groups.values())
        avg_mse_by_frame = []
        for frame_idx in range(max_frames):
            frame_mse = [frames[frame_idx]['continuous_mse']
                        for frames in traj_groups.values()
                        if frame_idx < len(frames)]
            if frame_mse:
                avg_mse_by_frame.append(np.mean(frame_mse))

        axes[0, 1].plot(range(len(avg_mse_by_frame)), avg_mse_by_frame,
                       marker='o', markersize=3, color='darkblue', linewidth=2)
        axes[0, 1].fill_between(range(len(avg_mse_by_frame)),
                               [np.percentile([frames[i]['continuous_mse']
                                              for frames in traj_groups.values() if i < len(frames)], 25)
                                for i in range(len(avg_mse_by_frame))],
                               [np.percentile([frames[i]['continuous_mse']
                                              for frames in traj_groups.values() if i < len(frames)], 75)
                                for i in range(len(avg_mse_by_frame))],
                               alpha=0.3)
        axes[0, 1].set_xlabel('Frame Index')
        axes[0, 1].set_ylabel('Average MSE')
        axes[0, 1].set_title('Average MSE Across All Trajectories (25-75 percentile)')
        axes[0, 1].grid(True, alpha=0.3)

        # Smoothness over time
        for i, (traj_name, frames) in enumerate(list(traj_groups.items())[:5]):
            frames_sorted = sorted(frames, key=lambda x: x['frame'])
            frame_ids = [f['frame'] for f in frames_sorted]
            smooth = [f['smoothness_mse'] for f in frames_sorted]
            axes[1, 0].plot(frame_ids, smooth, marker='o', markersize=2,
                           label=traj_name[:20], alpha=0.7)
        axes[1, 0].set_xlabel('Frame Index')
        axes[1, 0].set_ylabel('Smoothness MSE')
        axes[1, 0].set_title('Action Smoothness Over Time')
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)

        # Cosine similarity over time
        avg_cos_by_frame = []
        for frame_idx in range(max_frames):
            frame_cos = [frames[frame_idx]['cosine_similarity']
                        for frames in traj_groups.values()
                        if frame_idx < len(frames)]
            if frame_cos:
                avg_cos_by_frame.append(np.mean(frame_cos))

        axes[1, 1].plot(range(len(avg_cos_by_frame)), avg_cos_by_frame,
                       marker='o', markersize=3, color='green', linewidth=2)
        axes[1, 1].set_xlabel('Frame Index')
        axes[1, 1].set_ylabel('Average Cosine Similarity')
        axes[1, 1].set_title('Action Direction Similarity Over Time')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # Page 3: Trajectory Comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Per-Trajectory Performance', fontsize=16, fontweight='bold')

        # Sort trajectories by MSE
        traj_sorted = sorted(traj_rows, key=lambda x: x['continuous_mse'])
        top_n = min(15, len(traj_sorted))

        # Best and worst trajectories MSE
        traj_names_short = [Path(t['trajectory']).stem[:25] for t in traj_sorted[:top_n]]
        traj_mse = [t['continuous_mse'] for t in traj_sorted[:top_n]]
        colors = ['green' if i < 5 else 'orange' if i < 10 else 'red'
                 for i in range(top_n)]

        axes[0, 0].barh(range(top_n), traj_mse, color=colors, alpha=0.7)
        axes[0, 0].set_yticks(range(top_n))
        axes[0, 0].set_yticklabels(traj_names_short, fontsize=8)
        axes[0, 0].set_xlabel('Average MSE')
        axes[0, 0].set_title(f'Top {top_n} Trajectories by MSE (Best to Worst)')
        axes[0, 0].invert_yaxis()
        axes[0, 0].grid(True, alpha=0.3, axis='x')

        # Gripper accuracy by trajectory
        traj_gripper = [t['gripper_joint_accuracy'] for t in traj_sorted[:top_n]]
        axes[0, 1].barh(range(top_n), traj_gripper, color='skyblue', alpha=0.7)
        axes[0, 1].set_yticks(range(top_n))
        axes[0, 1].set_yticklabels(traj_names_short, fontsize=8)
        axes[0, 1].set_xlabel('Gripper Joint Accuracy')
        axes[0, 1].set_title('Gripper Accuracy (Same Trajectories)')
        axes[0, 1].set_xlim([0, 1])
        axes[0, 1].invert_yaxis()
        axes[0, 1].grid(True, alpha=0.3, axis='x')

        # Scatter: MSE vs Gripper Accuracy
        all_traj_mse = [t['continuous_mse'] for t in traj_rows]
        all_traj_gripper = [t['gripper_joint_accuracy'] for t in traj_rows]
        axes[1, 0].scatter(all_traj_mse, all_traj_gripper, alpha=0.6, s=50)
        axes[1, 0].set_xlabel('Trajectory MSE')
        axes[1, 0].set_ylabel('Gripper Accuracy')
        axes[1, 0].set_title('MSE vs Gripper Accuracy')
        axes[1, 0].grid(True, alpha=0.3)

        # Trajectory length vs MSE
        traj_lengths = [t['frames'] for t in traj_rows]
        axes[1, 1].scatter(traj_lengths, all_traj_mse, alpha=0.6, s=50, color='purple')
        axes[1, 1].set_xlabel('Trajectory Length (frames)')
        axes[1, 1].set_ylabel('Average MSE')
        axes[1, 1].set_title('Trajectory Length vs Error')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # Page 4: Gripper Performance
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Gripper Classification Performance', fontsize=16, fontweight='bold')

        # Gripper accuracy bars
        left_acc = np.mean([r['gripper_left_correct'] for r in frame_rows])
        right_acc = np.mean([r['gripper_right_correct'] for r in frame_rows])
        joint_acc = np.mean([r['gripper_joint_correct'] for r in frame_rows])

        gripper_types = ['Left', 'Right', 'Joint']
        accuracies = [left_acc, right_acc, joint_acc]
        colors_grip = ['lightblue', 'lightcoral', 'lightgreen']

        bars = axes[0, 0].bar(gripper_types, accuracies, color=colors_grip,
                             edgecolor='black', alpha=0.7)
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Gripper Classification Accuracy')
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

        # Gripper accuracy over time
        avg_left = []
        avg_right = []
        avg_joint = []
        for frame_idx in range(max_frames):
            left_at_frame = [frames[frame_idx]['gripper_left_correct']
                            for frames in traj_groups.values()
                            if frame_idx < len(frames)]
            right_at_frame = [frames[frame_idx]['gripper_right_correct']
                             for frames in traj_groups.values()
                             if frame_idx < len(frames)]
            joint_at_frame = [frames[frame_idx]['gripper_joint_correct']
                             for frames in traj_groups.values()
                             if frame_idx < len(frames)]
            if left_at_frame:
                avg_left.append(np.mean(left_at_frame))
                avg_right.append(np.mean(right_at_frame))
                avg_joint.append(np.mean(joint_at_frame))

        axes[0, 1].plot(range(len(avg_left)), avg_left, marker='o',
                       label='Left', linewidth=2, markersize=3)
        axes[0, 1].plot(range(len(avg_right)), avg_right, marker='s',
                       label='Right', linewidth=2, markersize=3)
        axes[0, 1].plot(range(len(avg_joint)), avg_joint, marker='^',
                       label='Joint', linewidth=2, markersize=3)
        axes[0, 1].set_xlabel('Frame Index')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Gripper Accuracy Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1])

        # Confusion matrix style - gripper correct/incorrect vs error level
        high_error_thresh = np.percentile(mse_vals, 75)
        high_err_correct = sum(1 for r in frame_rows
                              if r['continuous_mse'] > high_error_thresh
                              and r['gripper_joint_correct'])
        high_err_incorrect = sum(1 for r in frame_rows
                                if r['continuous_mse'] > high_error_thresh
                                and not r['gripper_joint_correct'])
        low_err_correct = sum(1 for r in frame_rows
                             if r['continuous_mse'] <= high_error_thresh
                             and r['gripper_joint_correct'])
        low_err_incorrect = sum(1 for r in frame_rows
                               if r['continuous_mse'] <= high_error_thresh
                               and not r['gripper_joint_correct'])

        confusion_data = np.array([[low_err_correct, low_err_incorrect],
                                   [high_err_correct, high_err_incorrect]])
        sns.heatmap(confusion_data, annot=True, fmt='d', cmap='YlOrRd',
                   xticklabels=['Correct', 'Incorrect'],
                   yticklabels=['Low MSE', 'High MSE'],
                   ax=axes[1, 0], cbar_kws={'label': 'Count'})
        axes[1, 0].set_title('Gripper Accuracy vs Continuous Error Level')
        axes[1, 0].set_ylabel('MSE Level')
        axes[1, 0].set_xlabel('Gripper Classification')

        # Error distribution by gripper correctness
        correct_errors = [r['continuous_mse'] for r in frame_rows if r['gripper_joint_correct']]
        incorrect_errors = [r['continuous_mse'] for r in frame_rows if not r['gripper_joint_correct']]

        axes[1, 1].boxplot([correct_errors, incorrect_errors],
                          tick_labels=['Gripper Correct', 'Gripper Incorrect'])
        axes[1, 1].set_ylabel('Continuous MSE')
        axes[1, 1].set_title('MSE Distribution by Gripper Correctness')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # Page 5: Correlation Analysis
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Metric Correlation Analysis', fontsize=16, fontweight='bold')

        # Correlation heatmap
        metric_data = {
            'MSE': [r['continuous_mse'] for r in frame_rows],
            'MAE': [r['continuous_mae'] for r in frame_rows],
            'Cosine': [r['cosine_similarity'] for r in frame_rows],
            'Magnitude': [r['magnitude_error'] for r in frame_rows],
            'Smoothness': [r['smoothness_mse'] for r in frame_rows],
            'Gripper': [float(r['gripper_joint_correct']) for r in frame_rows]
        }

        # Compute correlation matrix
        import pandas as pd
        df_metrics = pd.DataFrame(metric_data)
        corr_matrix = df_metrics.corr()

        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                   center=0, vmin=-1, vmax=1, ax=axes[0, 0],
                   cbar_kws={'label': 'Correlation'})
        axes[0, 0].set_title('Metric Correlation Matrix')

        # Scatter plots for key correlations
        axes[0, 1].scatter(metric_data['MSE'], metric_data['MAE'],
                          alpha=0.5, s=10, color='blue')
        axes[0, 1].set_xlabel('MSE')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_title('MSE vs MAE')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].scatter(metric_data['MSE'], metric_data['Cosine'],
                          alpha=0.5, s=10, color='green')
        axes[1, 0].set_xlabel('MSE')
        axes[1, 0].set_ylabel('Cosine Similarity')
        axes[1, 0].set_title('MSE vs Cosine Similarity')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].scatter(metric_data['Smoothness'], metric_data['MSE'],
                          alpha=0.5, s=10, color='purple')
        axes[1, 1].set_xlabel('Smoothness MSE')
        axes[1, 1].set_ylabel('Continuous MSE')
        axes[1, 1].set_title('Smoothness vs Prediction Error')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # Page 6: Summary Statistics
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle('Evaluation Summary', fontsize=16, fontweight='bold')

        # Create text summary
        ax = fig.add_subplot(111)
        ax.axis('off')

        summary_text = f"""
OFFLINE BC EVALUATION SUMMARY
{'='*60}

Experiment: {summary['exp_name']}
Checkpoint: {summary['checkpoint_path']}
Total Frames: {summary['num_frames']:,}
Total Trajectories: {len(traj_rows)}

CONTINUOUS ACTION METRICS
{'-'*60}
MSE:
    Mean:    {summary['continuous_mse']['mean']:.6f}
    Median:  {summary['continuous_mse']['median']:.6f}
    Std:     {summary['continuous_mse']['std']:.6f}
    Min:     {summary['continuous_mse']['min']:.6f}
    Max:     {summary['continuous_mse']['max']:.6f}

MAE:
    Mean:    {summary['continuous_mae']['mean']:.6f}
    Median:  {summary['continuous_mae']['median']:.6f}
    Std:     {summary['continuous_mae']['std']:.6f}

Cosine Similarity:
    Mean:    {summary['cosine_similarity']['mean']:.6f}
    Median:  {summary['cosine_similarity']['median']:.6f}

Smoothness MSE:
    Mean:    {summary['smoothness_mse']['mean']:.6f}
    Median:  {summary['smoothness_mse']['median']:.6f}

GRIPPER CLASSIFICATION
{'-'*60}
Joint Accuracy:  {summary['gripper_joint_accuracy']['mean']:.4f} ({summary['gripper_joint_accuracy']['mean']*100:.2f}%)

TRAJECTORY STATISTICS
{'-'*60}
Average Length: {np.mean([t['frames'] for t in traj_rows]):.1f} frames
Shortest:       {min(t['frames'] for t in traj_rows)} frames
Longest:        {max(t['frames'] for t in traj_rows)} frames

Best Trajectory MSE:  {min(t['continuous_mse'] for t in traj_rows):.6f}
Worst Trajectory MSE: {max(t['continuous_mse'] for t in traj_rows):.6f}
"""

        ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    print_green(f"[Visualization] Generated comprehensive report: {pdf_path}")
    return pdf_path


def _state_xyz(observations):
    """Extract left/right tcp_pose xyz (relative-to-reset, meters) from a flattened obs state."""
    state = observations.get("state") if isinstance(observations, dict) else None
    if state is None:
        return None, None
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] < STATE_RIGHT_POSE.stop:
        return None, None
    return (
        np.array(state[STATE_LEFT_POSE][:3], dtype=np.float32),
        np.array(state[STATE_RIGHT_POSE][:3], dtype=np.float32),
    )


def _stack_xyz(seq):
    """Stack a list of (3,) arrays into (N,3); rows with None become NaN."""
    if not seq:
        return np.empty((0, 3), dtype=np.float32)
    out = np.full((len(seq), 3), np.nan, dtype=np.float32)
    for i, v in enumerate(seq):
        if v is not None:
            out[i] = np.asarray(v, dtype=np.float32).reshape(-1)[:3]
    return out


def _one_step_action_xyz(actions, tcp_path, action_scale: float = 0.01):
    """Map each action to a one-step TCP endpoint from the matching real TCP pose."""
    actions = np.asarray(actions, dtype=np.float32)
    tcp_path = np.asarray(tcp_path, dtype=np.float32)
    if actions.shape[0] < 1 or tcp_path.shape[0] < 1:
        return np.empty((0, 3), dtype=np.float32)
    count = min(actions.shape[0], tcp_path.shape[0])
    eval_path = np.full((count, 3), np.nan, dtype=np.float32)
    eval_path[0] = tcp_path[0]
    if count > 1:
        eval_path[1:] = tcp_path[: count - 1] + actions[: count - 1] * action_scale
    return eval_path


def _step_error(actions, tcp_path, action_scale: float = 0.01):
    """Return one-step TCP endpoint MSE and last-step error."""
    actions = np.asarray(actions, dtype=np.float32)
    tcp_path = np.asarray(tcp_path, dtype=np.float32)
    if actions.shape[0] < 2 or tcp_path.shape[0] < 2:
        return None, None
    eval_path = _one_step_action_xyz(actions, tcp_path, action_scale=action_scale)
    count = min(eval_path.shape[0], tcp_path.shape[0])
    eval_next = eval_path[1:count]
    real_next = tcp_path[1:count]
    finite = np.all(np.isfinite(eval_next), axis=1) & np.all(np.isfinite(real_next), axis=1)
    if not np.any(finite):
        return None, None
    diff = eval_next[finite] - real_next[finite]
    mse = float(np.mean(diff ** 2))
    last_error = float(np.linalg.norm(diff[-1]))
    return mse, last_error


def _step_metric_arrays(actions, tcp_path, action_scale: float = 0.01, moving_threshold=None):
    """Return per-action one-step displacement diagnostics."""
    actions = np.asarray(actions, dtype=np.float32)
    tcp_path = np.asarray(tcp_path, dtype=np.float32)
    count = min(actions.shape[0], tcp_path.shape[0])
    empty = {
        "real_step_norm": np.empty((0,), dtype=np.float32),
        "eval_step_norm": np.empty((0,), dtype=np.float32),
        "step_error_norm": np.empty((0,), dtype=np.float32),
        "relative_step_error": np.empty((0,), dtype=np.float32),
        "direction_cosine": np.empty((0,), dtype=np.float32),
        "is_moving": np.empty((0,), dtype=bool),
    }
    if count < 2:
        return empty

    threshold = FLAGS.moving_step_threshold if moving_threshold is None else moving_threshold
    eval_step = actions[: count - 1] * action_scale
    real_step = tcp_path[1:count] - tcp_path[: count - 1]
    step_error = eval_step - real_step
    finite = (
        np.all(np.isfinite(eval_step), axis=1)
        & np.all(np.isfinite(real_step), axis=1)
        & np.all(np.isfinite(step_error), axis=1)
    )

    real_norm = np.full((count - 1,), np.nan, dtype=np.float32)
    eval_norm = np.full((count - 1,), np.nan, dtype=np.float32)
    error_norm = np.full((count - 1,), np.nan, dtype=np.float32)
    relative_error = np.full((count - 1,), np.nan, dtype=np.float32)
    direction_cosine = np.full((count - 1,), np.nan, dtype=np.float32)

    real_norm[finite] = np.linalg.norm(real_step[finite], axis=1)
    eval_norm[finite] = np.linalg.norm(eval_step[finite], axis=1)
    error_norm[finite] = np.linalg.norm(step_error[finite], axis=1)
    relative_error[finite] = error_norm[finite] / np.maximum(real_norm[finite], 1e-6)

    denom = real_norm * eval_norm
    valid_direction = finite & (denom > 1e-12)
    direction_cosine[valid_direction] = (
        np.sum(eval_step[valid_direction] * real_step[valid_direction], axis=1)
        / denom[valid_direction]
    )

    return {
        "real_step_norm": real_norm,
        "eval_step_norm": eval_norm,
        "step_error_norm": error_norm,
        "relative_step_error": relative_error,
        "direction_cosine": direction_cosine,
        "is_moving": finite & (real_norm > threshold),
    }


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _add_frame_step_metrics(rows, side, metrics):
    for idx, row in enumerate(rows):
        if idx < metrics["real_step_norm"].shape[0]:
            row[f"{side}_real_step_norm"] = _finite_or_none(metrics["real_step_norm"][idx])
            row[f"{side}_eval_step_norm"] = _finite_or_none(metrics["eval_step_norm"][idx])
            row[f"{side}_step_error_norm"] = _finite_or_none(metrics["step_error_norm"][idx])
            row[f"{side}_relative_step_error"] = _finite_or_none(
                metrics["relative_step_error"][idx]
            )
            row[f"{side}_direction_cosine"] = _finite_or_none(metrics["direction_cosine"][idx])
            row[f"{side}_is_moving"] = int(metrics["is_moving"][idx])
        else:
            row[f"{side}_real_step_norm"] = None
            row[f"{side}_eval_step_norm"] = None
            row[f"{side}_step_error_norm"] = None
            row[f"{side}_relative_step_error"] = None
            row[f"{side}_direction_cosine"] = None
            row[f"{side}_is_moving"] = 0


def _finite_values(values, mask=None):
    arr = np.asarray([np.nan if v is None else v for v in values], dtype=np.float64)
    finite = np.isfinite(arr)
    if mask is not None:
        finite &= np.asarray([False if v is None else bool(v) for v in mask], dtype=bool)
    return arr[finite]


def _metric_summary(values, mask=None):
    arr = _finite_values(values, mask)
    if arr.size == 0:
        return {"mean": None, "median": None, "p90": None, "max": None}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def _add_prefixed_stats(row, prefix, metrics):
    moving = metrics["is_moving"]
    for name in ("real_step_norm", "eval_step_norm", "step_error_norm"):
        stats = _metric_summary(metrics[name])
        row[f"{prefix}_{name}_mean"] = stats["mean"]
        row[f"{prefix}_{name}_median"] = stats["median"]
        row[f"{prefix}_{name}_p90"] = stats["p90"]
        row[f"{prefix}_{name}_max"] = stats["max"]

    rel_stats = _metric_summary(metrics["relative_step_error"], moving)
    cos_stats = _metric_summary(metrics["direction_cosine"], moving)
    row[f"{prefix}_relative_step_error_moving_mean"] = rel_stats["mean"]
    row[f"{prefix}_relative_step_error_moving_median"] = rel_stats["median"]
    row[f"{prefix}_relative_step_error_moving_p90"] = rel_stats["p90"]
    row[f"{prefix}_direction_cosine_moving_mean"] = cos_stats["mean"]
    row[f"{prefix}_direction_cosine_moving_median"] = cos_stats["median"]
    row[f"{prefix}_direction_cosine_moving_p90"] = cos_stats["p90"]
    row[f"{prefix}_moving_ratio"] = float(np.mean(moving)) if moving.size else None


def _mean_available(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def _xyz_mse(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    count = min(a.shape[0], b.shape[0])
    if count < 1:
        return None
    a = a[:count]
    b = b[:count]
    finite = np.all(np.isfinite(a), axis=1) & np.all(np.isfinite(b), axis=1)
    if not np.any(finite):
        return None
    return float(np.mean((a[finite] - b[finite]) ** 2))


def _plot_step_action_trajectory(ax, pred_actions, tcp_path, title,
                                 action_scale: float = 0.01):
    """Plot real TCP trajectory against one-step eval TCP endpoints.

    Real trajectory is the ground-truth tcp_pose from observations. Eval trajectory
    uses each real TCP pose as the origin for the matching predicted action:
        eval_pos[t + 1] = real_pos[t] + action[t] * action_scale

    Args:
        pred_actions: (N, 3) predicted action xyz
        tcp_path: (N, 3) ground-truth tcp_pose xyz per frame
        title: subplot title
        action_scale: env ACTION_SCALE[0], default 0.01 m per unit
    """
    if pred_actions.shape[0] < 2 or tcp_path.shape[0] < 2:
        ax.set_title(f"{title}\n(insufficient data)", fontsize=10)
        ax.axis('off')
        return

    real_traj = tcp_path
    eval_traj = _one_step_action_xyz(pred_actions, tcp_path, action_scale)
    pred_mse, _ = _step_error(pred_actions, tcp_path, action_scale)
    pred_mse_text = f"{pred_mse:.4f}" if pred_mse is not None else "nan"

    ax.plot(real_traj[:, 0], real_traj[:, 1], real_traj[:, 2],
            color="green", linewidth=2.5, alpha=0.9, label="Real EE path")
    ax.scatter(*real_traj[0], color="green", marker="o", s=80, zorder=10)
    ax.scatter(*real_traj[-1], color="green", marker="*", s=120, zorder=10)

    ax.plot(eval_traj[:, 0], eval_traj[:, 1], eval_traj[:, 2],
            color="red", linewidth=2.0, alpha=0.8, label="Eval EE path")
    ax.scatter(*eval_traj[0], color="red", marker="o", s=60, zorder=10)
    ax.scatter(*eval_traj[-1], color="red", marker="*", s=100, zorder=10)

    ax.set_xlabel("x (m)", fontsize=9)
    ax.set_ylabel("y (m)", fontsize=9)
    ax.set_zlabel("z (m)", fontsize=9)
    ax.set_title(
        f"{title}\neval step MSE={pred_mse_text}",
        fontsize=10,
        fontweight='bold',
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)


def plot_step_action_trajectories(output_dir, traj_xyz):
    """Render one-step eval trajectory comparison as standalone PNGs (not in PDF).

    Eval trajectory maps each predicted action to an endpoint from the matching real
    TCP pose. Real trajectory is the ground-truth tcp_pose path from observations.
    """
    if not traj_xyz:
        return
    png_dir = output_dir / "trajectory_step"
    png_dir.mkdir(exist_ok=True)

    # Worst one-step trajectory error first; limit count if requested.
    ordered = sorted(
        traj_xyz,
        key=lambda s: s.get("pred_step_mse") or s.get("continuous_mse", 0.0),
        reverse=True,
    )
    count = FLAGS.traj_count
    if count and count > 0:
        ordered = ordered[:count]

    for xyz_seq in ordered:
        name = Path(xyz_seq["trajectory"]).stem
        mse = xyz_seq.get("continuous_mse", float("nan"))

        pred_l = _stack_xyz(xyz_seq["pred_left"])
        pred_r = _stack_xyz(xyz_seq["pred_right"])
        path_l = _stack_xyz(xyz_seq["path_left"])
        path_r = _stack_xyz(xyz_seq["path_right"])

        fig = plt.figure(figsize=(14, 6))
        fig.suptitle(f"Eval vs Real End-Effector Trajectory — {name}  (MSE={mse:.4f})",
                     fontsize=14, fontweight="bold")

        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        _plot_step_action_trajectory(ax1, pred_l, path_l, "Left Arm")

        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        _plot_step_action_trajectory(ax2, pred_r, path_r, "Right Arm")

        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(png_dir / f"{name}_action_step.png", dpi=120)
        plt.close(fig)

    print_green(
        f"[Visualization] One-step action trajectory plots: {len(ordered)} figures -> {png_dir}"
    )


def _plot_action_diagnostics_row(axes, pred_actions, demo_actions, title):
    count = min(pred_actions.shape[0], demo_actions.shape[0])
    frames = np.arange(count)
    component_names = ("X", "Y", "Z")

    pred = pred_actions[:count]
    demo = demo_actions[:count]

    for dim, ax in enumerate(axes):
        ax.plot(frames, demo[:, dim], color="black", linewidth=1.5,
                alpha=0.9, label="Demo action")
        ax.plot(frames, pred[:, dim], color="red", linewidth=1.2,
                alpha=0.85, label="Eval action")
        ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.9)
        ax.set_title(
            f"{title}: {component_names[dim]} action",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xlabel("Frame")
        ax.set_ylabel("action value")
        if dim == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)


def plot_action_diagnostics(output_dir, traj_xyz):
    """Render per-frame predicted vs demo xyz actions as PNGs."""
    if not traj_xyz:
        return
    png_dir = output_dir / "trajectory_action_diagnostics"
    png_dir.mkdir(exist_ok=True)

    ordered = sorted(
        traj_xyz,
        key=lambda s: s.get("continuous_mse", 0.0),
        reverse=True,
    )
    count = FLAGS.traj_count
    if count and count > 0:
        ordered = ordered[:count]

    for xyz_seq in ordered:
        name = Path(xyz_seq["trajectory"]).stem
        pred_l = _stack_xyz(xyz_seq["pred_left"])
        pred_r = _stack_xyz(xyz_seq["pred_right"])
        demo_l = _stack_xyz(xyz_seq["demo_left"])
        demo_r = _stack_xyz(xyz_seq["demo_right"])

        fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False)
        fig.suptitle(
            f"Per-frame Action XYZ — {name}",
            fontsize=14,
            fontweight="bold",
        )
        _plot_action_diagnostics_row(axes[0], pred_l, demo_l, "Left Arm")
        _plot_action_diagnostics_row(axes[1], pred_r, demo_r, "Right Arm")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(png_dir / f"{name}_action_diagnostics.png", dpi=120)
        plt.close(fig)

    print_green(
        f"[Visualization] Action diagnostics plots: {len(ordered)} figures -> {png_dir}"
    )


def _plot_step_diagnostics_row(axes, metrics, title):
    frames = np.arange(metrics["real_step_norm"].shape[0])
    moving = metrics["is_moving"]

    axes[0].plot(frames, metrics["real_step_norm"] * 1000.0, color="black",
                 linewidth=1.5, label="Real step")
    axes[0].plot(frames, metrics["step_error_norm"] * 1000.0, color="red",
                 linewidth=1.2, alpha=0.85, label="Eval error")
    axes[0].set_title(f"{title}: step size vs error", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("mm")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(frames[moving], metrics["relative_step_error"][moving],
                    s=8, color="red", alpha=0.7)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_title(f"{title}: relative error (moving)", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("error / real step")
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(frames[moving], metrics["direction_cosine"][moving],
                    s=8, color="green", alpha=0.7)
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[2].axhline(0.0, color="gray", linestyle=":", linewidth=1.0)
    axes[2].set_title(f"{title}: direction cosine (moving)", fontsize=10, fontweight="bold")
    axes[2].set_xlabel("Frame")
    axes[2].set_ylabel("cosine")
    axes[2].set_ylim([-1.05, 1.05])
    axes[2].grid(True, alpha=0.3)


def plot_step_diagnostics(output_dir, traj_xyz):
    """Render per-step absolute/relative error diagnostics as standalone PNGs."""
    if not traj_xyz:
        return
    png_dir = output_dir / "trajectory_step_diagnostics"
    png_dir.mkdir(exist_ok=True)

    ordered = sorted(
        traj_xyz,
        key=lambda s: s.get("pred_step_mse") or s.get("continuous_mse", 0.0),
        reverse=True,
    )
    count = FLAGS.traj_count
    if count and count > 0:
        ordered = ordered[:count]

    for xyz_seq in ordered:
        name = Path(xyz_seq["trajectory"]).stem
        pred_l = _stack_xyz(xyz_seq["pred_left"])
        pred_r = _stack_xyz(xyz_seq["pred_right"])
        path_l = _stack_xyz(xyz_seq["path_left"])
        path_r = _stack_xyz(xyz_seq["path_right"])
        left_metrics = _step_metric_arrays(pred_l, path_l)
        right_metrics = _step_metric_arrays(pred_r, path_r)

        fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False)
        fig.suptitle(
            f"One-Step TCP Diagnostics — {name}  "
            f"(moving > {FLAGS.moving_step_threshold * 1000:.1f} mm)",
            fontsize=14,
            fontweight="bold",
        )
        _plot_step_diagnostics_row(axes[0], left_metrics, "Left Arm")
        _plot_step_diagnostics_row(axes[1], right_metrics, "Right Arm")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(png_dir / f"{name}_step_diagnostics.png", dpi=120)
        plt.close(fig)

    print_green(
        f"[Visualization] One-step diagnostics plots: {len(ordered)} figures -> {png_dir}"
    )


def evaluate_split(agent, split_name, dir_path, rng):
    frame_rows = []
    traj_rows = []
    high_error = []
    traj_xyz = []  # per-trajectory action/path xyz sequences for 3D plotting
    trajectories = load_trajectories(dir_path, max_trajs=FLAGS.max_trajs)

    for path, traj in trajectories:
        traj_mse = []
        traj_mae = []
        traj_joint_acc = []
        traj_frame_rows = []
        prev_pred = None
        prev_demo = None
        xyz_seq = {
            "split": split_name,
            "trajectory": str(path),
            "pred_left": [], "pred_right": [],  # predicted action xyz for one-step TCP endpoints
            "demo_left": [], "demo_right": [],   # demo action xyz for action-space comparison
            "path_left": [], "path_right": [],   # tcp_pose xyz (ground-truth EE path)
        }
        for frame_idx, transition in enumerate(traj):
            prepared = prepare_transition(
                transition,
                skip_zero_action=False,
            )
            if prepared is None:
                continue
            rng, key = jax.random.split(rng)
            pred = agent.sample_actions(
                observations=jax.device_put(prepared["observations"]),
                seed=key,
                argmax=FLAGS.argmax,
            )
            pred_action = np.asarray(jax.device_get(pred))
            demo_action = np.asarray(prepared["actions"], dtype=np.float32)
            pred_cont = continuous_action(pred_action)
            demo_cont = continuous_action(demo_action)
            diff = pred_cont - demo_cont
            mse = float(np.mean(diff ** 2))
            mae = float(np.mean(np.abs(diff)))
            denom = float(np.linalg.norm(pred_cont) * np.linalg.norm(demo_cont))
            cosine = float(np.dot(pred_cont, demo_cont) / denom) if denom > 1e-8 else 0.0
            pred_left, pred_right, pred_joint = gripper_classes(pred_action)
            demo_left, demo_right, demo_joint = gripper_classes(demo_action)
            pred_delta = np.zeros_like(pred_cont) if prev_pred is None else pred_cont - prev_pred
            demo_delta = np.zeros_like(demo_cont) if prev_demo is None else demo_cont - prev_demo
            smooth_mse = float(np.mean((pred_delta - demo_delta) ** 2))
            row = {
                "split": split_name,
                "trajectory": str(path),
                "frame": frame_idx,
                "continuous_mse": mse,
                "continuous_mae": mae,
                "cosine_similarity": cosine,
                "magnitude_error": float(np.linalg.norm(pred_cont) - np.linalg.norm(demo_cont)),
                "smoothness_mse": smooth_mse,
                "gripper_left_correct": int(pred_left == demo_left),
                "gripper_right_correct": int(pred_right == demo_right),
                "gripper_joint_correct": int(pred_joint == demo_joint),
            }
            frame_rows.append(row)
            traj_frame_rows.append(row)
            high_error.append((mse, row, prepared["observations"]))
            # Record action xyz for one-step TCP endpoint mapping and tcp_pose paths.
            pred_vec = np.asarray(pred_action, dtype=np.float32).reshape(-1)
            demo_vec = np.asarray(demo_action, dtype=np.float32).reshape(-1)
            xyz_seq["pred_left"].append(pred_vec[ACT_LEFT_XYZ])
            xyz_seq["pred_right"].append(pred_vec[ACT_RIGHT_XYZ])
            xyz_seq["demo_left"].append(demo_vec[ACT_LEFT_XYZ])
            xyz_seq["demo_right"].append(demo_vec[ACT_RIGHT_XYZ])
            path_left, path_right = _state_xyz(prepared["observations"])
            xyz_seq["path_left"].append(path_left)
            xyz_seq["path_right"].append(path_right)
            traj_mse.append(mse)
            traj_mae.append(mae)
            traj_joint_acc.append(float(pred_joint == demo_joint))
            prev_pred = pred_cont
            prev_demo = demo_cont
        if traj_mse:
            pred_l = _stack_xyz(xyz_seq["pred_left"])
            pred_r = _stack_xyz(xyz_seq["pred_right"])
            path_l = _stack_xyz(xyz_seq["path_left"])
            path_r = _stack_xyz(xyz_seq["path_right"])
            pred_l_mse, pred_l_last = _step_error(pred_l, path_l)
            pred_r_mse, pred_r_last = _step_error(pred_r, path_r)
            left_metrics = _step_metric_arrays(pred_l, path_l)
            right_metrics = _step_metric_arrays(pred_r, path_r)
            _add_frame_step_metrics(traj_frame_rows, "left", left_metrics)
            _add_frame_step_metrics(traj_frame_rows, "right", right_metrics)

            row = {
                "split": split_name,
                "trajectory": str(path),
                "frames": len(traj_mse),
                "continuous_mse": float(np.mean(traj_mse)),
                "continuous_mae": float(np.mean(traj_mae)),
                "gripper_joint_accuracy": float(np.mean(traj_joint_acc)),
                "pred_step_mse": _mean_available([pred_l_mse, pred_r_mse]),
                "pred_step_last_error": _mean_available([pred_l_last, pred_r_last]),
                "pred_left_step_mse": pred_l_mse,
                "pred_right_step_mse": pred_r_mse,
            }
            _add_prefixed_stats(row, "left", left_metrics)
            _add_prefixed_stats(row, "right", right_metrics)
            traj_rows.append(row)
            xyz_seq["continuous_mse"] = float(np.mean(traj_mse))
            xyz_seq["pred_step_mse"] = row["pred_step_mse"]
            traj_xyz.append(xyz_seq)
    return frame_rows, traj_rows, high_error, traj_xyz, rng


def main(_):
    if FLAGS.exp_name not in CONFIG_MAPPING:
        raise ValueError(f"Experiment {FLAGS.exp_name!r} not found in CONFIG_MAPPING.")
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    default_checkpoint_path = os.fspath(task_bc_checkpoint_dir(FLAGS.exp_name))
    checkpoint_path = os.path.abspath(
        FLAGS.bc_checkpoint_path or default_checkpoint_path
    )
    default_output_dir = task_bc_eval_dir(FLAGS.exp_name) / "offline"
    output_dir = Path(FLAGS.output_dir or default_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = config.get_environment(fake_env=True, classifier=False)
    agent = restore_agent(create_agent(config, env), checkpoint_path)
    rng = jax.random.PRNGKey(FLAGS.seed)

    # BC评估只使用success数据（BC训练也只用success数据）
    success_dir = os.path.abspath(FLAGS.success_dir or task_success_dir(FLAGS.exp_name))
    all_frame_rows, all_traj_rows, all_high_error, all_traj_xyz, rng = evaluate_split(
        agent, "success", success_dir, rng
    )

    write_csv(output_dir / "per_frame.csv", all_frame_rows)
    write_csv(output_dir / "per_trajectory.csv", all_traj_rows)

    summary = {
        "exp_name": FLAGS.exp_name,
        "checkpoint_path": checkpoint_path,
        "num_frames": len(all_frame_rows),
        "moving_step_threshold_m": FLAGS.moving_step_threshold,
        "continuous_mse": summarize([r["continuous_mse"] for r in all_frame_rows]),
        "continuous_mae": summarize([r["continuous_mae"] for r in all_frame_rows]),
        "cosine_similarity": summarize([r["cosine_similarity"] for r in all_frame_rows]),
        "smoothness_mse": summarize([r["smoothness_mse"] for r in all_frame_rows]),
        "gripper_joint_accuracy": summarize(
            [r["gripper_joint_correct"] for r in all_frame_rows]
        ),
        "pred_step_mse": summarize(
            [r["pred_step_mse"] for r in all_traj_rows if r["pred_step_mse"] is not None]
        ),
        "pred_step_last_error": summarize(
            [
                r["pred_step_last_error"]
                for r in all_traj_rows
                if r["pred_step_last_error"] is not None
            ]
        ),
        "step_diagnostics": {
            "left": {
                "real_step_norm": _metric_summary(
                    [r.get("left_real_step_norm") for r in all_frame_rows]
                ),
                "eval_step_norm": _metric_summary(
                    [r.get("left_eval_step_norm") for r in all_frame_rows]
                ),
                "step_error_norm": _metric_summary(
                    [r.get("left_step_error_norm") for r in all_frame_rows]
                ),
                "relative_step_error_moving": _metric_summary(
                    [r.get("left_relative_step_error") for r in all_frame_rows],
                    [r.get("left_is_moving") for r in all_frame_rows],
                ),
                "direction_cosine_moving": _metric_summary(
                    [r.get("left_direction_cosine") for r in all_frame_rows],
                    [r.get("left_is_moving") for r in all_frame_rows],
                ),
                "moving_ratio": _mean_available(
                    [r.get("left_is_moving") for r in all_frame_rows]
                ),
            },
            "right": {
                "real_step_norm": _metric_summary(
                    [r.get("right_real_step_norm") for r in all_frame_rows]
                ),
                "eval_step_norm": _metric_summary(
                    [r.get("right_eval_step_norm") for r in all_frame_rows]
                ),
                "step_error_norm": _metric_summary(
                    [r.get("right_step_error_norm") for r in all_frame_rows]
                ),
                "relative_step_error_moving": _metric_summary(
                    [r.get("right_relative_step_error") for r in all_frame_rows],
                    [r.get("right_is_moving") for r in all_frame_rows],
                ),
                "direction_cosine_moving": _metric_summary(
                    [r.get("right_direction_cosine") for r in all_frame_rows],
                    [r.get("right_is_moving") for r in all_frame_rows],
                ),
                "moving_ratio": _mean_available(
                    [r.get("right_is_moving") for r in all_frame_rows]
                ),
            },
        },
    }
    with open(output_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    high_dir = output_dir / "high_error_frames"
    high_dir.mkdir(exist_ok=True)
    for rank, (_, row, obs) in enumerate(
        sorted(all_high_error, key=lambda item: item[0], reverse=True)[: FLAGS.high_error_count]
    ):
        export_frame(high_dir, rank, row, obs)

    # Generate visualizations
    if FLAGS.generate_plots:
        try:
            generate_visualizations(output_dir, all_frame_rows, all_traj_rows, summary)
        except Exception as e:
            print(f"Warning: Failed to generate visualizations: {e}")
            import traceback
            traceback.print_exc()
            print("CSV and JSON outputs are still available.")

    # Generate action and one-step end-effector comparison plots (standalone PNGs)
    if FLAGS.plot_trajectories and all_traj_xyz:
        try:
            plot_action_diagnostics(output_dir, all_traj_xyz)
            plot_step_action_trajectories(output_dir, all_traj_xyz)
        except Exception as e:
            print(f"Warning: Failed to generate trajectory comparison plots: {e}")
            import traceback
            traceback.print_exc()

    print_green(f"[BC Offline Eval] wrote results to {output_dir}")
    env.close()


if __name__ == "__main__":
    app.run(main)
