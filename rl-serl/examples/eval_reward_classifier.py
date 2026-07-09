#!/usr/bin/env python3
"""Evaluate trained reward classifier for OpenArm (rl-serl).

Migrated from rl_deploy/classifier/eval_classifier.py. Loads checkpoint and
evaluates on success demo images. The saved demo image_primary is already the
policy/classifier view produced by NetworkPrimaryImageCropWrapper.

Uses the SAME sampling rule as classifier training:
  - Positive samples: last N frames of success trajectories (task completed)
  - Negative samples: first M frames of success trajectories (task not yet completed)

This ensures training/evaluation data distribution consistency.

Extended with detailed analysis:
  - Per-class accuracy breakdown
  - Probability distribution histograms
  - Confusion matrix
  - Misclassified sample analysis
  - All results saved to checkpoint directory
"""
import compat  # noqa: F401

import glob
import os
import pickle as pkl

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from absl import app, flags
from flax.training import checkpoints
from tqdm import tqdm

from rl_launcher.networks import create_classifier
from experiments.artifacts import task_classifier_ckpt_dir, task_success_dir
from experiments.mappings import CONFIG_MAPPING

IMAGE_KEY = "image_primary"
HOLDOUT_TRAJ_COUNT = 3

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "openarm_pickplace", "Experiment name in CONFIG_MAPPING.")
flags.DEFINE_string("checkpoint_path", None, "Checkpoint directory. Defaults to task config.")
flags.DEFINE_integer("checkpoint_step", 0, "Checkpoint step (0=latest)")
flags.DEFINE_string("success_dir", None, "Success pkl directory. Defaults to the task folder.")
flags.DEFINE_string(
    "failure_dir",
    None,
    "Deprecated: negative samples now come from success trajectory head frames. Ignored if provided.",
)
flags.DEFINE_string(
    "image_key",
    None,
    "Classifier image key override. Defaults to the task config's single classifier key.",
)
flags.DEFINE_string("split", "holdout", "Data split to evaluate: 'train', 'holdout', or 'all'")


def resolve_task_settings():
    if FLAGS.exp_name not in CONFIG_MAPPING:
        raise ValueError(f"Experiment {FLAGS.exp_name!r} not found in CONFIG_MAPPING.")
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    classifier_keys = list(getattr(config, "classifier_keys", [IMAGE_KEY]))
    if FLAGS.image_key:
        image_key = FLAGS.image_key
    else:
        if len(classifier_keys) != 1:
            raise ValueError(
                "eval_reward_classifier.py currently supports one classifier image key. "
                f"Got classifier_keys={classifier_keys}; pass --image_key to choose one."
            )
        image_key = classifier_keys[0]
    checkpoint_path = FLAGS.checkpoint_path or getattr(
        config, "classifier_ckpt_path", str(task_classifier_ckpt_dir(FLAGS.exp_name))
    )
    success_dir = FLAGS.success_dir or str(task_success_dir(FLAGS.exp_name))
    return (
        config,
        image_key,
        os.path.abspath(checkpoint_path),
        os.path.abspath(success_dir),
    )


def split_pkl_paths(dir_path, holdout_count):
    if not os.path.exists(dir_path):
        return [], []
    pkl_paths = sorted(glob.glob(os.path.join(dir_path, "*.pkl")))
    if holdout_count <= 0:
        return pkl_paths, []
    if len(pkl_paths) <= holdout_count:
        return pkl_paths, pkl_paths
    return pkl_paths[:-holdout_count], pkl_paths[-holdout_count:]


def extract_traj_images(traj, image_key, frame_count=None, take_from="tail"):
    """Extract images from a trajectory.

    Args:
        traj: List of transition dicts
        image_key: Key to extract from next_observations
        frame_count: Number of frames to extract (None = all)
        take_from: "tail" for last N frames, "head" for first N frames
    """
    images = []
    if not isinstance(traj, list):
        return images

    if take_from == "tail":
        selected = traj[-frame_count:] if frame_count and frame_count > 0 else traj
    elif take_from == "head":
        selected = traj[:frame_count] if frame_count and frame_count > 0 else traj
    else:
        raise ValueError(f"Unknown take_from={take_from!r}, expected 'tail' or 'head'")

    for transition in selected:
        next_obs = transition.get("next_observations", {})
        img = next_obs.get(image_key)
        if img is None:
            continue
        img = np.asarray(img)
        if img.shape == (1, 128, 128, 3):
            img = img[0]
        if img.shape != (128, 128, 3):
            continue
        images.append(img[None, ...].astype(np.uint8))
    return images


def load_images_from_paths(pkl_paths, image_key, frame_count=None, take_from="tail"):
    images = []
    for path in pkl_paths:
        try:
            with open(path, "rb") as handle:
                traj = pkl.load(handle)
            images.extend(
                extract_traj_images(
                    traj,
                    image_key=image_key,
                    frame_count=frame_count,
                    take_from=take_from,
                )
            )
        except Exception as exc:
            print(f"Error loading trajectory {path}: {exc}")
    return images


def load_classifier_images(
    success_dir,
    image_key,
    success_tail_frames,
    failure_head_frames,
    *,
    split=None,
    holdout_count=0,
):
    """Load positive and negative samples from success trajectories.

    Positive samples: last N frames of success trajectories (task completed).
    Negative samples: first M frames of success trajectories (task not yet completed).

    This matches the training data distribution.

    Args:
        success_dir: Directory containing success trajectory pkl files
        image_key: Key to extract from observations
        success_tail_frames: Number of frames from trajectory end (positive samples)
        failure_head_frames: Number of frames from trajectory start (negative samples)
        split: "train", "holdout", or None (all data)
        holdout_count: Number of trajectories to hold out for validation
    """
    if split is None:
        success_paths = sorted(glob.glob(os.path.join(success_dir, "*.pkl")))
    else:
        success_train, success_holdout = split_pkl_paths(success_dir, holdout_count)
        if split == "train":
            success_paths = success_train
        elif split == "holdout":
            success_paths = success_holdout
        elif split == "all":
            success_paths = success_train + success_holdout
        else:
            raise ValueError(f"Unknown split={split!r}")

    # Positive samples: tail frames (task completed)
    success_images = load_images_from_paths(
        success_paths,
        image_key=image_key,
        frame_count=success_tail_frames,
        take_from="tail",
    )
    # Negative samples: head frames (task not yet completed)
    failure_images = load_images_from_paths(
        success_paths,
        image_key=image_key,
        frame_count=failure_head_frames,
        take_from="head",
    )
    return success_images, failure_images


def save_detailed_analysis(
    checkpoint_path,
    all_probs,
    all_labels,
    all_preds,
    misclassified,
    success_data,
    failure_data,
    split_name,
):
    """Save simplified evaluation analysis: probability distribution plot and misclassified images."""
    import cv2

    output_dir = os.path.abspath(checkpoint_path)
    os.makedirs(output_dir, exist_ok=True)

    # Convert to numpy arrays
    probs = np.array(all_probs)
    labels = np.array(all_labels)
    preds = np.array(all_preds)

    # Split probabilities by class
    success_probs = probs[labels == 1]
    failure_probs = probs[labels == 0]

    print(f"\n{'='*60}")
    print("DETAILED STATISTICS")
    print(f"{'='*60}")
    print(f"Success class (label=1):")
    print(f"  Mean prob: {success_probs.mean():.4f}")
    print(f"  Std prob:  {success_probs.std():.4f}")
    print(f"  Min prob:  {success_probs.min():.4f}")
    print(f"  Max prob:  {success_probs.max():.4f}")
    print(f"\nFailure class (label=0):")
    print(f"  Mean prob: {failure_probs.mean():.4f}")
    print(f"  Std prob:  {failure_probs.std():.4f}")
    print(f"  Min prob:  {failure_probs.min():.4f}")
    print(f"  Max prob:  {failure_probs.max():.4f}")
    print(f"\nMisclassified samples: {len(misclassified)}")
    print(f"{'='*60}")

    # 1. Save probability distribution histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(success_probs, bins=30, alpha=0.6, label='Success (label=1)', color='green', edgecolor='black')
    axes[0].hist(failure_probs, bins=30, alpha=0.6, label='Failure (label=0)', color='red', edgecolor='black')
    axes[0].axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold=0.5')
    axes[0].set_xlabel('Predicted Probability', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title(f'Probability Distribution ({split_name} split)', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot
    data_for_box = [failure_probs, success_probs]
    axes[1].boxplot(data_for_box, tick_labels=['Failure (label=0)', 'Success (label=1)'], patch_artist=True)
    axes[1].axhline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold=0.5')
    axes[1].set_ylabel('Predicted Probability', fontsize=12)
    axes[1].set_title('Probability Distribution (Box Plot)', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].legend()

    plt.tight_layout()
    hist_path = os.path.join(output_dir, f'eval_probability_distribution_{split_name}.png')
    plt.savefig(hist_path, dpi=150)
    print(f"Saved: {hist_path}")
    plt.close()

    # 2. Save misclassified sample images
    if misclassified:
        misc_img_dir = os.path.join(output_dir, f'misclassified_samples_{split_name}')
        os.makedirs(misc_img_dir, exist_ok=True)

        for i, m in enumerate(misclassified[:20]):  # Save first 20
            img = m['image']
            if img.shape == (1, 128, 128, 3):
                img = img[0]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            error_type = "FN" if m['label'] == 1 else "FP"
            filename = f"{i:03d}_{error_type}_true{m['label']}_pred{m['pred']}_prob{m['prob']:.3f}.png"
            img_path = os.path.join(misc_img_dir, filename)
            cv2.imwrite(img_path, img_rgb)

        print(f"Saved: {min(20, len(misclassified))} misclassified images to {misc_img_dir}/")


def main(_):
    global IMAGE_KEY

    config, image_key, checkpoint_path, success_dir = resolve_task_settings()
    IMAGE_KEY = image_key

    # Read classifier frame counts from config
    success_tail_frames = getattr(config, "classifier_success_tail_frames", 10)
    failure_head_frames = getattr(config, "classifier_failure_head_frames", 10)

    success_images, failure_images = load_classifier_images(
        success_dir,
        IMAGE_KEY,
        success_tail_frames,
        failure_head_frames,
        split=FLAGS.split,
        holdout_count=HOLDOUT_TRAJ_COUNT,
    )
    success_data = [(img, 1) for img in success_images]
    failure_data = [(img, 0) for img in failure_images]
    all_data = success_data + failure_data

    print(f"Exp: {FLAGS.exp_name}")
    print(f"Classifier keys from config: {getattr(config, 'classifier_keys', None)}")
    print(f"Success directory: {success_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Image key: {IMAGE_KEY}")
    print(f"Evaluation split: {FLAGS.split}")
    print(f"Success tail frames per trajectory: {success_tail_frames}")
    print(f"Failure head frames per trajectory: {failure_head_frames}")
    print("Positive samples: last N frames of success trajectories (task completed)")
    print("Negative samples: first M frames of success trajectories (task not yet completed)")
    print(f"Loaded {len(success_data)} success images, {len(failure_data)} failure images")

    if not all_data:
        print("No demo data found.")
        return

    rng = jax.random.PRNGKey(0)
    sample_obs = {IMAGE_KEY: all_data[0][0]}
    classifier = create_classifier(rng, sample_obs, image_keys=[IMAGE_KEY])

    if FLAGS.checkpoint_step:
        classifier = checkpoints.restore_checkpoint(
            checkpoint_path, classifier, step=FLAGS.checkpoint_step
        )
    else:
        classifier = checkpoints.restore_checkpoint(checkpoint_path, classifier)

    print("\nRunning inference...")
    correct = 0
    success_correct = 0
    failure_correct = 0

    # Collect detailed results
    all_probs = []
    all_labels = []
    all_preds = []
    misclassified = []

    for idx, (img, label) in enumerate(tqdm(all_data)):
        obs = {IMAGE_KEY: img}
        logits = classifier.apply_fn({"params": classifier.params}, obs, train=False)
        # Squeeze to scalar before int() conversion: (1,) or (1, 1) → ()
        if logits.ndim > 0:
            logits = logits.squeeze()

        # Get probability and prediction
        prob = float(jax.nn.sigmoid(logits))
        pred = int(prob >= 0.5)

        all_probs.append(prob)
        all_labels.append(label)
        all_preds.append(pred)

        if pred == label:
            correct += 1
            if label == 1:
                success_correct += 1
            else:
                failure_correct += 1
        else:
            # Record misclassified samples
            misclassified.append({
                'idx': idx,
                'label': label,
                'pred': pred,
                'prob': prob,
                'image': img,
            })

    accuracy = correct / len(all_data)
    success_accuracy = success_correct / len(success_data) if success_data else 0.0
    failure_accuracy = failure_correct / len(failure_data) if failure_data else 0.0

    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {accuracy:.4f} ({correct}/{len(all_data)})")
    print(f"Success Accuracy: {success_accuracy:.4f} ({success_correct}/{len(success_data)})")
    print(f"Failure Accuracy: {failure_accuracy:.4f} ({failure_correct}/{len(failure_data)})")
    print(f"{'='*60}")

    # Save detailed analysis
    save_detailed_analysis(
        checkpoint_path,
        all_probs,
        all_labels,
        all_preds,
        misclassified,
        success_data,
        failure_data,
        FLAGS.split,
    )


if __name__ == "__main__":
    app.run(main)
