#!/usr/bin/env python3
"""Train binary reward classifier for OpenArm (rl-serl).


Training data: positive samples from the last N frames of success trajectories,
negative samples from the first M frames of success trajectories. N and M are
configured by the task config's classifier_success_tail_frames and
classifier_failure_head_frames. This creates a temporal contrast "before success"
vs "after success" that is easier to learn than "complete failure" vs "success".
"""
import project_paths  # noqa: F401

import glob
import os
import pickle as pkl

import flax.linen as nn
import gymnasium as gym
import jax
import matplotlib.pyplot as plt
import numpy as np
from data_contract import validate_transition
import optax
import pandas as pd
from absl import app, flags
from flax.training import checkpoints
from gymnasium import spaces
from jax import numpy as jnp
from tqdm import tqdm

from rl_launcher.data import MemoryEfficientReplayBufferDataStore
from rl_launcher.networks import create_classifier
from rl_launcher.utils import concat_batches
from rl_launcher.vision import batched_random_crop

from experiments.artifacts import (
    task_classifier_ckpt_dir,
    task_success_dir,
)
from experiments.mappings import CONFIG_MAPPING

plt.switch_backend("Agg")

# Image key matching the policy stream (post-SERLObsWrapper)
IMAGE_KEY = "image_primary"

# Training hyperparams
HOLDOUT_TRAJ_COUNT = 3

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "openarm_pickplace", "Experiment name in CONFIG_MAPPING.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoint. Defaults to task config.")
flags.DEFINE_integer("batch_size", 64, "Batch size")
flags.DEFINE_integer("num_epochs", 100, "Number of epochs")
flags.DEFINE_string(
    "success_dir",
    None,
    "Success pkl directory. Defaults to the task folder.",
)
flags.DEFINE_string(
    "image_key",
    None,
    "Classifier image key override. Defaults to the task config's single classifier key.",
)


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
                "train_reward_classifier.py currently supports one classifier image key. "
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


def fix_image_shape(x):
    """Reshape batch to (B, 1, 128, 128, 3) for classifier network."""
    shape = x.shape
    if len(shape) == 4 and shape[1] == 128 and shape[2] == 128 and shape[3] == 3:
        x = jnp.expand_dims(x, axis=1)
    final_shape = x.shape
    if len(final_shape) != 5 or final_shape[1:] != (1, 128, 128, 3):
        x = jnp.reshape(x, (shape[0], 1, 128, 128, 3))
    return x


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
    for idx, transition in enumerate(traj):
        validate_transition(transition, source=f"classifier trajectory[{idx}]")

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

    This creates a temporal contrast that is easier to learn than using completely
    failed trajectories, which may contain noise and irrelevant states.

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


def populate_data_store_from_images(data_store, images):
    fake_action_space = spaces.Box(-1, 1, shape=(14,), dtype=np.float32)
    for img in images:
        transition = {
            "observations": {IMAGE_KEY: img},
            "next_observations": {IMAGE_KEY: img},
            "actions": np.zeros(fake_action_space.shape, dtype=np.float32),
            "rewards": np.asarray(0.0, dtype=np.float32),
            "masks": np.asarray(1.0, dtype=np.float32),
            "dones": False,
        }
        data_store.insert(transition)


def main(_):
    global IMAGE_KEY

    config, image_key, checkpoint_path, success_dir = resolve_task_settings()
    IMAGE_KEY = image_key
    image_keys = [IMAGE_KEY]
    os.makedirs(checkpoint_path, exist_ok=True)

    fake_obs_space = spaces.Dict({IMAGE_KEY: spaces.Box(0, 255, (1, 128, 128, 3), np.uint8)})
    fake_action_space = spaces.Box(-1, 1, shape=(14,), dtype=np.float32)

    pos_buffer = MemoryEfficientReplayBufferDataStore(
        fake_obs_space, fake_action_space, capacity=100000, image_keys=image_keys
    )
    neg_buffer = MemoryEfficientReplayBufferDataStore(
        fake_obs_space, fake_action_space, capacity=100000, image_keys=image_keys
    )

    # Read classifier frame counts from config
    success_tail_frames = getattr(config, "classifier_success_tail_frames", 10)
    failure_head_frames = getattr(config, "classifier_failure_head_frames", 10)

    print(f"Exp: {FLAGS.exp_name}")
    print(f"Classifier keys from config: {getattr(config, 'classifier_keys', None)}")
    print(f"Success directory: {success_dir}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Image key: {IMAGE_KEY}")
    print(f"Success tail frames per trajectory: {success_tail_frames}")
    print(f"Failure head frames per trajectory: {failure_head_frames}")
    print("Positive samples: last N frames of success trajectories (task completed)")
    print("Negative samples: first M frames of success trajectories (task not yet completed)")
    print("Image preprocessing: using demo image_primary as stored (no extra crop)")

    success_images, failure_images = load_classifier_images(
        success_dir,
        IMAGE_KEY,
        success_tail_frames,
        failure_head_frames,
        split="train",
        holdout_count=HOLDOUT_TRAJ_COUNT,
    )
    val_success_images, val_failure_images = load_classifier_images(
        success_dir,
        IMAGE_KEY,
        success_tail_frames,
        failure_head_frames,
        split="holdout",
        holdout_count=HOLDOUT_TRAJ_COUNT,
    )

    print("\nSummary:")
    print(f" - Total Success Images: {len(success_images)}")
    print(f" - Total Failure Images: {len(failure_images)}")
    print(f" - Validation Success Images: {len(val_success_images)}")
    print(f" - Validation Failure Images: {len(val_failure_images)}")
    print(" - Positive Image Source: success trajectories (tail frames)")
    print(" - Negative Image Source: success trajectories (head frames)")
    print(f" - Success Tail Frames Per Trajectory: {success_tail_frames}")
    print(f" - Failure Head Frames Per Trajectory: {failure_head_frames}")
    print(f" - Holdout Trajectories Per Class: {HOLDOUT_TRAJ_COUNT}")
    print(f" - Training Batch Size: {FLAGS.batch_size}")
    print(f" - Total Epochs: {FLAGS.num_epochs}")

    if not success_images:
        raise ValueError("No success images found.")
    if not failure_images:
        raise ValueError("No failure images found.")
    if not val_success_images:
        raise ValueError("No validation success images found.")
    if not val_failure_images:
        raise ValueError("No validation failure images found.")

    populate_data_store_from_images(pos_buffer, success_images)
    populate_data_store_from_images(neg_buffer, failure_images)

    devices = jax.local_devices()
    print(f"\nJAX Devices: {devices}")
    mesh = jax.sharding.Mesh(devices, ("batch",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("batch"))

    pos_iterator = pos_buffer.get_iterator(
        sample_args={"batch_size": FLAGS.batch_size // 2, "pack_obs_and_next_obs": False},
        device=sharding,
    )
    neg_iterator = neg_buffer.get_iterator(
        sample_args={"batch_size": FLAGS.batch_size // 2, "pack_obs_and_next_obs": False},
        device=sharding,
    )

    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)

    init_batch = next(pos_iterator)
    init_obs_processed = {}
    for key_name in image_keys:
        batch_data = fix_image_shape(init_batch["observations"][key_name])
        init_obs_processed[key_name] = batch_data[0]

    classifier = create_classifier(key, init_obs_processed, image_keys)

    def data_augmentation_fn(rng_key, observations):
        observations = observations.copy()
        for pixel_key in image_keys:
            observations[pixel_key] = batched_random_crop(
                observations[pixel_key], rng_key, padding=4, num_batch_dims=2
            )
        return observations

    @jax.jit
    def train_step(state, batch, key):
        def loss_fn(params):
            logits = state.apply_fn({"params": params}, batch["data"], rngs={"dropout": key}, train=True)
            return optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        logits = state.apply_fn({"params": state.params}, batch["data"], train=False, rngs={"dropout": key})
        train_accuracy = jnp.mean((nn.sigmoid(logits) >= 0.5) == batch["labels"])
        return state.apply_gradients(grads=grads), loss, train_accuracy

    @jax.jit
    def eval_step(state, batch):
        logits = state.apply_fn({"params": state.params}, batch["data"], train=False)
        loss = optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()
        accuracy = jnp.mean((nn.sigmoid(logits) >= 0.5) == batch["labels"])
        return loss, accuracy

    print("Starting training...")
    logs = {"epoch": [], "loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    best_val_accuracy = 0.0

    # Prepare validation data (once, outside training loop)
    val_data = {}
    for key_name in image_keys:
        pos_imgs = jnp.stack([fix_image_shape(img)[0] for img in val_success_images])
        neg_imgs = jnp.stack([fix_image_shape(img)[0] for img in val_failure_images])
        val_data[key_name] = jnp.concatenate([pos_imgs, neg_imgs], axis=0)

    val_labels = jnp.concatenate([
        jnp.ones((len(val_success_images), 1)),
        jnp.zeros((len(val_failure_images), 1))
    ])
    val_batch = {"data": val_data, "labels": val_labels}

    for epoch in range(FLAGS.num_epochs):
        epoch_losses = []
        epoch_accuracies = []
        steps_per_epoch = min(len(pos_buffer), len(neg_buffer)) // (FLAGS.batch_size // 2)

        for _ in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{FLAGS.num_epochs}"):
            rng, aug_key, train_key = jax.random.split(rng, 3)
            pos_batch = next(pos_iterator)
            neg_batch = next(neg_iterator)
            batch = concat_batches(pos_batch, neg_batch, axis=0)

            batch_data = {}
            for key_name in image_keys:
                batch_data[key_name] = fix_image_shape(batch["observations"][key_name])
            batch_data = data_augmentation_fn(aug_key, batch_data)

            labels = jnp.concatenate([
                jnp.ones((FLAGS.batch_size // 2, 1)),
                jnp.zeros((FLAGS.batch_size // 2, 1))
            ])
            train_batch = {"data": batch_data, "labels": labels}
            classifier, loss, accuracy = train_step(classifier, train_batch, train_key)
            epoch_losses.append(float(loss))
            epoch_accuracies.append(float(accuracy))

        val_loss, val_accuracy = eval_step(classifier, val_batch)

        logs["epoch"].append(epoch + 1)
        logs["loss"].append(float(np.mean(epoch_losses)))
        logs["accuracy"].append(float(np.mean(epoch_accuracies)))
        logs["val_loss"].append(float(val_loss))
        logs["val_accuracy"].append(float(val_accuracy))

        print(
            f"Epoch {epoch+1}: "
            f"loss={logs['loss'][-1]:.4f}, "
            f"acc={logs['accuracy'][-1]:.4f}, "
            f"val_loss={logs['val_loss'][-1]:.4f}, "
            f"val_acc={logs['val_accuracy'][-1]:.4f}"
        )

        # Save checkpoint if validation accuracy improves
        if logs['val_accuracy'][-1] >= best_val_accuracy:
            best_val_accuracy = logs['val_accuracy'][-1]
            checkpoints.save_checkpoint(
                checkpoint_path, classifier, step=epoch + 1, keep=5, overwrite=True
            )
            print(f"  -> Best model saved (val_acc={best_val_accuracy:.4f})")
        elif (epoch + 1) % 10 == 0:
            checkpoints.save_checkpoint(
                checkpoint_path, classifier, step=epoch + 1, keep=5, overwrite=True
            )
            print(f"  Checkpoint saved at epoch {epoch+1}")

    df = pd.DataFrame(logs)
    csv_path = os.path.join(checkpoint_path, "training_log.csv")
    df.to_csv(csv_path, index=False)
    print(f"Training log saved to {csv_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df["epoch"], df["loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(df["epoch"], df["accuracy"], label="train")
    axes[1].plot(df["epoch"], df["val_accuracy"], label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plot_path = os.path.join(checkpoint_path, "training_curves.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Training curves saved to {plot_path}")
    print(f"\nTraining complete! Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    app.run(main)
