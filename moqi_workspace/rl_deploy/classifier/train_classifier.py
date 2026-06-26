import glob
import os
import pickle as pkl

import cv2
import flax.linen as nn
import gym
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
from absl import app, flags
from flax.training import checkpoints
from gym import spaces
from jax import numpy as jnp
from tqdm import tqdm

from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from serl_launcher.networks.reward_classifier import create_classifier
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop

plt.switch_backend("Agg")


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "checkpoint_path",
    "/home/sj/Desktop/zy/moqi_workspace/rl_deploy/classifier/classifier_ckpt",
    "Path to save checkpoint",
)
flags.DEFINE_integer("batch_size", 64, "Batch size")
flags.DEFINE_integer("num_epochs", 100, "Number of epochs")


SUCCESS_DIR = "/home/sj/Desktop/zy/moqi_workspace/rl_deploy/demo/collected/success"
IMAGE_KEY = "image_primary"
SUCCESS_TAIL_FRAMES_PER_TRAJ = 30
FAILURE_HEAD_FRAMES_PER_TRAJ = 30
HOLDOUT_TRAJ_COUNT = 3


def populate_data_store_from_images(data_store, images):
    print(f"Populating data store with {len(images)} images...")
    for img in tqdm(images):
        transition = {
            "observations": {
                "image_0": img,
                "state": np.zeros((1, 14), dtype=np.float32),
            },
            "next_observations": {
                "image_0": img,
                "state": np.zeros((1, 14), dtype=np.float32),
            },
            "actions": np.zeros((1, 14), dtype=np.float32),
            "rewards": np.array([0.0], dtype=np.float32),
            "masks": np.array([1.0], dtype=np.float32),
            "dones": np.array([0.0], dtype=np.float32),
        }
        data_store.insert(transition)
    return data_store


def fix_image_shape(x):
    if isinstance(x, np.ndarray):
        x = jnp.array(x)
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
        return [], pkl_paths
    return pkl_paths[:-holdout_count], pkl_paths[-holdout_count:]


def extract_traj_images(traj, frame_count, take_from="tail"):
    images = []
    if not isinstance(traj, list):
        return images

    if frame_count > 0:
        if take_from == "head":
            selected = traj[:frame_count]
        else:
            selected = traj[-frame_count:]
    else:
        selected = traj

    for transition in selected:
        next_obs = transition.get("next_observations", {})
        img = next_obs.get(IMAGE_KEY)
        if img is None:
            continue
        img = np.asarray(img)
        if img.shape == (1, 128, 128, 3):
            img = img[0]
        if img.shape != (128, 128, 3):
            continue
        images.append(img[None, ...].astype(np.uint8))
    return images


def load_images_from_paths(pkl_paths, frame_count, take_from="tail"):
    images = []
    for path in pkl_paths:
        try:
            with open(path, "rb") as handle:
                traj = pkl.load(handle)
            images.extend(
                extract_traj_images(traj, frame_count=frame_count, take_from=take_from)
            )
        except Exception as exc:
            print(f"Error loading trajectory {path}: {exc}")
    return images


def load_images_from_transition_dir(dir_path, frame_count, take_from="tail", split="train"):
    if not os.path.exists(dir_path):
        print(f"Transition dir not found, skip: {dir_path}")
        return []

    train_paths, holdout_paths = split_pkl_paths(dir_path, HOLDOUT_TRAJ_COUNT)
    if split == "holdout":
        selected_paths = holdout_paths
    elif split == "all":
        selected_paths = train_paths + holdout_paths
    else:
        selected_paths = train_paths

    print(
        f"Found {len(train_paths) + len(holdout_paths)} trajectories in {dir_path} "
        f"(train={len(train_paths)}, holdout={len(holdout_paths)}, using={len(selected_paths)} for {split})"
    )
    return load_images_from_paths(selected_paths, frame_count=frame_count, take_from=take_from)


def main(_):
    STATE_DIM = 14
    ACTION_DIM = 14
    image_keys = ["image_0"]

    observation_space = spaces.Dict(
        {
            "image_0": spaces.Box(0, 255, shape=(1, 128, 128, 3), dtype=np.uint8),
            "state": spaces.Box(-np.inf, np.inf, shape=(1, STATE_DIM), dtype=np.float32),
        }
    )
    action_space = spaces.Box(-1, 1, shape=(1, ACTION_DIM), dtype=np.float32)

    pos_buffer = MemoryEfficientReplayBufferDataStore(
        observation_space, action_space, capacity=10000, image_keys=image_keys
    )
    neg_buffer = MemoryEfficientReplayBufferDataStore(
        observation_space, action_space, capacity=10000, image_keys=image_keys
    )

    success_images = load_images_from_transition_dir(
        SUCCESS_DIR,
        frame_count=SUCCESS_TAIL_FRAMES_PER_TRAJ,
        take_from="tail",
        split="train",
    )
    failure_images = load_images_from_transition_dir(
        SUCCESS_DIR,
        frame_count=FAILURE_HEAD_FRAMES_PER_TRAJ,
        take_from="head",
        split="train",
    )
    val_success_images = load_images_from_transition_dir(
        SUCCESS_DIR,
        frame_count=SUCCESS_TAIL_FRAMES_PER_TRAJ,
        take_from="tail",
        split="holdout",
    )
    val_failure_images = load_images_from_transition_dir(
        SUCCESS_DIR,
        frame_count=FAILURE_HEAD_FRAMES_PER_TRAJ,
        take_from="head",
        split="holdout",
    )

    print("\nSummary:")
    print(f" - Total Success Images: {len(success_images)}")
    print(f" - Total Failure Images: {len(failure_images)}")
    print(f" - Validation Success Images: {len(val_success_images)}")
    print(f" - Validation Failure Images: {len(val_failure_images)}")
    print(" - Failure Image Source: success trajectories (head frames)")
    print(f" - Success Tail Frames Per Trajectory: {SUCCESS_TAIL_FRAMES_PER_TRAJ}")
    print(f" - Failure Head Frames Per Trajectory: {FAILURE_HEAD_FRAMES_PER_TRAJ}")
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
        for pixel_key in image_keys:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng_key, padding=4, num_batch_dims=2
                    )
                }
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
    logs = {
        "epoch": [],
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    best_val_accuracy = 0.0
    state = classifier

    val_images = val_success_images + val_failure_images
    val_labels = np.concatenate(
        [
            np.ones((len(val_success_images), 1), dtype=np.float32),
            np.zeros((len(val_failure_images), 1), dtype=np.float32),
        ],
        axis=0,
    )
    val_data_np = np.stack(val_images, axis=0)
    val_data = {
        "image_0": fix_image_shape(val_data_np),
        "state": jnp.zeros((len(val_images), 1, STATE_DIM), dtype=jnp.float32),
    }
    val_labels = jnp.asarray(val_labels)

    for epoch in range(FLAGS.num_epochs):
        losses = []
        accuracies = []
        num_steps = max(1, min(len(pos_buffer), len(neg_buffer)) // max(FLAGS.batch_size // 2, 1))

        for _ in tqdm(range(num_steps), desc=f"Epoch {epoch + 1}/{FLAGS.num_epochs}"):
            rng, aug_key, step_key = jax.random.split(rng, 3)
            pos_batch = next(pos_iterator)
            neg_batch = next(neg_iterator)
            batch = concat_batches(pos_batch, neg_batch, axis=0)
            labels = jnp.concatenate(
                [
                    jnp.ones((FLAGS.batch_size // 2, 1), dtype=jnp.float32),
                    jnp.zeros((FLAGS.batch_size // 2, 1), dtype=jnp.float32),
                ],
                axis=0,
            )
            data = data_augmentation_fn(aug_key, batch["observations"])
            train_batch = {"data": data, "labels": labels}
            state, loss, accuracy = train_step(state, train_batch, step_key)
            losses.append(float(loss))
            accuracies.append(float(accuracy))

        epoch_loss = float(np.mean(losses))
        epoch_accuracy = float(np.mean(accuracies))
        val_loss, val_accuracy = eval_step(
            state,
            {"data": val_data, "labels": val_labels},
        )
        val_loss = float(val_loss)
        val_accuracy = float(val_accuracy)
        logs["epoch"].append(epoch + 1)
        logs["loss"].append(epoch_loss)
        logs["accuracy"].append(epoch_accuracy)
        logs["val_loss"].append(val_loss)
        logs["val_accuracy"].append(val_accuracy)
        print(
            f"Epoch {epoch + 1}: "
            f"loss={epoch_loss:.4f} accuracy={epoch_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f}"
        )

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path),
                state,
                step=epoch + 1,
                overwrite=True,
            )

    os.makedirs(FLAGS.checkpoint_path, exist_ok=True)
    pd.DataFrame(logs).to_csv(os.path.join(FLAGS.checkpoint_path, "train_metrics.csv"), index=False)

    plt.figure(figsize=(8, 4))
    plt.plot(logs["epoch"], logs["loss"], label="loss")
    plt.plot(logs["epoch"], logs["accuracy"], label="accuracy")
    plt.legend()
    plt.xlabel("epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(FLAGS.checkpoint_path, "train_metrics.png"))
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    app.run(main)
