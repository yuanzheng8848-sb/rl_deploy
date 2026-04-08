import jax
from jax import numpy as jnp
import optax
from tqdm import tqdm
from absl import app, flags
from flax.training import checkpoints
import flax.linen as nn
import pickle as pkl
import numpy as np
import os
import copy
import cv2
import glob
import time
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import pandas as pd

from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.networks.reward_classifier import create_classifier
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

import gym
from gym import spaces

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_path", "/home/sj/Desktop/zy/moqi_workspace/rl_deploy/classifier/classifier_ckpt", "Path to save checkpoint")
flags.DEFINE_integer("batch_size", 64, "Batch size")
flags.DEFINE_integer("num_epochs", 100, "Number of epochs")

# 使用 demo/record_data：每 session 的 cam_2 前 10 帧=失败，后 10 帧=成功
RECORD_DATA_DIR = "/home/sj/Desktop/zy/moqi_workspace/rl_deploy/demo/record_data"
CAM_SUBDIR = "cam_2_rgb"
N_FAILURE_FRAMES = 10  # 每 session 取前 N 帧为失败
N_SUCCESS_FRAMES = 10  # 每 session 取后 N 帧为成功
EXTRA_FAILURE_DIR = "/home/sj/Desktop/zy/moqi_workspace/rl_deploy/classifier/extra_failure_images"


def populate_data_store_from_images(data_store, images):
    print(f"Populating data store with {len(images)} images...")
    for img in tqdm(images):
        # Create dummy transition
        transition = {
            "observations": {
                "image_0": img, # (1, 128, 128, 3)
                "state": np.zeros((1, 14), dtype=np.float32)
            },
            "next_observations": {
                "image_0": img,
                "state": np.zeros((1, 14), dtype=np.float32)
            },
            "actions": np.zeros((1, 14), dtype=np.float32),
            "rewards": np.array([0.0], dtype=np.float32), 
            "masks": np.array([1.0], dtype=np.float32),
            "dones": np.array([0.0], dtype=np.float32)
        }
        data_store.insert(transition)
    return data_store

def fix_image_shape(x):
    """
    Ensure image shape is (Batch, 1, 128, 128, 3)
    """
    shape = x.shape
    # If (B, 128, 128, 3), add time dim -> (B, 1, 128, 128, 3)
    if len(shape) == 4 and shape[1] == 128 and shape[2] == 128 and shape[3] == 3:
        x = jnp.expand_dims(x, axis=1)
    
    final_shape = x.shape
    if len(final_shape) != 5 or final_shape[1] != 1 or final_shape[2] != 128 or final_shape[3] != 128 or final_shape[4] != 3:
         # Try to reshape if total elements match
         try:
             x = jnp.reshape(x, (shape[0], 1, 128, 128, 3))
         except:
             # If that fails, maybe it was (B, 128, 128, 3) and we need to expand
             try:
                 x = jnp.reshape(x, (shape[0], 128, 128, 3))
                 x = jnp.expand_dims(x, axis=1)
             except:
                 raise ValueError(f"Could not fix shape {shape} to (B, 1, 128, 128, 3)")
    return x

def load_extra_failure_images(extra_failure_dir):
    images = []
    if not os.path.exists(extra_failure_dir):
        print(f"Extra failure image dir not found, skip: {extra_failure_dir}")
        return images
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    paths = []
    for p in patterns:
        paths.extend(glob.glob(os.path.join(extra_failure_dir, p)))
    paths = sorted(paths)
    print(f"Found {len(paths)} extra failure images in {extra_failure_dir}")
    for path in paths:
        try:
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 128))
            images.append(img[None, ...])
        except Exception as e:
            print(f"Error loading extra failure image {path}: {e}")
    return images

def main(_):
    STATE_DIM = 14 
    ACTION_DIM = 14 
    image_keys = ["image_0"]
    
    observation_space = spaces.Dict({
        "image_0": spaces.Box(0, 255, shape=(1, 128, 128, 3), dtype=np.uint8),
        "state": spaces.Box(-np.inf, np.inf, shape=(1, STATE_DIM), dtype=np.float32)
    })
    action_space = spaces.Box(-1, 1, shape=(1, ACTION_DIM), dtype=np.float32)

    pos_buffer = MemoryEfficientReplayBufferDataStore(
        observation_space, action_space, capacity=10000, image_keys=image_keys
    )
    neg_buffer = MemoryEfficientReplayBufferDataStore(
        observation_space, action_space, capacity=10000, image_keys=image_keys
    )
    
    # --- Load Data ---
    # 规则：每个 session 下 cam_2_rgb 前 N 帧=失败，后 N 帧=成功
    success_images = []
    failure_images = []
    
    if not os.path.exists(RECORD_DATA_DIR):
        print(f"Warning: Record data dir not found: {RECORD_DATA_DIR}")
    else:
        sessions = sorted([d for d in os.listdir(RECORD_DATA_DIR) if os.path.isdir(os.path.join(RECORD_DATA_DIR, d)) and d.startswith("session_")])
        print(f"Found {len(sessions)} sessions in {RECORD_DATA_DIR}")
        print(f"Rule: per session cam_2_rgb first {N_FAILURE_FRAMES} frames = FAILURE, last {N_SUCCESS_FRAMES} frames = SUCCESS")

        session_stats = []
        for session in sessions:
            img_dir = os.path.join(RECORD_DATA_DIR, session, "images", CAM_SUBDIR)
            if not os.path.exists(img_dir):
                continue
            image_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
            total = len(image_paths)
            if total < N_FAILURE_FRAMES + N_SUCCESS_FRAMES:
                session_stats.append((session, 0, 0, "SKIP (too few frames)"))
                continue

            n_fail = min(N_FAILURE_FRAMES, total)
            n_succ = min(N_SUCCESS_FRAMES, total)
            failure_paths = image_paths[:n_fail]
            success_paths = image_paths[-n_succ:]

            for path in failure_paths:
                try:
                    img = cv2.imread(path)
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (128, 128))
                    img = img[None, ...]
                    failure_images.append(img)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
            for path in success_paths:
                try:
                    img = cv2.imread(path)
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (128, 128))
                    img = img[None, ...]
                    success_images.append(img)
                except Exception as e:
                    print(f"Error loading {path}: {e}")

            session_stats.append((session, n_fail, n_succ, "OK"))

        print("\n" + "="*60)
        print(f"{'Session':<35} | {'Fail#':<6} | {'Succ#':<6} | Status")
        print("-" * 60)
        for name, nf, ns, status in session_stats:
            print(f"{name:<35} | {nf:<6} | {ns:<6} | {status}")
        print("="*60)

    extra_failure_images = load_extra_failure_images(EXTRA_FAILURE_DIR)
    failure_images.extend(extra_failure_images)

    print(f"\nSummary:")
    print(f" - Total Success Images: {len(success_images)}")
    print(f" - Total Failure Images: {len(failure_images)}")
    print(f" - Extra Failure Images: {len(extra_failure_images)}")
    print(f" - Training Batch Size:  {FLAGS.batch_size}")
    print(f" - Total Epochs:         {FLAGS.num_epochs}")

    if not success_images:
        raise ValueError("No success images found.")
    if not failure_images:
        raise ValueError("No failure images found.")

    populate_data_store_from_images(pos_buffer, success_images)
    populate_data_store_from_images(neg_buffer, failure_images)

    if len(pos_buffer) == 0 or len(neg_buffer) == 0:
        raise ValueError("Buffers cannot be empty.")

    devices = jax.local_devices()
    print(f"\nJAX Devices: {devices}")
    mesh = jax.sharding.Mesh(devices, ('batch',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('batch'))
    
    pos_iterator = pos_buffer.get_iterator(
        sample_args={"batch_size": FLAGS.batch_size // 2, "pack_obs_and_next_obs": False}, 
        device=sharding
    )
    neg_iterator = neg_buffer.get_iterator(
        sample_args={"batch_size": FLAGS.batch_size // 2, "pack_obs_and_next_obs": False}, 
        device=sharding
    )

    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    
    # --- Init Network ---
    init_batch = next(pos_iterator)
    init_obs_processed = {}
    for k in image_keys:
        # 获取 Batch 数据
        batch_data = fix_image_shape(init_batch["observations"][k])
        # 只取第一条数据 (Single Sample) 用于初始化 -> (1, 128, 128, 3)
        init_obs_processed[k] = batch_data[0]

    print(f"Init sample shape: {init_obs_processed['image_0'].shape}") # 应为 (1, 128, 128, 3)
    classifier = create_classifier(key, init_obs_processed, image_keys)
    # --------------------

    def data_augmentation_fn(rng, observations):
        for pixel_key in image_keys:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
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

    print("Starting training...")
    logs = {
        "epoch": [],
        "loss": [],
        "accuracy": [],
        "epoch_time": [],
        "cumulative_time": [],
        "samples_per_second": []
    }
    
    start_time = time.time()
    cumulative_time = 0
    
    for epoch in tqdm(range(FLAGS.num_epochs)):
        epoch_start = time.time()
        
        try:
            pos_sample = next(pos_iterator)
            neg_sample = next(neg_iterator)
        except StopIteration:
            continue

        def process_obs_batch(obs_dict):
            new_obs = {}
            for k in image_keys:
                new_obs[k] = fix_image_shape(obs_dict[k])
            return new_obs

        pos_obs = process_obs_batch(pos_sample["observations"])
        neg_obs = process_obs_batch(neg_sample["observations"])

        sample = concat_batches(pos_obs, neg_obs, axis=0)
        
        rng, key = jax.random.split(rng)
        sample = data_augmentation_fn(key, sample)

        labels = jnp.concatenate([
            jnp.ones((FLAGS.batch_size // 2, 1)), 
            jnp.zeros((FLAGS.batch_size // 2, 1))
        ], axis=0)
        
        batch = {"data": sample, "labels": labels}

        rng, key = jax.random.split(rng)
        classifier, loss, acc = train_step(classifier, batch, key)
        
        epoch_end = time.time()
        duration = epoch_end - epoch_start
        cumulative_time += duration
        
        # Record logs
        logs["epoch"].append(epoch)
        logs["loss"].append(float(loss))
        logs["accuracy"].append(float(acc))
        logs["epoch_time"].append(duration)
        logs["cumulative_time"].append(cumulative_time)
        logs["samples_per_second"].append(FLAGS.batch_size / duration)
        
        if epoch % 10 == 0:
            tqdm.write(f"Epoch {epoch}: Loss {loss:.4f}, Acc {acc:.4f}, Speed {FLAGS.batch_size/duration:.1f} samples/s")

    # --- Generate Visualizations ---
    log_dir = "/home/sj/Desktop/zy/moqi_workspace/rl_deploy/classifier/training_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    df = pd.DataFrame(logs)
    df.to_csv(os.path.join(log_dir, "training_metrics.csv"), index=False)
    
    # helper for smoothing
    def smooth(y, window=5):
        if len(y) < window: return y
        return np.convolve(y, np.ones(window)/window, mode='valid')

    # 1. Data Distribution Pie Chart
    plt.figure(figsize=(8, 8))
    dist = [len(success_images), len(failure_images)]
    plt.pie(dist, labels=['Success', 'Failure'], autopct='%1.1f%%', colors=['tab:blue', 'tab:red'], startangle=140)
    plt.title('Training Data Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "data_distribution.png"))
    plt.close()

    # 2. Loss and Accuracy Curve（仅原始曲线，不做拟合/平滑）
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(df['epoch'], df['loss'], color='tab:red', linewidth=1.5, label='Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:blue')
    ax2.plot(df['epoch'], df['accuracy'], color='tab:blue', linewidth=1.5, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    plt.title('Classifier Training: Loss and Accuracy Trends')
    fig.tight_layout()
    plt.savefig(os.path.join(log_dir, "loss_accuracy_curves.png"))
    plt.close()
    
    # 3. Training Efficiency (Samples/s)
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['samples_per_second'], color='tab:green', alpha=0.4)
    speed_smooth = smooth(df['samples_per_second'].values)
    plt.plot(df['epoch'].values[len(df)-len(speed_smooth):], speed_smooth, color='tab:green', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Samples per Second')
    plt.title('Classifier Training Efficiency (Throughput)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "training_efficiency.png"))
    plt.close()
    
    # 4. Cumulative Time
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['cumulative_time'], color='tab:orange')
    plt.xlabel('Epoch')
    plt.ylabel('Cumulative Time (s)')
    plt.title('Classifier Training: Time Accumulation')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "cumulative_time.png"))
    plt.close()
    
    print(f"\nTraining logs and {len(logs['epoch'])} epochs of data visualized in: {log_dir}")

    if not os.path.exists(FLAGS.checkpoint_path):
        os.makedirs(FLAGS.checkpoint_path)
    
    abs_checkpoint_path = os.path.abspath(FLAGS.checkpoint_path)
    checkpoints.save_checkpoint(abs_checkpoint_path, classifier, step=FLAGS.num_epochs, overwrite=True)
    print(f"Classifier saved to {abs_checkpoint_path}")

if __name__ == "__main__":
    app.run(main)
