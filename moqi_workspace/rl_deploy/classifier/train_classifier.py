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

from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.networks.reward_classifier import create_classifier
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

import gym
from gym import spaces

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_path", "../classifier_ckpt_cam1", "Path to save checkpoint")
flags.DEFINE_integer("batch_size", 64, "Batch size")
flags.DEFINE_integer("num_epochs", 100, "Number of epochs")

RECORD_DATA_DIR = "/home/sj/Desktop/zy/moqi_workspace/record_data"


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
    success_images = []
    failure_images = []
    
    if not os.path.exists(RECORD_DATA_DIR):
        raise FileNotFoundError(f"Record data dir not found: {RECORD_DATA_DIR}")

    sessions = sorted([d for d in os.listdir(RECORD_DATA_DIR) if os.path.isdir(os.path.join(RECORD_DATA_DIR, d))])
    
    print(f"Found {len(sessions)} sessions in {RECORD_DATA_DIR}")

    for session in sessions:
        if session in ["success", "failure"]:
            continue
            
        img_dir = os.path.join(RECORD_DATA_DIR, session, "images", "cam_1_rgb")
        if not os.path.exists(img_dir):
            continue
            
        # Manually glob and sort to ensure order
        image_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg"))) 
        # Add other extensions if needed, but ensure sorting is correct across them if mixed
        # For now assuming primarily jpg based on file listing check

        current_session_images = []
        for path in image_paths:
            try:
                img = cv2.imread(path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (128, 128))
                img = img[None, ...] 
                current_session_images.append(img)
            except Exception as e:
                print(f"Error loading {path}: {e}")

        if len(current_session_images) < 20:
             print(f"Warning: Session {session} has fewer than 20 images ({len(current_session_images)}). Skipping.")
             continue
        
        # First 30 frames -> failure
        failure_images.extend(current_session_images[:30])
        # Last 30 frames -> success
        success_images.extend(current_session_images[-30:])
        
    print(f"Total Success Images: {len(success_images)}")
    print(f"Total Failure Images: {len(failure_images)}")

    if not success_images:
        raise ValueError("No success images found.")
    if not failure_images:
        raise ValueError("No failure images found.")

    populate_data_store_from_images(pos_buffer, success_images)
    populate_data_store_from_images(neg_buffer, failure_images)

    if len(pos_buffer) == 0 or len(neg_buffer) == 0:
        raise ValueError("Buffers cannot be empty.")

    devices = jax.local_devices()
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
    for epoch in tqdm(range(FLAGS.num_epochs)):
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
        
        if epoch % 10 == 0:
            tqdm.write(f"Epoch {epoch}: Loss {loss:.4f}, Acc {acc:.4f}")

    if not os.path.exists(FLAGS.checkpoint_path):
        os.makedirs(FLAGS.checkpoint_path)
    
    abs_checkpoint_path = os.path.abspath(FLAGS.checkpoint_path)
    checkpoints.save_checkpoint(abs_checkpoint_path, classifier, step=FLAGS.num_epochs, overwrite=True)
    print(f"Classifier saved to {abs_checkpoint_path}")

if __name__ == "__main__":
    app.run(main)
