#!/usr/bin/env python3
"""
Standalone BC training script - does NOT import from train.py
All necessary code is self-contained to avoid flag conflicts.
"""

import os
import sys
import pickle as pkl
import glob
from pathlib import Path
from collections import deque

import cv2
import gymnasium as gym
import numpy as np
import tqdm
import jax
import jax.numpy as jnp
from absl import app, flags
from flax.training import checkpoints
from scipy.spatial.transform import Rotation as R

# Configure JAX CUDA paths
import site
nvidia_base = os.path.join(site.getsitepackages()[0], "nvidia")
for lib in ("cublas/lib", "cudnn/lib", "cufft/lib", "cusolver/lib", "cusparse/lib", "nccl/lib", "nvjitlink/lib"):
    path = os.path.join(nvidia_base, lib)
    if os.path.exists(path):
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{path}:{current_ld}"
os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={nvidia_base}"

import jax
import jax.numpy as jnp

# Compatibility shims for newer JAX releases
if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree_util.tree_map
if not hasattr(jax, "tree_leaves"):
    jax.tree_leaves = jax.tree_util.tree_leaves

_orig_shaped_array_update = jax.core.ShapedArray.update

def _compat_shaped_array_update(self, *args, **kwargs):
    # `named_shape` was removed from newer JAX ShapedArray constructors
    kwargs.pop("named_shape", None)
    return _orig_shaped_array_update(self, *args, **kwargs)

jax.core.ShapedArray.update = _compat_shaped_array_update

# Add paths
ROOT_DIR = Path(__file__).resolve().parents[2]
PYROKI_DIR = Path(__file__).resolve().parent.parent / "pyroki"
HIL_SERL_LAUNCHER_DIR = ROOT_DIR / "hil-serl" / "serl_launcher"
if str(HIL_SERL_LAUNCHER_DIR) not in sys.path:
    sys.path.insert(0, str(HIL_SERL_LAUNCHER_DIR))
if str(PYROKI_DIR) not in sys.path:
    sys.path.insert(0, str(PYROKI_DIR))

# Import after path setup
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from serl_launcher.utils.launcher import make_sac_pixel_agent_hybrid_dual_arm, make_wandb_logger
from openarm_env import DefaultOpenArmConfig, OpenArmEnv
from franka_env.utils.transformations import construct_adjoint_matrix, construct_homogeneous_matrix

# Constants
MODEL_IMAGE_SIZE = (128, 128)
TRAINING_IMAGE_KEYS = ["image_primary", "image_left", "image_right"]
PROPRIO_KEYS = ["tcp_pose", "tcp_vel", "gripper_pose"]

# JAX setup
devices = jax.local_devices()
if hasattr(jax.sharding, "PositionalSharding"):
    sharding = jax.sharding.PositionalSharding(devices).replicate()
else:
    sharding = jax.sharding.SingleDeviceSharding(devices[0])

# Flags
FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "bc_standalone", "Experiment name.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("checkpoint_path", "./checkpoints_bc_standalone", "BC checkpoint directory.")
flags.DEFINE_multi_string("demo_path", [], "Demo directories to load.")
flags.DEFINE_integer("bc_steps", 5000, "Number of BC training steps.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("log_period", 10, "Logging period.")
flags.DEFINE_integer("checkpoint_period", 500, "Checkpoint save period.")
flags.DEFINE_boolean("debug", False, "Disable wandb.")
flags.DEFINE_string("encoder_type", "resnet-pretrained", "Encoder type.")
flags.DEFINE_float("discount", 0.97, "Discount factor.")


class TrainOpenArmConfig(DefaultOpenArmConfig):
    REALSENSE_CAMERAS = {
        "image_primary": "local-head",
        "image_left": "local-left",
        "image_right": "local-right",
    }


def stack_obs(obs):
    dict_list = {key: [item[key] for item in obs] for key in obs[0]}
    return jax.tree_util.tree_map(
        lambda values: np.stack(values),
        dict_list,
        is_leaf=lambda value: isinstance(value, list),
    )


def space_stack(space, repeat):
    if isinstance(space, gym.spaces.Box):
        return gym.spaces.Box(
            low=np.repeat(space.low[None], repeat, axis=0),
            high=np.repeat(space.high[None], repeat, axis=0),
            dtype=space.dtype,
        )
    if isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict(
            {key: space_stack(value, repeat) for key, value in space.spaces.items()}
        )
    raise TypeError(f"Unsupported space type: {type(space)}")


class ChunkingWrapper(gym.Wrapper):
    def __init__(self, env, obs_horizon=1):
        super().__init__(env)
        self.obs_horizon = int(obs_horizon)
        self.current_obs = deque(maxlen=self.obs_horizon)
        self.observation_space = space_stack(self.env.observation_space, self.obs_horizon)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.current_obs.append(obs)
        return stack_obs(self.current_obs), reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_obs.extend([obs] * self.obs_horizon)
        return stack_obs(self.current_obs), info


class DualRelativeFrame(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.left_transform = np.eye(6)
        self.right_transform = np.eye(6)

    def transform_observation(self, obs):
        obs = {
            "images": obs["images"],
            "state": {key: np.array(value, copy=True) for key, value in obs["state"].items()},
        }
        tcp_vel = obs["state"]["tcp_vel"]
        tcp_vel[0] = np.linalg.inv(self.left_transform) @ tcp_vel[0]
        tcp_vel[1] = np.linalg.inv(self.right_transform) @ tcp_vel[1]
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        tcp_pose = np.asarray(obs["state"]["tcp_pose"])
        self.left_transform = construct_adjoint_matrix(tcp_pose[0])
        self.right_transform = construct_adjoint_matrix(tcp_pose[1])
        return self.transform_observation(obs), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        tcp_pose = np.asarray(obs["state"]["tcp_pose"])
        self.left_transform = construct_adjoint_matrix(tcp_pose[0])
        self.right_transform = construct_adjoint_matrix(tcp_pose[1])
        return self.transform_observation(obs), reward, done, truncated, info


class Quat2EulerWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        state_space = self.observation_space["state"]
        state_space.spaces["tcp_pose"] = gym.spaces.Box(-np.inf, np.inf, shape=(2, 6), dtype=np.float32)

    def observation(self, obs):
        obs = {
            "images": obs["images"],
            "state": {key: np.array(value, copy=True) for key, value in obs["state"].items()},
        }
        tcp_pose = obs["state"]["tcp_pose"]
        euler_pose = np.zeros((2, 6), dtype=np.float32)
        for arm_idx in range(2):
            euler_pose[arm_idx, :3] = tcp_pose[arm_idx, :3]
            euler_pose[arm_idx, 3:] = R.from_quat(tcp_pose[arm_idx, 3:]).as_euler("xyz")
        obs["state"]["tcp_pose"] = euler_pose
        return obs


class NetworkPrimaryImageCropWrapper(gym.ObservationWrapper):
    def __init__(self, env, crop_ratio=0.3, y_offset_ratio=0.0):
        super().__init__(env)
        self.crop_ratio = float(crop_ratio)
        self.y_offset_ratio = float(y_offset_ratio)

    def observation(self, obs):
        if "images" not in obs or "image_primary" not in obs["images"]:
            return obs
        obs = {
            "images": {key: np.array(value, copy=True) for key, value in obs["images"].items()},
            "state": obs["state"],
        }
        img = obs["images"]["image_primary"]
        if img.ndim == 4:  # Stacked
            cropped = np.stack([self._crop_single(frame) for frame in img])
            obs["images"]["image_primary"] = cropped
        else:
            obs["images"]["image_primary"] = self._crop_single(img)
        return obs

    def _crop_single(self, img):
        h, w = img.shape[:2]
        crop_h = max(1, int(round(h * self.crop_ratio)))
        crop_w = max(1, int(round(w * self.crop_ratio)))
        center_y = (h / 2.0) + (h * self.y_offset_ratio)
        y0 = int(np.clip(center_y - crop_h / 2.0, 0, max(0, h - crop_h)))
        x0 = max(0, (w - crop_w) // 2)
        cropped = img[y0 : y0 + crop_h, x0 : x0 + crop_w]
        return cv2.resize(cropped, MODEL_IMAGE_SIZE).astype(np.uint8)


class SERLObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, proprio_keys):
        super().__init__(env)
        self.proprio_keys = list(proprio_keys)
        self.proprio_space = gym.spaces.Dict(
            {key: self.env.observation_space["state"][key] for key in self.proprio_keys}
        )
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.flatten_space(self.proprio_space),
                **self.env.observation_space["images"].spaces,
            }
        )

    def observation(self, obs):
        proprio = {key: obs["state"][key] for key in self.proprio_keys}
        return {
            "state": gym.spaces.flatten(self.proprio_space, proprio),
            **obs["images"],
        }


class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=0.0):
        super().__init__(env)
        self.penalty = float(penalty)

    def step(self, action):
        return self.env.step(action)


def create_bc_env():
    """Create minimal env for BC agent creation."""
    env = OpenArmEnv(
        fake_env=True,
        save_video=False,
        hz=10,
        config=TrainOpenArmConfig(),
        max_episode_length=400,
    )
    env = DualRelativeFrame(env)
    env = Quat2EulerWrapper(env)
    env = NetworkPrimaryImageCropWrapper(env, crop_ratio=0.3, y_offset_ratio=0.0)
    env = SERLObsWrapper(env, proprio_keys=PROPRIO_KEYS)
    env = ChunkingWrapper(env, obs_horizon=1)
    env = GripperPenaltyWrapper(env, penalty=0.0)
    from gymnasium.wrappers import RecordEpisodeStatistics
    return RecordEpisodeStatistics(env)


def load_demo_files(paths):
    """Load all demo transitions from pkl files."""
    transitions = []
    for path in paths:
        if not path:
            continue
        norm_path = os.path.abspath(path)
        if os.path.isdir(norm_path):
            for file_path in sorted(glob.glob(os.path.join(norm_path, "*.pkl"))):
                with open(file_path, "rb") as f:
                    transitions.extend(pkl.load(f))
        elif os.path.isfile(norm_path) and norm_path.endswith('.pkl'):
            with open(norm_path, "rb") as f:
                transitions.extend(pkl.load(f))
    return transitions


def main(_):
    if not FLAGS.demo_path:
        raise ValueError("--demo_path is required.")

    FLAGS.checkpoint_path = os.path.abspath(FLAGS.checkpoint_path)
    os.makedirs(FLAGS.checkpoint_path, exist_ok=True)
    print(f"\033[92m[BC] checkpoint_path={FLAGS.checkpoint_path}\033[00m")
    print(f"\033[92m[BC] demo_path={FLAGS.demo_path}\033[00m")

    # Create env and agent
    env = create_bc_env()
    agent = make_sac_pixel_agent_hybrid_dual_arm(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=TRAINING_IMAGE_KEYS,
        encoder_type=FLAGS.encoder_type,
        discount=FLAGS.discount,
    )

    # Load checkpoint if exists
    start_step = 0
    if os.path.exists(FLAGS.checkpoint_path):
        latest = checkpoints.latest_checkpoint(FLAGS.checkpoint_path)
        if latest:
            ckpt = checkpoints.restore_checkpoint(FLAGS.checkpoint_path, agent.state)
            agent = agent.replace(state=ckpt)
            start_step = int(os.path.basename(latest)[11:]) + 1
            print(f"\033[92m[BC] Resumed from step {start_step}\033[00m")

    # Load demos
    demo_buffer = MemoryEfficientReplayBufferDataStore(
        env.observation_space,
        env.action_space,
        capacity=50000,
        image_keys=TRAINING_IMAGE_KEYS,
        include_grasp_penalty=True,
    )
    demo_transitions = load_demo_files(FLAGS.demo_path)
    for transition in demo_transitions:
        demo_buffer.insert(transition)
    print(f"\033[92m[BC] Loaded {len(demo_buffer)} demo transitions\033[00m")

    if len(demo_buffer) == 0:
        raise ValueError("No demo data loaded. Check --demo_path.")

    # WandB
    wandb_logger = make_wandb_logger(
        project="hil-serl-bc-standalone",
        description=FLAGS.exp_name,
        debug=FLAGS.debug,
    )

    # Training loop
    demo_iterator = demo_buffer.get_iterator(
        sample_args={"batch_size": FLAGS.batch_size, "pack_obs_and_next_obs": True},
        device=sharding,
    )

    print(f"\033[92m[BC] Training from step {start_step} to {FLAGS.bc_steps}\033[00m")
    for step in tqdm.tqdm(range(start_step, FLAGS.bc_steps), desc="BC training"):
        batch = next(demo_iterator)
        agent, update_info = agent.update(
            batch,
            networks_to_update=frozenset({"actor", "critic", "grasp_critic"}),
            bc_mode=True,
        )

        if wandb_logger and step % FLAGS.log_period == 0:
            wandb_logger.log(update_info, step=step)

        if step > 0 and step % FLAGS.checkpoint_period == 0:
            agent = jax.block_until_ready(agent)
            checkpoints.save_checkpoint(FLAGS.checkpoint_path, agent.state, step=step, keep=5, overwrite=False)
            print(f"\033[92m[BC] Saved checkpoint at step {step}\033[00m")

    # Final checkpoint
    agent = jax.block_until_ready(agent)
    checkpoints.save_checkpoint(FLAGS.checkpoint_path, agent.state, step=FLAGS.bc_steps, keep=5, overwrite=False)
    print(f"\033[92m[BC] Training complete. Final checkpoint saved.\033[00m")


if __name__ == "__main__":
    app.run(main)
