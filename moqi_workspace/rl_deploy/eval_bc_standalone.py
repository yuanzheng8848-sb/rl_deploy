#!/usr/bin/env python3
"""
Standalone BC evaluation script - completely self-contained.
Does NOT import from train.py to avoid flag conflicts.
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
from franka_env.utils.transformations import construct_adjoint_matrix

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

# Flags (eval-specific, no conflicts)
FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "bc_eval_standalone", "Experiment name.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("checkpoint_path", "./checkpoints_bc_standalone", "BC checkpoint directory.")
flags.DEFINE_integer("checkpoint_step", 0, "Checkpoint step to evaluate (0=latest).")
flags.DEFINE_multi_string("demo_path", [], "Demo directories.")
flags.DEFINE_integer("eval_steps", 1000, "Number of evaluation batches.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
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
    if not os.path.exists(FLAGS.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path not found: {FLAGS.checkpoint_path}")

    print(f"\033[92m[BC Eval] checkpoint_path={FLAGS.checkpoint_path}\033[00m")
    print(f"\033[92m[BC Eval] demo_path={FLAGS.demo_path}\033[00m")

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

    # Load checkpoint
    if FLAGS.checkpoint_step > 0:
        ckpt = checkpoints.restore_checkpoint(FLAGS.checkpoint_path, agent.state, step=FLAGS.checkpoint_step)
        eval_step = FLAGS.checkpoint_step
    else:
        latest = checkpoints.latest_checkpoint(FLAGS.checkpoint_path)
        if not latest:
            raise FileNotFoundError(f"No checkpoint found in {FLAGS.checkpoint_path}")
        ckpt = checkpoints.restore_checkpoint(FLAGS.checkpoint_path, agent.state)
        eval_step = int(os.path.basename(latest)[11:])

    agent = agent.replace(state=ckpt)
    print(f"\033[92m[BC Eval] Loaded checkpoint at step {eval_step}\033[00m")

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
        # Remove obs_horizon dimension from demo data
        # Handle any obs_horizon value by taking the last frame
        obs = transition["observations"]
        next_obs = transition["next_observations"]
        for key in TRAINING_IMAGE_KEYS:
            if key in obs:
                # If stacked (obs_horizon > 1), take the last frame
                if obs[key].ndim == 4:
                    obs[key] = obs[key][-1]  # Take last frame: (T, H, W, C) -> (H, W, C)
            if key in next_obs:
                if next_obs[key].ndim == 4:
                    next_obs[key] = next_obs[key][-1]
        demo_buffer.insert(transition)
    print(f"\033[92m[BC Eval] Loaded {len(demo_buffer)} demo transitions\033[00m")

    if len(demo_buffer) == 0:
        raise ValueError("No demo data loaded.")

    # WandB
    wandb_logger = make_wandb_logger(
        project="hil-serl-bc-eval-standalone",
        description=f"{FLAGS.exp_name}_ckpt{eval_step}",
        debug=FLAGS.debug,
    )

    # Evaluation loop
    demo_iterator = demo_buffer.get_iterator(
        sample_args={"batch_size": FLAGS.batch_size, "pack_obs_and_next_obs": False},
        device=sharding,
    )

    bc_losses = []
    bc_mses = []
    action_l2_errors = []

    print(f"\033[92m[BC Eval] Evaluating for {FLAGS.eval_steps} steps...\033[00m")
    for eval_idx in tqdm.tqdm(range(FLAGS.eval_steps), desc="BC eval"):
        batch = next(demo_iterator)

        # With pack_obs_and_next_obs=False, observations are already (B, T, H, W, C) with T=1
        batch = next(demo_iterator)

        # Extract continuous actions (skip gripper indices 6 and 13)
        continuous_demo_actions = jnp.concatenate([
            batch["actions"][..., :6],
            batch["actions"][..., 7:13]
        ], axis=-1)

        # Forward pass
        rng = jax.random.PRNGKey(FLAGS.seed + eval_idx)
        action_dist = agent.forward_policy(batch["observations"], rng=rng, train=False)
        predicted_actions = action_dist.mode()

        # Metrics
        target_actions = jnp.clip(continuous_demo_actions, -1.0 + 1e-6, 1.0 - 1e-6)
        log_probs = action_dist.log_prob(target_actions)
        bc_loss = float(-log_probs.mean())
        bc_mse = float(((predicted_actions - target_actions) ** 2).sum(-1).mean())
        action_l2 = float(jnp.linalg.norm(predicted_actions - target_actions, axis=-1).mean())

        bc_losses.append(bc_loss)
        bc_mses.append(bc_mse)
        action_l2_errors.append(action_l2)

        if wandb_logger and eval_idx % 10 == 0:
            wandb_logger.log(
                {
                    "eval/bc_loss": bc_loss,
                    "eval/bc_mse": bc_mse,
                    "eval/action_l2_error": action_l2,
                },
                step=eval_idx,
            )

    # Summary
    mean_bc_loss = float(np.mean(bc_losses))
    mean_bc_mse = float(np.mean(bc_mses))
    mean_action_l2 = float(np.mean(action_l2_errors))
    std_bc_loss = float(np.std(bc_losses))
    std_bc_mse = float(np.std(bc_mses))
    std_action_l2 = float(np.std(action_l2_errors))

    print(f"\033[92m\n[BC Eval] Results for checkpoint step {eval_step}:\033[00m")
    print(f"  BC Loss:    {mean_bc_loss:.4f} ± {std_bc_loss:.4f}")
    print(f"  BC MSE:     {mean_bc_mse:.4f} ± {std_bc_mse:.4f}")
    print(f"  Action L2:  {mean_action_l2:.4f} ± {std_action_l2:.4f}")

    if wandb_logger:
        wandb_logger.log(
            {
                "eval_summary/mean_bc_loss": mean_bc_loss,
                "eval_summary/mean_bc_mse": mean_bc_mse,
                "eval_summary/mean_action_l2_error": mean_action_l2,
                "eval_summary/std_bc_loss": std_bc_loss,
                "eval_summary/std_bc_mse": std_bc_mse,
                "eval_summary/std_action_l2_error": std_action_l2,
                "eval_summary/checkpoint_step": eval_step,
            },
            step=FLAGS.eval_steps,
        )

    print("\033[92m[BC Eval] Complete.\033[00m")


if __name__ == "__main__":
    app.run(main)
