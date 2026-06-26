#!/usr/bin/env python3
"""Run a BC checkpoint directly on the real OpenArm environment."""
import compat  # noqa: F401  (sys.path + CUDA/JAX patches; must be first)

import csv
import os
import time
from pathlib import Path

import jax
import numpy as np
from absl import app, flags
from flax.training import checkpoints

from rl_launcher.agents import make_sac_pixel_agent_hybrid_dual_arm

from bc_utils import action_vector
from experiments.artifacts import task_bc_checkpoint_dir, task_bc_eval_dir
from experiments.mappings import CONFIG_MAPPING


FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", "openarm_pickplace", "Experiment name in CONFIG_MAPPING.")
flags.DEFINE_string("bc_checkpoint_path", None, "Defaults to task checkpoints_bc.")
flags.DEFINE_integer("checkpoint_step", 0, "Checkpoint step, 0 means latest.")
flags.DEFINE_integer("eval_n_trajs", 5, "Number of real rollouts.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("argmax", True, "Use deterministic action mode.")
flags.DEFINE_boolean("save_video", False, "Pass save_video to task env.")
flags.DEFINE_string("output_dir", None, "Defaults to task bc_eval/real.")


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
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(_):
    if FLAGS.exp_name not in CONFIG_MAPPING:
        raise ValueError(f"Experiment {FLAGS.exp_name!r} not found in CONFIG_MAPPING.")
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    checkpoint_path = os.path.abspath(
        FLAGS.bc_checkpoint_path or task_bc_checkpoint_dir(FLAGS.exp_name)
    )
    output_dir = Path(FLAGS.output_dir or (task_bc_eval_dir(FLAGS.exp_name) / "real"))
    output_dir.mkdir(parents=True, exist_ok=True)

    env = config.get_environment(fake_env=False, save_video=FLAGS.save_video, classifier=True)
    agent = restore_agent(create_agent(config, env), checkpoint_path)
    rng = jax.random.PRNGKey(FLAGS.seed)

    episode_rows = []
    step_rows = []
    for episode_idx in range(FLAGS.eval_n_trajs):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_return = 0.0
        step = 0
        action_norms = []
        gripper_changes = 0
        last_gripper = None
        start_time = time.time()

        while not done and not truncated:
            rng, key = jax.random.split(rng)
            action = agent.sample_actions(
                observations=jax.device_put(obs),
                seed=key,
                argmax=FLAGS.argmax,
            )
            action = np.asarray(jax.device_get(action))
            next_obs, reward, done, truncated, info = env.step(action)
            action_for_metrics = action_vector(action)
            action_norm = float(np.linalg.norm(action_for_metrics))
            action_norms.append(action_norm)
            gripper = tuple(np.rint(action_for_metrics[[6, 13]]).astype(int).tolist())
            if last_gripper is not None and gripper != last_gripper:
                gripper_changes += 1
            last_gripper = gripper
            step_rows.append(
                {
                    "episode": episode_idx,
                    "step": step,
                    "reward": float(np.asarray(reward)),
                    "done": int(done),
                    "truncated": int(truncated),
                    "action_norm": action_norm,
                    "gripper_left": gripper[0],
                    "gripper_right": gripper[1],
                }
            )
            episode_return += float(np.asarray(reward))
            obs = next_obs
            step += 1

        elapsed = time.time() - start_time
        episode_rows.append(
            {
                "episode": episode_idx,
                "return": episode_return,
                "success": int(episode_return > 0.0),
                "steps": step,
                "elapsed_sec": elapsed,
                "mean_action_norm": float(np.mean(action_norms)) if action_norms else 0.0,
                "max_action_norm": float(np.max(action_norms)) if action_norms else 0.0,
                "gripper_changes": gripper_changes,
                "truncated": int(truncated),
            }
        )
        print_green(
            f"[BC Real Eval] episode {episode_idx}: return={episode_return:.3f}, steps={step}"
        )

    write_csv(output_dir / "episodes.csv", episode_rows)
    write_csv(output_dir / "steps.csv", step_rows)
    success_rate = np.mean([row["success"] for row in episode_rows]) if episode_rows else 0.0
    print_green(f"[BC Real Eval] success_rate={success_rate:.3f}; results={output_dir}")
    env.close()


if __name__ == "__main__":
    app.run(main)
