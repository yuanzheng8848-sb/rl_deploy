#!/usr/bin/env python3
"""Run a BC checkpoint directly on the real OpenArm environment."""
import compat  # noqa: F401  (sys.path + CUDA/JAX patches; must be first)

import csv
import os
import time
from pathlib import Path

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from flax.training import checkpoints
try:
    from pynput import keyboard
except Exception:  # pragma: no cover - optional desktop dependency
    keyboard = None

from rl_launcher.agents import make_sac_pixel_agent_hybrid_dual_arm
from rl_launcher.networks import load_classifier_func

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
flags.DEFINE_boolean("render", True, "Render camera images with classifier prob and reward.")
flags.DEFINE_float("render_fps", 20.0, "Render refresh cap.")


def print_green(text):
    print(f"\033[92m{text}\033[00m")


def print_yellow(text):
    print(f"\033[93m{text}\033[00m")


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


def make_keyboard_state():
    return {"label": None, "quit_requested": False}


def start_keyboard_listener(state):
    if keyboard is None:
        print_yellow("[BC Real Eval] pynput unavailable; keyboard labels require the OpenCV window focus.")
        return None

    def on_press(key):
        try:
            if key == keyboard.Key.enter:
                state["label"] = "success"
                print_green("[BC Real Eval] ENTER -> success requested")
            elif key == keyboard.Key.space:
                state["label"] = "failure"
                print_yellow("[BC Real Eval] SPACE -> failure requested")
            elif key == keyboard.Key.esc:
                state["quit_requested"] = True
                print_yellow("[BC Real Eval] ESC -> quit requested")
        except Exception:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


def consume_keyboard_label(state):
    if state.get("quit_requested"):
        state["quit_requested"] = False
        return "quit"
    label = state.get("label")
    if label:
        state["label"] = None
        return label
    return None


def image_from_obs(obs, key):
    img = obs.get(key) if isinstance(obs, dict) else None
    if img is None:
        return None
    img = np.asarray(img)
    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]
    if img.ndim != 3 or img.shape[-1] != 3:
        return None
    return img.astype(np.uint8)


def make_classifier_prob(config, env):
    classifier_keys = list(getattr(config, "classifier_keys", []) or [])
    checkpoint_path = getattr(config, "classifier_ckpt_path", None)
    if not classifier_keys or not checkpoint_path:
        return None, 0.0
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        return None, 0.0
    classifier_logits = load_classifier_func(
        key=jax.random.PRNGKey(0),
        sample=env.observation_space.sample(),
        image_keys=classifier_keys,
        checkpoint_path=checkpoint_path,
    )
    threshold = float(getattr(config, "classifier_threshold", 0.5))

    def classifier_prob(obs):
        logits = classifier_logits(obs)
        prob = jax.nn.sigmoid(jnp.squeeze(logits))
        return float(np.asarray(jax.device_get(prob)))

    return classifier_prob, threshold


def render_feedback(obs, image_keys, prob, threshold, reward, done, info):
    if not FLAGS.render:
        return None
    panels = []
    for key in image_keys:
        img = image_from_obs(obs, key)
        if img is None:
            panel = np.zeros((320, 320, 3), dtype=np.uint8)
        else:
            panel = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (320, 320))
        color = (0, 220, 0) if float(np.asarray(reward)) > 0 else (0, 0, 255)
        cv2.putText(panel, key, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, f"prob={prob:.3f}/{threshold:.2f}", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, f"reward={float(np.asarray(reward)):.1f}", (12, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        cv2.putText(panel, f"done={bool(done)} succeed={bool(info.get('succeed', False))}", (12, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(panel, "ENTER=success  SPACE=failure  q=quit", (12, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 1, cv2.LINE_AA)
        panels.append(panel)
    if panels:
        cv2.imshow("rl-serl BC real eval", np.concatenate(panels, axis=1))
    delay = max(1, int(1000 / max(float(FLAGS.render_fps), 1.0)))
    key = cv2.waitKey(delay) & 0xFF
    if key in (10, 13):
        return "success"
    if key == ord(" "):
        return "failure"
    if key in (27, ord("q")):
        return "quit"
    return None


def main(_):
    if FLAGS.exp_name not in CONFIG_MAPPING:
        raise ValueError(f"Experiment {FLAGS.exp_name!r} not found in CONFIG_MAPPING.")
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    checkpoint_path = os.path.abspath(
        FLAGS.bc_checkpoint_path or task_bc_checkpoint_dir(FLAGS.exp_name)
    )
    output_dir = Path(FLAGS.output_dir or (task_bc_eval_dir(FLAGS.exp_name) / "real"))
    output_dir.mkdir(parents=True, exist_ok=True)

    env = config.get_environment(fake_env=False, save_video=FLAGS.save_video, classifier=False)
    agent = restore_agent(create_agent(config, env), checkpoint_path)
    classifier_prob_fn, classifier_threshold = make_classifier_prob(config, env)
    keyboard_state = make_keyboard_state()
    keyboard_listener = start_keyboard_listener(keyboard_state)
    rng = jax.random.PRNGKey(FLAGS.seed)

    episode_rows = []
    step_rows = []
    quit_requested = False
    try:
        for episode_idx in range(FLAGS.eval_n_trajs):
            if quit_requested:
                break
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
                info = info if isinstance(info, dict) else {}
                prob = classifier_prob_fn(next_obs) if classifier_prob_fn is not None else 0.0
                cv2_label = render_feedback(
                    next_obs,
                    config.image_keys,
                    prob,
                    classifier_threshold,
                    reward,
                    bool(done or truncated),
                    info,
                )
                key_label = consume_keyboard_label(keyboard_state) or cv2_label
                if key_label == "success":
                    reward = np.asarray(1.0, dtype=np.float32)
                    done = True
                    truncated = False
                    info["keyboard_label"] = "success"
                    info["succeed"] = True
                    print_green("[BC Real Eval] success reward=1, ending episode.")
                elif key_label == "failure":
                    reward = np.asarray(0.0, dtype=np.float32)
                    done = True
                    truncated = False
                    info["keyboard_label"] = "failure"
                    info["succeed"] = False
                    print_yellow("[BC Real Eval] failure reward=0, ending episode.")
                elif key_label == "quit":
                    done = True
                    truncated = True
                    quit_requested = True
                    info["keyboard_label"] = "quit"
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
                    "classifier_prob": prob,
                    "classifier_threshold": classifier_threshold,
                    "classifier_reference_success": int(prob > classifier_threshold),
                    "keyboard_label": info.get("keyboard_label", ""),
                    "succeed": int(bool(info.get("succeed", False))),
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
    finally:
        if keyboard_listener is not None:
            keyboard_listener.stop()
        env.close()
        if FLAGS.render:
            cv2.destroyAllWindows()

    write_csv(output_dir / "episodes.csv", episode_rows)
    write_csv(output_dir / "steps.csv", step_rows)
    success_rate = np.mean([row["success"] for row in episode_rows]) if episode_rows else 0.0
    print_green(f"[BC Real Eval] success_rate={success_rate:.3f}; results={output_dir}")


if __name__ == "__main__":
    app.run(main)
