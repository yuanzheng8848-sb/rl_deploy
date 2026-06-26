#!/usr/bin/env python3
"""RLPD training entrypoint for OpenArm (rl-serl).

Migrated and slimmed from rl_deploy/train.py. Keeps the actor / learner / eval
loops and the agentlace client/server wiring, but drops the removed features:
  - Arm-Focus scaffolding
  - KeyboardRewardWrapper (reward now comes from the vision classifier)
  - Handoff-Focus tagging + demo-buffer repeat insertion
  - Episode diagnostic metrics (joint-distance / monotonicity / autonomy / ...)

Environment assembly lives in experiments/<exp_name>/config.py via
CONFIG_MAPPING; this file no longer hard-codes any task/robot logic.

Usage:
  python train_rlpd.py --exp_name openarm_pickplace --learner
  python train_rlpd.py --exp_name openarm_pickplace --actor
  python train_rlpd.py --exp_name openarm_pickplace --eval --eval_n_trajs 10
"""
import compat  # noqa: F401  (sets sys.path + CUDA/JAX patches; must be first)

import copy
import glob
import os
import pickle as pkl
import time
from collections import deque

import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints

import jax
import jax.numpy as jnp

from rl_launcher.agents import (
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
    make_wandb_logger,
)
from rl_launcher.data import MemoryEfficientReplayBufferDataStore, QueuedDataStore
from rl_launcher.utils import Timer, concat_batches, TrainerClient, TrainerServer

from experiments.artifacts import task_bc_checkpoint_dir, task_rlpd_checkpoint_dir, task_success_dir
from experiments.mappings import CONFIG_MAPPING


FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", "openarm_pickplace", "Experiment name in CONFIG_MAPPING.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Run learner loop.")
flags.DEFINE_boolean("actor", False, "Run actor loop.")
flags.DEFINE_boolean("eval", False, "Run evaluation loop.")
flags.DEFINE_boolean("debug", False, "Disable wandb when true.")
flags.DEFINE_boolean("mock", False, "Use fake env / mock hardware observations.")
flags.DEFINE_boolean("save_video", False, "Compatibility flag for env creation.")
flags.DEFINE_string("ip", "localhost", "Learner IP.")
flags.DEFINE_string(
    "checkpoint_path",
    None,
    "Checkpoint directory. Defaults to experiments/<exp_name>/checkpoints_rlpd.",
)
flags.DEFINE_string(
    "bc_checkpoint_path",
    None,
    "BC checkpoint directory. Defaults to experiments/<exp_name>/checkpoints_bc.",
)
flags.DEFINE_integer("bc_checkpoint_step", 0, "BC checkpoint step, 0 = latest.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Checkpoint step for eval, 0 = latest.")
flags.DEFINE_integer("eval_n_trajs", 5, "Number of evaluation trajectories.")

# Optional overrides (default None -> use TrainConfig value).
flags.DEFINE_integer("max_steps", None, "Override max training steps.")
flags.DEFINE_integer("batch_size", None, "Override learner batch size.")

devices = jax.local_devices()
num_devices = len(devices)
if hasattr(jax.sharding, "PositionalSharding"):
    sharding = jax.sharding.PositionalSharding(devices).replicate()
else:
    sharding = jax.sharding.SingleDeviceSharding(devices[0])


def print_green(text):
    print(f"\033[92m {text}\033[00m")


def print_yellow(text):
    print(f"\033[93m {text}\033[00m")


def _cfg_value(config, name, override):
    return override if override is not None else getattr(config, name)


def resolve_checkpoint_path(exp_name):
    return os.path.abspath(FLAGS.checkpoint_path or task_rlpd_checkpoint_dir(exp_name))


def resolve_bc_checkpoint_path(exp_name):
    return os.path.abspath(FLAGS.bc_checkpoint_path or task_bc_checkpoint_dir(exp_name))


# ---------------------------------------------------------------------------
# agent + buffer helpers
# ---------------------------------------------------------------------------
def create_agent(config, env):
    agent = make_sac_pixel_agent_hybrid_dual_arm(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
        discount=config.discount,
    )
    return jax.device_put(jax.tree_util.tree_map(jnp.array, agent), sharding)


def restore_rlpd_or_bc(agent, checkpoint_path, bc_checkpoint_path):
    latest_rlpd = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        latest_rlpd = checkpoints.latest_checkpoint(os.path.abspath(checkpoint_path))
    if latest_rlpd:
        ckpt = checkpoints.restore_checkpoint(os.path.abspath(checkpoint_path), agent.state)
        print_green(f"Loaded RLPD checkpoint: {latest_rlpd}")
        return agent.replace(state=ckpt)

    latest_bc = None
    if bc_checkpoint_path and os.path.exists(bc_checkpoint_path):
        latest_bc = checkpoints.latest_checkpoint(os.path.abspath(bc_checkpoint_path))
    if latest_bc:
        if FLAGS.bc_checkpoint_step:
            ckpt = checkpoints.restore_checkpoint(
                os.path.abspath(bc_checkpoint_path),
                agent.state,
                step=FLAGS.bc_checkpoint_step,
            )
            print_green(
                f"Loaded BC checkpoint step {FLAGS.bc_checkpoint_step}: {bc_checkpoint_path}"
            )
        else:
            ckpt = checkpoints.restore_checkpoint(os.path.abspath(bc_checkpoint_path), agent.state)
            print_green(f"Loaded BC checkpoint: {latest_bc}")
        return agent.replace(state=ckpt)

    print_yellow("No RLPD or BC checkpoint found; using default initialization.")
    return agent


def load_transition_dir(dir_path, data_store):
    loaded_files = 0
    loaded_transitions = 0
    if not dir_path or not os.path.exists(dir_path):
        return loaded_files, loaded_transitions
    for path in sorted(glob.glob(os.path.join(dir_path, "*.pkl"))):
        with open(path, "rb") as handle:
            transitions = pkl.load(handle)
        loaded_files += 1
        for transition in transitions:
            transition.setdefault("grasp_penalty", np.asarray(0.0, dtype=np.float32))
            data_store.insert(transition)
            loaded_transitions += 1
    return loaded_files, loaded_transitions


def load_transition_files(paths, data_store):
    loaded_files = 0
    loaded_transitions = 0
    if not paths:
        return loaded_files, loaded_transitions
    seen = set()
    for path in paths:
        if path is None:
            continue
        norm = os.path.abspath(os.fspath(path))
        if norm in seen:
            continue
        seen.add(norm)
        if os.path.isdir(norm):
            files, transitions = load_transition_dir(norm, data_store)
            loaded_files += files
            loaded_transitions += transitions
            continue
        if not os.path.isfile(norm):
            continue
        with open(norm, "rb") as handle:
            transitions = pkl.load(handle)
        loaded_files += 1
        for transition in transitions:
            transition.setdefault("grasp_penalty", np.asarray(0.0, dtype=np.float32))
            data_store.insert(transition)
            loaded_transitions += 1
    return loaded_files, loaded_transitions


def save_transition_dump(base_dir, subdir, step, transitions):
    if not transitions:
        return
    target_dir = os.path.join(base_dir, subdir)
    os.makedirs(target_dir, exist_ok=True)
    with open(os.path.join(target_dir, f"transitions_{step}.pkl"), "wb") as handle:
        pkl.dump(transitions, handle)


def find_wrapper(env, cls):
    cur = env
    for _ in range(32):
        if isinstance(cur, cls):
            return cur
        if not hasattr(cur, "env"):
            break
        cur = cur.env
    return None


def is_intervention_mode_active(env):
    from openarm_env.envs.wrappers import DualSpacemouseIntervention

    wrapper = find_wrapper(env, DualSpacemouseIntervention)
    return bool(wrapper is not None and wrapper._intervention_mode)


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------
def evaluate(agent, env, sampling_rng, checkpoint_path):
    if checkpoint_path and os.path.exists(checkpoint_path):
        if FLAGS.eval_checkpoint_step:
            ckpt = checkpoints.restore_checkpoint(
                checkpoint_path,
                agent.state,
                step=FLAGS.eval_checkpoint_step,
            )
        else:
            ckpt = checkpoints.restore_checkpoint(
                checkpoint_path,
                agent.state,
            )
        agent = agent.replace(state=ckpt)

    successes = 0
    times = []
    for episode in range(FLAGS.eval_n_trajs):
        obs, _ = env.reset()
        done = False
        start_time = time.time()
        while not done:
            sampling_rng, key = jax.random.split(sampling_rng)
            actions = agent.sample_actions(
                observations=jax.device_put(obs), seed=key, argmax=False
            )
            actions = np.asarray(jax.device_get(actions))
            obs, reward, done, truncated, info = env.step(actions)
            done = bool(done or truncated)
        if reward:
            successes += 1
            times.append(time.time() - start_time)
        print(f"[Eval] episode={episode} reward={float(np.asarray(reward))}")

    print(f"success rate: {successes / max(FLAGS.eval_n_trajs, 1):.3f}")
    if times:
        print(f"average success time: {np.mean(times):.3f}s")


# ---------------------------------------------------------------------------
# actor
# ---------------------------------------------------------------------------
def actor(config, agent, data_store, intvn_data_store, env, sampling_rng, checkpoint_path):
    if FLAGS.eval:
        evaluate(agent, env, sampling_rng, checkpoint_path)
        return

    max_steps = _cfg_value(config, "max_steps", FLAGS.max_steps)
    start_step = 0
    buffer_dir = os.path.join(checkpoint_path, "buffer") if checkpoint_path else None
    if buffer_dir and os.path.exists(buffer_dir):
        existing = sorted(glob.glob(os.path.join(buffer_dir, "transitions_*.pkl")))
        if existing:
            start_step = int(os.path.basename(existing[-1])[12:-4]) + 1

    datastore_dict = {"actor_env": data_store, "actor_env_intvn": intvn_data_store}
    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=3000,
    )

    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    obs, _ = env.reset()
    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0
    last_timer_stats_step = None
    transitions = []
    demo_transitions = []

    pbar = tqdm.tqdm(
        total=max(max_steps - start_step, 0), initial=0, dynamic_ncols=True, desc="actor"
    )
    step = start_step
    while step < max_steps:
        timer.tick("total")

        with timer.context("sample_actions"):
            if is_intervention_mode_active(env):
                actions = np.zeros(env.action_space.shape, dtype=np.float32)
            elif step < config.random_steps:
                actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs), seed=key, argmax=False
                )
                actions = np.asarray(jax.device_get(actions))

        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(actions)
            sampled_transition = (
                info.get("sampled_transition") if isinstance(info, dict) else None
            )

            if isinstance(info, dict) and info.get("intervention_idle", False):
                timer.tock("total")
                if step % config.log_period == 0 and last_timer_stats_step != step:
                    client.request("send-stats", {"timer": timer.get_average_times(reset=False)})
                    last_timer_stats_step = step
                continue

            intervened = "intervene_action" in info
            if intervened:
                actions = np.asarray(info["intervene_action"], dtype=np.float32)
                intervention_steps += 1
                if not already_intervened:
                    intervention_count += 1
                already_intervened = True
            else:
                already_intervened = False

            if sampled_transition is not None:
                transition = {
                    "observations": sampled_transition["observations"],
                    "actions": np.asarray(sampled_transition["actions"], dtype=np.float32),
                    "next_observations": sampled_transition["next_observations"],
                    "rewards": np.asarray(sampled_transition["rewards"], dtype=np.float32),
                    "masks": np.asarray(1.0 - float(sampled_transition["dones"]), dtype=np.float32),
                    "dones": bool(sampled_transition["dones"]),
                    "infos": copy.deepcopy(sampled_transition.get("infos", info)),
                }
                reward = transition["rewards"]
                done = transition["dones"]
                truncated = bool(sampled_transition.get("truncated", False))
                intervened = True
            else:
                transition = {
                    "observations": obs,
                    "actions": np.asarray(actions, dtype=np.float32),
                    "next_observations": next_obs,
                    "rewards": np.asarray(reward, dtype=np.float32),
                    "masks": np.asarray(1.0 - float(done), dtype=np.float32),
                    "dones": bool(done),
                    "infos": copy.deepcopy(info),
                }

            transition["grasp_penalty"] = (
                np.asarray(info.get("grasp_penalty", 0.0), dtype=np.float32)
                if isinstance(info, dict)
                else np.asarray(0.0, dtype=np.float32)
            )

            data_store.insert(transition)
            transitions.append(transition.copy())
            if intervened:
                intvn_data_store.insert(transition)
                demo_transitions.append(transition.copy())

            obs = next_obs
            running_return += float(np.asarray(reward))

            if done or truncated:
                if isinstance(info, dict) and "episode" in info and isinstance(info["episode"], dict):
                    info["episode"]["intervention_count"] = intervention_count
                    info["episode"]["intervention_steps"] = intervention_steps
                    client.request("send-stats", {"environment": info})
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                already_intervened = False
                client.update()
                obs, _ = env.reset()

        if step > 0 and step % config.steps_per_update == 0:
            client.update()

        if checkpoint_path and step > 0 and step % config.checkpoint_period == 0:
            save_transition_dump(checkpoint_path, "buffer", step, transitions)
            save_transition_dump(checkpoint_path, "demo_buffer", step, demo_transitions)
            transitions = []
            demo_transitions = []

        timer.tock("total")
        if step % config.log_period == 0 and last_timer_stats_step != step:
            client.request("send-stats", {"timer": timer.get_average_times()})
            last_timer_stats_step = step
        step += 1
        pbar.update(1)
    pbar.close()


# ---------------------------------------------------------------------------
# learner
# ---------------------------------------------------------------------------
def learner(config, rng, agent, replay_buffer, demo_buffer, wandb_logger, checkpoint_path):
    max_steps = _cfg_value(config, "max_steps", FLAGS.max_steps)
    batch_size = _cfg_value(config, "batch_size", FLAGS.batch_size)

    start_step = 0
    latest = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        latest = checkpoints.latest_checkpoint(os.path.abspath(checkpoint_path))
    if latest:
        start_step = int(os.path.basename(latest)[11:]) + 1
    step_state = {"value": start_step}

    def stats_callback(req_type, payload):
        assert req_type == "send-stats"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step_state["value"])
        return {}

    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    pbar = tqdm.tqdm(
        total=config.training_starts,
        initial=len(replay_buffer),
        desc="Filling replay buffer",
        leave=True,
    )
    while len(replay_buffer) < config.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)
    pbar.close()

    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    def make_replay_iterator(bs):
        return replay_buffer.get_iterator(
            sample_args={"batch_size": bs, "pack_obs_and_next_obs": True},
            device=sharding,
        )

    def make_demo_iterator(bs):
        return demo_buffer.get_iterator(
            sample_args={"batch_size": bs, "pack_obs_and_next_obs": True},
            device=sharding,
        )

    demo_batch_size = max(1, batch_size // 2)
    use_demo = len(demo_buffer) > 0
    replay_batch_size = batch_size // 2 if use_demo else batch_size
    replay_iterator = make_replay_iterator(replay_batch_size)
    demo_iterator = make_demo_iterator(demo_batch_size) if use_demo else None
    if use_demo:
        print_green(
            f"demo buffer available at startup; mixed mode "
            f"(replay_batch={replay_batch_size}, demo_batch={demo_batch_size})."
        )
    else:
        print_yellow("no demo data at startup; replay-only mode (will switch when demos arrive).")

    train_critic_networks = frozenset({"critic", "grasp_critic"})
    train_networks = frozenset({"critic", "grasp_critic", "actor", "temperature"})
    timer = Timer()

    for step in tqdm.tqdm(range(start_step, max_steps), dynamic_ncols=True, desc="learner"):
        step_state["value"] = step
        if not use_demo and len(demo_buffer) > 0:
            use_demo = True
            replay_batch_size = batch_size // 2
            replay_iterator = make_replay_iterator(replay_batch_size)
            demo_iterator = make_demo_iterator(demo_batch_size)
            print_green(f"demo buffer became available; switching to mixed mode at step {step}.")

        for _ in range(max(config.critic_actor_ratio - 1, 0)):
            with timer.context("sample_replay"):
                batch = next(replay_iterator)
                if use_demo:
                    batch = concat_batches(batch, next(demo_iterator), axis=0)
            with timer.context("train_critics"):
                agent, _ = agent.update(batch, networks_to_update=train_critic_networks)

        with timer.context("train"):
            batch = next(replay_iterator)
            if use_demo:
                batch = concat_batches(batch, next(demo_iterator), axis=0)
            agent, update_info = agent.update(batch, networks_to_update=train_networks)

        if step > 0 and step % config.steps_per_update == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if wandb_logger and step % config.log_period == 0:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log(
                {
                    "timer": timer.get_average_times(),
                    "replay_size": len(replay_buffer),
                    "intervention_buffer_size": len(demo_buffer),
                },
                step=step,
            )

        if checkpoint_path and step > 0 and step % config.checkpoint_period == 0:
            checkpoints.save_checkpoint(
                os.path.abspath(checkpoint_path), agent.state, step=step, keep=100
            )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(_):
    if sum([FLAGS.actor, FLAGS.learner, FLAGS.eval]) != 1:
        raise ValueError("Exactly one of --actor, --learner, --eval must be true.")

    config = CONFIG_MAPPING[FLAGS.exp_name]()
    batch_size = _cfg_value(config, "batch_size", FLAGS.batch_size)
    if batch_size % max(num_devices, 1) != 0:
        raise ValueError("batch_size must be divisible by the number of local JAX devices.")

    checkpoint_path = resolve_checkpoint_path(FLAGS.exp_name)
    bc_checkpoint_path = resolve_bc_checkpoint_path(FLAGS.exp_name)
    rng = jax.random.PRNGKey(config.seed if hasattr(config, "seed") else FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    # learner uses a fake env (no hardware); actor/eval use the real env + classifier reward.
    env = config.get_environment(
        fake_env=FLAGS.learner or FLAGS.mock,
        save_video=FLAGS.save_video,
        classifier=(FLAGS.actor or FLAGS.eval),
    )

    print(f"Exp: {FLAGS.exp_name} | image_keys={config.image_keys}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"BC checkpoint path: {bc_checkpoint_path}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    agent = create_agent(config, env)
    agent = restore_rlpd_or_bc(agent, checkpoint_path, bc_checkpoint_path)

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding)
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=True,
        )
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=True,
        )
        if checkpoint_path:
            files, transitions = load_transition_dir(os.path.join(checkpoint_path, "buffer"), replay_buffer)
            print_green(f"loaded checkpoint replay buffer: {files} files, {transitions} transitions")
            files, transitions = load_transition_dir(
                os.path.join(checkpoint_path, "demo_buffer"), demo_buffer
            )
            print_green(
                f"loaded online intervention demo buffer: {files} files, "
                f"{transitions} transitions from {os.path.join(checkpoint_path, 'demo_buffer')}"
            )
        success_demo_dir = task_success_dir(FLAGS.exp_name)
        files, transitions = load_transition_dir(success_demo_dir, demo_buffer)
        print_green(
            f"loaded offline success demos: {files} files, {transitions} transitions from {success_demo_dir}"
        )
        print_green(f"replay buffer size: {len(replay_buffer)}")
        print_green(f"demo buffer size: {len(demo_buffer)}")

        wandb_logger = make_wandb_logger(
            project="rl-serl", description=FLAGS.exp_name, debug=FLAGS.debug
        )
        learner(config, sampling_rng, agent, replay_buffer, demo_buffer, wandb_logger, checkpoint_path)
    else:
        sampling_rng = jax.device_put(sampling_rng, sharding)
        data_store = QueuedDataStore(50000)
        intvn_data_store = QueuedDataStore(50000)
        actor(config, agent, data_store, intvn_data_store, env, sampling_rng, checkpoint_path)


if __name__ == "__main__":
    app.run(main)
