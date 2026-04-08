#!/usr/bin/env python3

import os
import sys
import ctypes
import site
import time
import json
import datetime
import cv2
import threading
from pathlib import Path

IS_TRAIN_MODE = any(arg == "--mode=train" for arg in sys.argv)
os.environ.setdefault("SERL_FORCE_CPU_RESNET_PICKLE_LOAD", "1")

nvidia_base = os.path.join(site.getsitepackages()[0], "nvidia")
libs = [
    "cublas/lib",
    "cudnn/lib",
    "cufft/lib",
    "cusolver/lib",
    "cusparse/lib",
    "nccl/lib",
    "nvjitlink/lib",
]
for lib in libs:
    path = os.path.join(nvidia_base, lib)
    if os.path.exists(path):
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        ld_entries = [entry for entry in current_ld.split(":") if entry]
        if path not in ld_entries:
            os.environ["LD_LIBRARY_PATH"] = ":".join([path] + ld_entries)

required_xla_flags = [
    f"--xla_gpu_cuda_data_dir={nvidia_base}",
    "--xla_gpu_strict_conv_algorithm_picker=false",
    "--xla_gpu_autotune_level=0",
]
current_xla_flags = os.environ.get("XLA_FLAGS", "").strip()
merged_xla_flags = current_xla_flags
for required_flag in required_xla_flags:
    if required_flag not in merged_xla_flags:
        merged_xla_flags = " ".join(
            flag for flag in [merged_xla_flags, required_flag] if flag
        )
os.environ["XLA_FLAGS"] = merged_xla_flags

try:
    nvjitlink_path = os.path.join(nvidia_base, "nvjitlink/lib/libnvJitLink.so.12")
    if os.path.exists(nvjitlink_path):
        ctypes.CDLL(nvjitlink_path)

    cusparse_path = os.path.join(nvidia_base, "cusparse/lib/libcusparse.so.12")
    if os.path.exists(cusparse_path):
        ctypes.CDLL(cusparse_path)
except Exception as e:
    print(f"[DEBUG] Failed to preload libraries: {e}")

import jax
import jaxlib
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints

import gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics

InputDevice = None
ecodes = None
RealsenseCamera = None
OpenCVCamera = None

if not IS_TRAIN_MODE:
    try:
        from evdev import InputDevice, ecodes
    except Exception:
        InputDevice = None
        ecodes = None

    sys.path.append(str(Path(__file__).parent.parent / "pyroki"))
    try:
        from realsense_camera import RealsenseCamera, OpenCVCamera
    except Exception as e:
        RealsenseCamera = None
        OpenCVCamera = None
        print(f"Failed to import camera modules: {e}")

from openarm_env import OpenArmEnv, DefaultOpenArmConfig
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.wrappers import Quat2EulerWrapper

from serl_launcher.agents.continuous.bc import BCAgent
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from serl_launcher.utils.launcher import make_bc_agent, make_wandb_logger
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

from pick_place_demo_utils import (
    TRAINING_IMAGE_KEYS,
    insert_transitions_into_buffer,
    load_demo_transitions,
    resolve_demo_path,
)


FLAGS = flags.FLAGS

flags.DEFINE_enum("mode", "train", ["train", "eval"], "Run mode.")
flags.DEFINE_string("arm", "right", "控制哪个机械臂: 'left', 'right', 或 'both'")
flags.DEFINE_integer("hz", 5, "环境控制频率（Hz）")
flags.DEFINE_integer("max_traj_length", 300, "最大轨迹长度")
flags.DEFINE_integer("seed", 42, "随机种子")
flags.DEFINE_integer("batch_size", 256, "Batch 大小")
flags.DEFINE_integer("max_steps", 10000, "最大 BC 训练步数")
flags.DEFINE_integer("replay_buffer_capacity", 10000, "Replay buffer 容量")
flags.DEFINE_integer("log_period", 10, "日志记录周期")
flags.DEFINE_integer("save_period", 1000, "保存 checkpoint 周期")
flags.DEFINE_string("exp_name", "pick_place_bc", "实验名称")
flags.DEFINE_string("encoder_type", "resnet-pretrained", "编码器类型")
flags.DEFINE_string(
    "bc_log_dir",
    "/home/sj/Desktop/zy/moqi_workspace/rl_deploy/logs",
    "BC 本地训练日志目录",
)
flags.DEFINE_string(
    "demo_path",
    "/home/sj/Desktop/zy/moqi_workspace/rl_deploy/demo/merged_demos.pkl",
    "自定义演示数据路径（当 demo_pkl_variant=custom 时使用）",
)
flags.DEFINE_enum(
    "demo_pkl_variant",
    "v2",
    ["v1", "v2", "custom"],
    "选择加载的 demo pkl。",
)
flags.DEFINE_boolean(
    "demo_drop_over_limit_transitions",
    False,
    "加载 demo 时丢弃超出 OpenArmEnv 单步动作范围的 transition。",
)
flags.DEFINE_string(
    "checkpoint_path",
    "/home/sj/Desktop/zy/moqi_workspace/rl_deploy/bc_checkpoints",
    "BC checkpoint 保存路径",
)
flags.DEFINE_integer("eval_checkpoint_step", 0, "评估指定步数的 checkpoint；0 表示最新")
flags.DEFINE_integer("eval_n_trajs", 5, "评估轨迹数量")
flags.DEFINE_boolean("mock", True, "评估时是否使用 mock 环境")
flags.DEFINE_boolean("render", False, "Enable visualization")
flags.DEFINE_boolean("debug", False, "调试模式")

print(f"[BC GPU] jax={jax.__version__} jaxlib={jaxlib.__version__}")
print(f"[BC GPU] backend={jax.default_backend()} devices={jax.devices()}")
print(f"[BC GPU] XLA_FLAGS={os.environ.get('XLA_FLAGS', '')}")


devices = jax.local_devices()
num_devices = len(devices)
if len(devices) == 1:
    sharding = jax.sharding.SingleDeviceSharding(devices[0])
else:
    mesh = jax.sharding.Mesh(devices, ("devices",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("devices"))


HEAD_CAMERA_DEVICE = "/dev/video12"
HEAD_CAMERA_WIDTH = 640
HEAD_CAMERA_HEIGHT = 480
MODEL_IMAGE_SIZE = (128, 128)
APPLY_HEAD_CAMERA_CROP = False


def ensure_gpu_available():
    gpu_devices = [d for d in devices if getattr(d, "platform", "") == "gpu"]
    if gpu_devices:
        print(f"[BC GPU] Detected GPU devices: {gpu_devices}")
        return

    raise RuntimeError(
        "train_pick_place_bc_gpu.py requires a GPU-enabled JAX runtime, "
        f"but detected backend={jax.default_backend()} devices={jax.devices()}. "
        "Run `conda run -n gpu_zy python -c \"import jax; print(jax.devices())\"` "
        "to verify gpu_zy before launching GPU training."
    )


class LocalTrainLogger:
    def __init__(self, root_dir: str, exp_name: str):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(root_dir, f"{exp_name}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.metrics_path = os.path.join(self.run_dir, "train_metrics.jsonl")
        self.summary_path = os.path.join(self.run_dir, "summary.json")
        self.config_path = os.path.join(self.run_dir, "config.json")

    def write_config(self, config: dict):
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def log_metrics(self, payload: dict):
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def write_summary(self, payload: dict):
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


class LocalOpenArmEnv(OpenArmEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "displayer"):
            self.displayer = None
        if self.fake_env:
            print("[LocalOpenArmEnv] Enforcing local camera init for FAKE/MOCK mode.")
            self.init_cameras(None)

    def init_cameras(self, config):
        self.cameras = []
        self.latest_images_raw = {}

        try:
            from mock_hardware import MockCamera
        except ImportError:
            MockCamera = None

        try:
            if self.fake_env:
                cam_left = MockCamera(width=640, height=480, fps=30)
            else:
                cam_left = RealsenseCamera(
                    device_id="150622074105",
                    enable_depth=False,
                    width=640,
                    height=480,
                    fps=30,
                )
            self.cameras.append(("image_left", cam_left))
            print("Initialized Left Camera (150622074105)")
        except Exception as e:
            print(f"Failed to init Left Camera: {e}")

        try:
            if self.fake_env:
                cam_right = MockCamera(width=640, height=480, fps=30)
            else:
                cam_right = RealsenseCamera(
                    device_id="236422072385",
                    enable_depth=False,
                    width=640,
                    height=480,
                    fps=30,
                )
            self.cameras.append(("image_right", cam_right))
            print("Initialized Right Camera (236422072385)")
        except Exception as e:
            print(f"Failed to init Right Camera: {e}")

        try:
            if self.fake_env:
                cam_head = MockCamera(width=HEAD_CAMERA_WIDTH, height=HEAD_CAMERA_HEIGHT, fps=30)
            else:
                cam_head = OpenCVCamera(
                    HEAD_CAMERA_DEVICE,
                    width=HEAD_CAMERA_WIDTH,
                    height=HEAD_CAMERA_HEIGHT,
                    fps=30,
                    exposure=150,
                )
            self.cameras.append(("image_primary", cam_head))
            print(f"Initialized Head Camera ({HEAD_CAMERA_DEVICE})")
        except Exception as e:
            print(f"Failed to init Head Camera: {e}")

        self.stop_event = threading.Event()
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _capture_loop(self):
        while not self.stop_event.is_set():
            for name, cam in self.cameras:
                try:
                    img = cam.get_data(viz=False)
                    is_realsense = isinstance(img, (list, tuple))
                    if is_realsense:
                        img = img[0]

                    if img is None:
                        continue

                    if name == "image_primary" and APPLY_HEAD_CAMERA_CROP:
                        h, w = img.shape[:2]
                        crop_h_ratio = 0.35
                        crop_w_ratio = 0.35
                        cy, cx = h // 2, w // 2
                        half_h = int(h * crop_h_ratio / 2)
                        half_w = int(w * crop_w_ratio / 2)
                        img = img[cy - half_h : cy + half_h, cx - half_w : cx + half_w]

                    if is_realsense:
                        img_rgb_full = img
                    else:
                        img_rgb_full = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.latest_images_raw[name] = img_rgb_full

                    img_resized = cv2.resize(img, MODEL_IMAGE_SIZE)
                    if is_realsense:
                        img_rgb = img_resized
                    else:
                        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                    self.latest_images[name] = img_rgb
                except Exception as e:
                    print(f"[Error] Capture error {name}: {e}")
            time.sleep(0.01)

    def step(self, action):
        return super().step(action)

    def close(self):
        if hasattr(self, "stop_event"):
            self.stop_event.set()
        if hasattr(self, "capture_thread"):
            self.capture_thread.join(timeout=1.0)
        try:
            super().close()
        except AttributeError as e:
            if "displayer" not in str(e):
                raise


def build_env(fake_env):
    class TrainConfig(DefaultOpenArmConfig):
        REALSENSE_CAMERAS = {
            "image_primary": "real_camera_serial_or_empty",
            "image_left": "real_camera_serial_or_empty",
            "image_right": "real_camera_serial_or_empty",
        }

    env = LocalOpenArmEnv(
        fake_env=fake_env,
        save_video=bool(FLAGS.eval_checkpoint_step),
        arm=FLAGS.arm,
        hz=FLAGS.hz,
        config=TrainConfig(),
        max_episode_length=FLAGS.max_traj_length,
    )
    env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    env = RecordEpisodeStatistics(env)
    return env


def render_observation(obs):
    def process_img(img_key):
        if img_key not in obs:
            return np.zeros((128, 128, 3), dtype=np.uint8)
        img = obs[img_key]
        if img.ndim == 4:
            img = img[0]
        if img.size == 0:
            return np.zeros((128, 128, 3), dtype=np.uint8)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    img_primary = cv2.resize(process_img("image_primary"), (256, 256))
    img_left = cv2.resize(process_img("image_left"), (256, 256))
    img_right = cv2.resize(process_img("image_right"), (256, 256))

    cv2.putText(img_primary, "Primary (Head)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(img_left, "Left", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(img_right, "Right (Wrist)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    vis_img = np.hstack([img_left, img_primary, img_right])
    cv2.imshow("BC Eval View (Left | Primary | Right)", vis_img)
    cv2.waitKey(1)


def restore_bc_checkpoint(agent: BCAgent):
    if FLAGS.eval_checkpoint_step:
        restored_state = checkpoints.restore_checkpoint(
            FLAGS.checkpoint_path, agent.state, step=FLAGS.eval_checkpoint_step
        )
        print(f"Restored BC checkpoint step {FLAGS.eval_checkpoint_step}")
        return agent.replace(state=restored_state)

    latest_ckpt = checkpoints.latest_checkpoint(FLAGS.checkpoint_path)
    if latest_ckpt:
        print(f"Restoring latest BC checkpoint from {latest_ckpt}")
        restored_state = checkpoints.restore_checkpoint(FLAGS.checkpoint_path, agent.state)
        return agent.replace(state=restored_state)
    raise FileNotFoundError(f"No checkpoint found in {FLAGS.checkpoint_path}")


def train_bc(agent: BCAgent, env):
    os.makedirs(FLAGS.checkpoint_path, exist_ok=True)
    os.makedirs(FLAGS.bc_log_dir, exist_ok=True)
    demo_path = resolve_demo_path(FLAGS.demo_pkl_variant, FLAGS.demo_path)
    if not os.path.exists(demo_path):
        raise FileNotFoundError(f"Demo path not found: {demo_path}")
    local_logger = LocalTrainLogger(FLAGS.bc_log_dir, FLAGS.exp_name)

    replay_buffer = MemoryEfficientReplayBufferDataStore(
        env.observation_space,
        env.action_space,
        capacity=max(FLAGS.replay_buffer_capacity, 10000),
        image_keys=TRAINING_IMAGE_KEYS,
    )
    transitions, stats = load_demo_transitions(
        demo_path,
        training_image_keys=TRAINING_IMAGE_KEYS,
        drop_over_limit_transitions=FLAGS.demo_drop_over_limit_transitions,
    )
    insert_transitions_into_buffer(replay_buffer, transitions)
    print(f"Loaded BC demos from {demo_path}: {stats.summary()}")
    print(f"BC replay buffer size: {len(replay_buffer)}")
    local_logger.write_config(
        {
            "exp_name": FLAGS.exp_name,
            "demo_path": demo_path,
            "demo_pkl_variant": FLAGS.demo_pkl_variant,
            "demo_drop_over_limit_transitions": FLAGS.demo_drop_over_limit_transitions,
            "batch_size": FLAGS.batch_size,
            "max_steps": FLAGS.max_steps,
            "save_period": FLAGS.save_period,
            "log_period": FLAGS.log_period,
            "encoder_type": FLAGS.encoder_type,
            "seed": FLAGS.seed,
            "checkpoint_path": FLAGS.checkpoint_path,
            "bc_log_dir": local_logger.run_dir,
            "replay_buffer_capacity": max(FLAGS.replay_buffer_capacity, 10000),
            "loaded_transition_count": len(transitions),
            "demo_stats": {
                "total_raw": stats.total_raw,
                "loaded": stats.loaded,
                "over_limit": stats.over_limit,
                "skipped": stats.skipped,
            },
        }
    )
    print(f"[BC Train] Local logs: {local_logger.run_dir}")

    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size,
            "pack_obs_and_next_obs": True,
        },
        device=sharding,
    )
    wandb_logger = make_wandb_logger(
        project="serl_dev",
        description=FLAGS.exp_name,
        debug=FLAGS.debug,
    )

    latest_ckpt = checkpoints.latest_checkpoint(FLAGS.checkpoint_path)
    start_step = 0
    last_info = {}
    train_start_time = time.time()
    if latest_ckpt:
        print(f"Resuming BC from {latest_ckpt}")
        restored_state = checkpoints.restore_checkpoint(FLAGS.checkpoint_path, agent.state)
        agent = agent.replace(state=restored_state)
        start_step = int(agent.state.step)

    for step in tqdm.tqdm(range(start_step, FLAGS.max_steps), dynamic_ncols=True, desc="bc"):
        batch = next(replay_iterator)
        agent, info = agent.update(batch)
        info_np = jax.device_get(info)
        metrics_payload = {
            "step": int(step),
            "actor_loss": float(np.asarray(info_np["actor_loss"])),
            "mse": float(np.asarray(info_np["mse"])),
            "elapsed_sec": float(time.time() - train_start_time),
        }
        last_info = metrics_payload
        if step % FLAGS.log_period == 0 and wandb_logger is not None:
            wandb_logger.log(info, step=step)
            local_logger.log_metrics(metrics_payload)
        if (step + 1) % FLAGS.save_period == 0:
            checkpoints.save_checkpoint(
                FLAGS.checkpoint_path,
                agent.state,
                step=step + 1,
                keep=100,
                overwrite=True,
            )

    checkpoints.save_checkpoint(
        FLAGS.checkpoint_path,
        agent.state,
        step=FLAGS.max_steps,
        keep=100,
        overwrite=True,
    )
    local_logger.write_summary(
        {
            "final_step": int(FLAGS.max_steps),
            "elapsed_sec": float(time.time() - train_start_time),
            "last_metrics": last_info,
            "checkpoint_path": FLAGS.checkpoint_path,
            "log_dir": local_logger.run_dir,
        }
    )


def eval_bc(agent: BCAgent, env):
    agent = restore_bc_checkpoint(agent)
    success_counter = 0.0
    returns = []

    for episode in range(FLAGS.eval_n_trajs):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_return = 0.0
        steps = 0
        while (not done) and (not truncated) and steps < FLAGS.max_traj_length:
            actions = agent.sample_actions(
                observations=jax.device_put(obs),
                argmax=True,
            )
            actions = np.asarray(jax.device_get(actions))
            next_obs, reward, done, truncated, info = env.step(actions)
            obs = next_obs
            episode_return += float(np.asarray(reward))
            steps += 1
            if FLAGS.render:
                render_observation(obs)

        returns.append(episode_return)
        success = float(info.get("success", float(episode_return > 0.0))) if isinstance(info, dict) else 0.0
        success_counter += success
        print(
            f"[BC Eval] episode={episode + 1}/{FLAGS.eval_n_trajs} "
            f"steps={steps} return={episode_return:.4f} success={success}"
        )

    if returns:
        print(f"[BC Eval] success rate: {success_counter / len(returns):.4f}")
        print(f"[BC Eval] average return: {np.mean(returns):.4f}")


def main(_):
    ensure_gpu_available()
    assert FLAGS.batch_size % num_devices == 0
    env = build_env(fake_env=(FLAGS.mode == "train") or FLAGS.mock)
    agent: BCAgent = make_bc_agent(
        FLAGS.seed,
        env.observation_space.sample(),
        env.action_space.sample(),
        image_keys=TRAINING_IMAGE_KEYS,
        encoder_type=FLAGS.encoder_type,
    )
    agent = jax.device_put(jax.tree_util.tree_map(jnp.array, agent), sharding)

    try:
        if FLAGS.mode == "train":
            train_bc(agent, env)
        else:
            eval_bc(agent, env)
    finally:
        env.close()


if __name__ == "__main__":
    app.run(main)
