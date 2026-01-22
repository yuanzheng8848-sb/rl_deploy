#!/usr/bin/env python3

import os
import sys
import ctypes

# --- 强制指定 JAX 使用的 NVIDIA 库路径 ---
# 必须在导入 jax 或其他可能使用它的库之前完成此操作
# 这是为了解决在某些 Conda 环境下 JAX 无法找到 CUDA 库的问题
import site
nvidia_base = os.path.join(site.getsitepackages()[0], "nvidia")
libs = [
    "cublas/lib", "cudnn/lib", "cufft/lib", "cusolver/lib", 
    "cusparse/lib", "nccl/lib", "nvjitlink/lib"
]
for lib in libs:
    path = os.path.join(nvidia_base, lib)
    if os.path.exists(path):
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        # 将 NVIDIA 库路径添加到 LD_LIBRARY_PATH 环境变量中
        os.environ["LD_LIBRARY_PATH"] = f"{path}:{current_ld}"

# 设置 XLA_FLAGS 以帮助 JAX 找到 CUDA 数据目录
os.environ['XLA_FLAGS'] = f"--xla_gpu_cuda_data_dir={nvidia_base}"

# 显式按依赖顺序预加载库
try:
    # 首先预加载 nvJitLink (因为 cusparse 依赖于它)
    nvjitlink_path = os.path.join(nvidia_base, "nvjitlink/lib/libnvJitLink.so.12")
    if os.path.exists(nvjitlink_path):
        ctypes.CDLL(nvjitlink_path)
        print(f"[DEBUG] Successfully preloaded {nvjitlink_path}")
    
    # 然后预加载 cuSPARSE
    cusparse_path = os.path.join(nvidia_base, "cusparse/lib/libcusparse.so.12")
    if os.path.exists(cusparse_path):
        ctypes.CDLL(cusparse_path)
        print(f"[DEBUG] Successfully preloaded {cusparse_path}")
        
    sys.stdout.flush()
except Exception as e:
    print(f"[DEBUG] Failed to preload libraries: {e}")
    sys.stdout.flush()
# ------------------------------------------

import time
from functools import partial

import jax
print(f"DEBUG: JAX Devices: {jax.devices()}")
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import flax.linen as nn
import pickle as pkl
import os
import json
import cv2

# --- Local Camera Support Imports ---
from pathlib import Path
import threading
import queue

# Add pyroki to path for RealsenseCamera
sys.path.append(str(Path(__file__).parent.parent / "pyroki"))
try:
    from realsense_camera import RealsenseCamera, OpenCVCamera
except ImportError as e:
    print(f"Failed to import camera modules: {e}")

from openarm_env import OpenArmEnv, TrainConfig

class LocalOpenArmEnv(OpenArmEnv):
    """
    Subclass of OpenArmEnv that initializes and reads from local cameras
    instead of relying on the server to send images.
    """
    def init_cameras(self, config):
        self.cameras = []
        # Configuration from main_v5_record_velocity.py
        # Left: 150622074105, Right: 236422072385
        # Head: /dev/video18
        
        # 1. Left Camera
        try:
            cam_left = RealsenseCamera(
                device_id="150622074105",
                enable_depth=False,
                width=640,
                height=480,
                fps=30
            )
            self.cameras.append(("image_left", cam_left))
            print("Initialized Left Camera (150622074105)")
        except Exception as e:
            print(f"Failed to init Left Camera: {e}")

        # 2. Right Camera
        try:
            cam_right = RealsenseCamera(
                device_id="236422072385",
                enable_depth=False,
                width=640,
                height=480,
                fps=30
            )
            self.cameras.append(("image_right", cam_right))
            print("Initialized Right Camera (236422072385)")
        except Exception as e:
            print(f"Failed to init Right Camera: {e}")

        # 3. Head Camera (Primary)
        try:
            # Note: Exposure 150 as in reference
            cam_head = OpenCVCamera("/dev/video18", width=1280, height=960, fps=30, exposure=150)
            self.cameras.append(("image_primary", cam_head))
            print("Initialized Head Camera (/dev/video18)")
        except Exception as e:
            print(f"Failed to init Head Camera: {e}")

        # Start Capture Thread
        self.stop_event = threading.Event()
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _capture_loop(self):
        while not self.stop_event.is_set():
            for name, cam in self.cameras:
                try:
                    # get_data returns (color, depth) or just color depending on implementation
                    # RealsenseCamera.get_data returns [color, depth] (LIST)
                    # OpenCVCamera.get_data returns frame (ndarray) or None
                    
                    img = cam.get_data(viz=False)
                    
                    # Check if Realsense (returns list [color, depth])
                    is_realsense = isinstance(img, (list, tuple))
                    
                    # If list or tuple (color, depth), take color
                    if is_realsense:
                        img = img[0]
                    
                    if img is not None:
                        # Resize to 128x128 for RL
                        img_resized = cv2.resize(img, (128, 128))
                        
                        if is_realsense:
                            # Realsense returns RGB, use as is
                            img_rgb = img_resized
                        else:
                            # OpenCV returns BGR, convert to RGB
                            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                        
                        self.latest_images[name] = img_rgb
                    else:
                        # Debug print for missing frame
                        if self.cycle_count % 100 == 0: # Print occasionally
                             print(f"[Warn] Camera {name} returned None frame")
                        
                except Exception as e:
                    print(f"[Error] Capture error {name}: {e}")
                    pass
            
            # Update display queue
            # We need to construct a dict of BGR images for the displayer
            # DISABLE ImageDisplayer to avoid confusion with Actor visualization
            # if hasattr(self, "img_queue"):
            #     display_imgs = {}
            #     for name, img_rgb in self.latest_images.items():
            #          display_imgs[name] = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            #     
            #     # Non-blocking put
            #     try:
            #         # Empty queue first to avoid lag
            #         while not self.img_queue.empty():
            #             self.img_queue.get_nowait()
            #         self.img_queue.put(display_imgs)
            #     except:
            #         pass
                    
            time.sleep(0.03) # ~30Hz

    def close(self):
        if hasattr(self, "stop_event"):
            self.stop_event.set()
        if hasattr(self, "capture_thread"):
            self.capture_thread.join(timeout=1.0)
        super().close()

import gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics

# 导入 SERL 框架组件
from serl_launcher.agents.continuous.drq import DrQAgent
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.utils.train_utils import concat_batches

# 导入 AgentLace 用于分布式训练通信
from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_drq_agent,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

class ClassifierRewardWrapper(gym.Wrapper):
    def __init__(self, env, classifier_ckpt_path, reward_image_key="image_right"):
        super().__init__(env)
        self.reward_image_key = reward_image_key
        
        # Determine sampling rng
        rng = jax.random.PRNGKey(0)
        rng, key = jax.random.split(rng)
        
        # Create a dummy sample to initialize the classifier
        # The classifier implies input key "image_0" from training
        # We must map our reward_image_key to "image_0"
        dummy_img = jnp.zeros((1, 128, 128, 3), dtype=jnp.float32)
        sample = {"image_0": dummy_img, "state": jnp.zeros((1, 14))}
        
        print(f"[RewardWrapper] Loading classifier from {classifier_ckpt_path}...")
        self.classifier_func = load_classifier_func(
            key=key,
            sample=sample,
            image_keys=["image_0"],
            checkpoint_path=classifier_ckpt_path,
        )
        print("[RewardWrapper] Classifier loaded.")

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        
        # Rewritten to bypass obs wrappers and read raw camera image directly
        # This matches eval_classifier.py and main_v4.py logic
        
        raw_img = None
        try:
            # Access the base environment (LocalOpenArmEnv) to get 'latest_images'
            # effective bypass of all wrappers (Chunking, Norm, etc.)
            base_env = self.env.unwrapped
            if hasattr(base_env, "latest_images"):
                # latest_images is {name: np.array(128,128,3) uint8 RGB}
                raw_img = base_env.latest_images.get(self.reward_image_key)
        except Exception as e:
            print(f"[RewardWrapper] Error accessing base env: {e}")

        if raw_img is not None:
            # Match eval_classifier.py preprocessing:
            # 1. Ensure shape (128, 128, 3) - guaranteed by _capture_loop
            # 2. Add batch dimension -> (1, 128, 128, 3)
            # 3. Keep as uint8 (or whatever capture loop produces, which is uint8)
            
            img_input = raw_img[None, ...] # (1, 128, 128, 3)
            
            # Run classifier
            classifier_input = {"image_0": img_input}
            logits = self.classifier_func(classifier_input)
            prob = nn.sigmoid(logits).item()
            
            # Reward Logic
            # User Request: prob > 0.9 -> reward = 1.0, prob > 0.95 -> done = True
            if prob > 0.5:
                reward = 1.0
                if prob > 0.5:
                    done = True
                    print(f"[Reward] Success! Prob: {prob:.4f} (Terminated)")
                else:
                    print(f"[Reward] Good State! Prob: {prob:.4f} (Continuing)")
            else:
                reward = 0.0
            
            info["classifier_prob"] = prob
            info["success"] = (reward == 1.0)
            
            # Update info['success'] for done condition
            if done and prob > 0.95:
                 info["success"] = True
            else:
                 info["success"] = False

        else:
            # Fallback if raw image missing (e.g. first step or camera fail)
            # We explicitly do NOT want to use 'obs' if it differs from raw
            print(f"[RewardWrapper] Warning: Raw image '{self.reward_image_key}' not found in latest_images.")
            reward = 0.0
            info["classifier_prob"] = 0.0
        
        return obs, reward, done, truncated, info

class FixedGripperWrapper(gym.ActionWrapper):
    """
    Wrapper to enforce a fixed gripper action (e.g., 70% open).
    Ignores the agent's gripper output and replaces it with a fixed value.
    """
    def __init__(self, env, fixed_value=-0.4):
        super().__init__(env)
        self.fixed_value = fixed_value
    
    def action(self, action):
        # Action is assumed to be numpy array
        new_action = action.copy()
        
        # Access unwrapped env to check arm config
        # Handle various wrapping layers
        unwrapped = self.env.unwrapped
        arm = getattr(unwrapped, "arm", "both")
        
        if arm == "both":
            # Indices 6 and 13 are grippers
            if new_action.shape[0] >= 14:
                new_action[6] = self.fixed_value
                new_action[13] = self.fixed_value
        else:
            # Index 6 is gripper
            if new_action.shape[0] >= 7:
                 new_action[6] = self.fixed_value
        
        return new_action

# 本地导入 (自定义环境和 Wrappers)
from openarm_env import OpenArmEnv, DefaultOpenArmConfig
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.wrappers import Quat2EulerWrapper
from franka_env.utils.transformations import construct_homogeneous_matrix, construct_adjoint_matrix
from scipy.spatial.transform import Rotation as R

FLAGS = flags.FLAGS

# 定义命令行参数
flags.DEFINE_string("env", "OpenArmEnv", "环境名称")
flags.DEFINE_string("arm", "right", "控制哪个机械臂: 'left', 'right', 或 'both'")
flags.DEFINE_string("agent", "drq", "Agent 名称")
flags.DEFINE_string("exp_name", "forward_reach_10cm", "实验名称，用于 wandb 日志")
flags.DEFINE_integer("max_traj_length", 25, "最大轨迹长度")
flags.DEFINE_integer("seed", 42, "随机种子")
flags.DEFINE_bool("save_model", False, "是否保存模型")
flags.DEFINE_integer("critic_actor_ratio", 4, "Critic 与 Actor 的更新比例 (Critic 更新次数 / Actor 更新次数)")

flags.DEFINE_integer("max_steps", 20000, "最大训练步数")
flags.DEFINE_integer("replay_buffer_capacity", 2000, "Replay buffer 容量")
flags.DEFINE_integer("batch_size", 256, "Batch 大小")

flags.DEFINE_integer("random_steps", 50, "随机动作采样步数 (Warmup)")
flags.DEFINE_integer("training_starts", 50, "开始训练的步数")
flags.DEFINE_integer("steps_per_update", 30, "每隔多少步更新一次服务器 (Actor -> Learner)")

flags.DEFINE_integer("log_period", 10, "日志记录周期")
flags.DEFINE_integer("eval_period", 2000, "评估周期")

# 标志位：指示当前进程是 Learner 还是 Actor
flags.DEFINE_boolean("learner", False, "是 Learner 还是 Trainer")
flags.DEFINE_boolean("actor", False, "是 Learner 还是 Trainer")
flags.DEFINE_boolean("render", False, "是否渲染环境")
flags.DEFINE_string("ip", "localhost", "Learner 的 IP 地址")
# "small" 是 4 层卷积网络，"resnet" 和 "mobilenet" 是冻结权重的预训练网络
flags.DEFINE_string("encoder_type", "resnet-pretrained", "编码器类型")
flags.DEFINE_string("demo_path", "./rl_success_demos.pkl", "演示数据路径")
flags.DEFINE_integer("checkpoint_period", 1000, "保存 Checkpoint 的周期")
flags.DEFINE_string("checkpoint_path", "/home/sj/Desktop/zy/moqi_workspace/rl_deploy_2/checkpoints", "保存 Checkpoint 的路径")
flags.DEFINE_string(
    "reward_classifier_ckpt_path", "/home/sj/Desktop/zy/moqi_workspace/rl_deploy_2/classifier_ckpt_cam1", "Path to reward classifier ckpt for cam1."
)

flags.DEFINE_integer(
    "eval_checkpoint_step", 0, "从该步数的 ckpt 评估策略"
)
flags.DEFINE_integer("eval_n_trajs", 5, "评估时的轨迹数量")

flags.DEFINE_boolean(
    "debug", False, "调试模式"
)  # 调试模式将禁用 wandb 日志

flags.DEFINE_boolean(
    "mock", False, "使用模拟环境 (fake_env=True)，即使对于 Actor"
)

# 初始化 JAX 设备
devices = jax.local_devices()
num_devices = len(devices)
if len(devices) == 1:
    sharding = jax.sharding.SingleDeviceSharding(devices[0])
else:
    # Fallback/Polyfill for PositionalSharding on new JAX if needed, 
    # or just use NamedSharding for multi-device.
    # For now, simplistic approach for 0.8.x:
    mesh = jax.sharding.Mesh(devices, ('devices',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('devices'))


def print_green(x):
    """打印绿色文本"""
    return print("\033[92m {}\033[00m".format(x))


##############################################################################


def actor(agent: DrQAgent, data_store, env, sampling_rng):
    """
    Actor 循环，当 "--actor" 设置为 True 时运行。
    负责与环境交互、收集数据并发送给 Learner。
    """
    # 如果指定了评估步数，则进入评估模式
    if FLAGS.eval_checkpoint_step:
        success_counter = 0
        time_list = []

        # 加载指定步数的 Checkpoint
        ckpt = checkpoints.restore_checkpoint(
            FLAGS.checkpoint_path,
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)

        # 运行评估循环
        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset()
            done = False
            start_time = time.time()
            while not done:
                # 采样动作 (评估时使用确定性策略 argmax=True)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=True,
                )
                actions = np.asarray(jax.device_get(actions))

                next_obs, reward, done, truncated, info = env.step(actions)
                obs = next_obs

                if done:
                    if reward:
                        dt = time.time() - start_time
                        time_list.append(dt)
                        print(dt)

                    success_counter += reward
                    print(reward)
                    print(f"{success_counter}/{episode + 1}")

        print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        print(f"average time: {np.mean(time_list)}")
        return  # 评估完成后退出

    # 创建 TrainerClient 连接到 Learner
    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_store,
        wait_for_server=True,
    )

    # 回调函数：用于更新 Agent 的参数
    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    # 注册接收网络参数的回调
    client.recv_network_callback(update_params)

    print("[Actor] Resetting environment...")
    obs, _ = env.reset()
    print("[Actor] Environment reset done.")
    done = False

    # 训练循环
    timer = Timer()
    running_return = 0.0

    print("[Actor] Starting training loop...")
    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True):
        timer.tick("total")

        with timer.context("sample_actions"):
            # 在初始随机步数内，使用随机动作进行探索
            if step < FLAGS.random_steps:
                actions = env.action_space.sample()
            else:
                # 之后使用 Agent 策略采样动作 (随机性用于探索)
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    deterministic=False,
                )
                actions = np.asarray(jax.device_get(actions))

        # 执行环境步骤
        with timer.context("step_env"):
            # print(f"[Actor] Step {step} start...")
            next_obs, reward, done, truncated, info = env.step(actions)
            # print(f"[Actor] Step {step} done.")

            # 如果存在人工干预 (例如通过 3D 鼠标)，覆盖 Agent 的动作
            if "intervene_action" in info:
                actions = info.pop("intervene_action")

            reward = np.asarray(reward, dtype=np.float32)
            info = np.asarray(info)
            running_return += reward
            
            # 构建 Transition 数据字典
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done,
            )
            # 将数据插入本地队列，准备发送给 Learner
            data_store.insert(transition)



            obs = next_obs

            # --- Visualization ---
            try:
                if FLAGS.render:
                    import cv2
                    
                    # Helper to process image
                    def process_img(img_key):
                        if img_key in next_obs:
                            img = next_obs[img_key]
                            if img.ndim == 4:
                                img = img[0]
                            # Check if image is valid
                            if img.size == 0:
                                return np.zeros((128, 128, 3), dtype=np.uint8)
                            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        return np.zeros((128, 128, 3), dtype=np.uint8)

                    # Get all 3 images
                    img_primary = process_img("image_primary")
                    img_left = process_img("image_left")
                    img_right = process_img("image_right")

                    # Resize to common height for stacking (e.g., 256)
                    h, w = 256, 256
                    img_primary = cv2.resize(img_primary, (w, h))
                    img_left = cv2.resize(img_left, (w, h))
                    img_right = cv2.resize(img_right, (w, h))

                    # Add labels
                    def add_label(img, text):
                        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        return img

                    img_primary = add_label(img_primary, "Primary (Head)")
                    img_left = add_label(img_left, "Left")
                    img_right = add_label(img_right, "Right (Wrist)")

                    # Stack horizontally
                    vis_img = np.hstack([img_left, img_primary, img_right])

                    # Draw Reward & Probability on the combined image
                    # info is 0-d array wrapping dict
                    info_dict = info.item() if isinstance(info, np.ndarray) and info.ndim == 0 else info
                    if isinstance(info_dict, dict):
                        prob = info_dict.get("classifier_prob", 0.0)
                        # dist = info_dict.get("dist_to_target", 999.0)
                        rew = reward
                        
                        # Color: Green if success (prob > 0.5), Red otherwise
                        color = (0, 255, 0) if prob > 0.5 else (0, 0, 255)
                        
                        text = f"Prob: {prob:.4f}  Reward: {rew}"
                        cv2.putText(vis_img, text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    cv2.imshow("Actor View (Left | Primary | Right)", vis_img)
                    cv2.waitKey(1)
            except Exception as e:
                print(f"Vis Error: {e}")
            # ---------------------

            if done or truncated:
                stats = {"train": info}  # 发送统计数据给 Learner 记录日志
                client.request("send-stats", stats)
                running_return = 0.0
                obs, _ = env.reset()

        # 定期从 Learner 更新网络参数
        if step % FLAGS.steps_per_update == 0:
            client.update()

        timer.tock("total")

        # 定期发送计时器统计数据
        if step % FLAGS.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


##############################################################################


def learner(rng, agent: DrQAgent, replay_buffer, demo_buffer):
    """
    Learner 循环，当 "--learner" 设置为 True 时运行。
    负责从 Buffer 采样数据并更新 Agent 网络。
    """
    # 设置 wandb 和日志记录
    wandb_logger = make_wandb_logger(
        project="serl_dev",
        description=FLAGS.exp_name or FLAGS.env,
        debug=FLAGS.debug,
    )

    # 跟踪训练步数
    update_steps = int(agent.state.step)
    
    # 如果从 Checkpoint 恢复，调整 update_steps 以匹配 Checkpoint 的步数
    # 这样 WandB 日志可以连续记录
    if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path):
        latest_ckpt = checkpoints.latest_checkpoint(FLAGS.checkpoint_path)
        if latest_ckpt:
            try:
                ckpt_step = int(latest_ckpt.split('_')[-1])
                if ckpt_step < update_steps:
                    print(f"Adjusting logging step from {update_steps} to {ckpt_step} to match checkpoint.")
                    update_steps = ckpt_step
            except ValueError:
                pass

    def stats_callback(type: str, payload: dict) -> dict:
        """服务器接收到统计请求时的回调函数"""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=update_steps)
        return {}  # 不期望返回值

    # 创建 TrainerServer
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.start(threaded=True)

    # 等待 Replay Buffer 填充到 training_starts 数量
    pbar = tqdm.tqdm(
        total=FLAGS.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < FLAGS.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # 更新进度条
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # 更新进度条
    pbar.close()

    # 发送初始网络参数给 Actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # 创建数据迭代器 (RLPD 核心逻辑)
    # 50% 数据来自 Replay Buffer (在线数据)，50% 来自 Demo Buffer (演示数据)
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding,
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding,
    )

    # Learner 主循环
    timer = Timer()
    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True, desc="learner"):
        # 运行 n-1 次 Critic 更新和 1 次 Critic + Actor 更新
        # 这通过减少 CPU 到 GPU 的大 Batch 传输次数来加速训练
        for critic_step in range(FLAGS.critic_actor_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                # 拼接在线数据和演示数据
                batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                # 仅更新 Critic
                agent, critics_info = agent.update_critics(
                    batch,
                )

        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            # 更新 Actor 和 Critic (High UTD)
            agent, update_info = agent.update_high_utd(batch, utd_ratio=1)

        # 发布更新后的网络参数
        if step > 0 and step % (FLAGS.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        # 记录日志
        if update_steps % FLAGS.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=update_steps)
            wandb_logger.log({"timer": timer.get_average_times()}, step=update_steps)

        # 保存 Checkpoint
        if FLAGS.checkpoint_period and update_steps % FLAGS.checkpoint_period == 0:
            assert FLAGS.checkpoint_path is not None
            checkpoints.save_checkpoint(
                FLAGS.checkpoint_path, agent.state, step=update_steps, keep=100, overwrite=True
            )

        update_steps += 1


##############################################################################


def main(_):
    assert FLAGS.batch_size % num_devices == 0
    # 设置随机种子
    rng = jax.random.PRNGKey(FLAGS.seed)

    # 定义包含摄像头的配置，以触发 Observation Space 的创建
    # 键名必须与 _update_currpos 生成的匹配 ("image_primary" 等)
    class TrainConfig(DefaultOpenArmConfig):
        REALSENSE_CAMERAS = {
            "image_primary": "real_camera_serial_or_empty", # <--- 使用真实摄像头
            "image_left": "real_camera_serial_or_empty",
            "image_right": "real_camera_serial_or_empty"
        }
        
    # 加载固定目标位姿 (如果可用)
    fixed_target_pose = None
    target_pose_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "target_pose.json")
    if os.path.exists(target_pose_path):
        try:
            with open(target_pose_path, "r") as f:
                target_data = json.load(f)
            
            # 根据选择的手臂加载对应的目标位姿
            if FLAGS.arm == "right":
                if target_data.get("right"):
                    pose = target_data["right"]
                    fixed_target_pose = np.array(pose[4:7]) # x, y, z
                    print(f"Loaded fixed target pose for RIGHT arm: {fixed_target_pose}")
            elif FLAGS.arm == "left":
                if target_data.get("left"):
                    pose = target_data["left"]
                    fixed_target_pose = np.array(pose[4:7])
                    print(f"Loaded fixed target pose for LEFT arm: {fixed_target_pose}")
            elif FLAGS.arm == "both":
                if target_data.get("right"):
                    pose = target_data["right"]
                    fixed_target_pose = np.array(pose[4:7])
                    print(f"Loaded fixed target pose for BOTH (Right) arm: {fixed_target_pose}")
        except Exception as e:
            print(f"Failed to load target_pose.json: {e}")




    # 创建环境并加载数据集
    # 硬编码为 Right Arm
    # Use LocalOpenArmEnv instead of OpenArmEnv
    env = LocalOpenArmEnv(
        fake_env=FLAGS.learner or FLAGS.mock, # Learner 使用模拟环境
        save_video=FLAGS.eval_checkpoint_step,
        arm="right",
        hz=7, # <--- 修改控制频率 (HZ)
        config=TrainConfig(),
        fixed_target_pose=fixed_target_pose,
        max_episode_length=FLAGS.max_traj_length # Pass max_traj_length
    )
    
    # Wrappers (必须与 record_demo.py 匹配)
    env = RelativeFrame(env) # 相对坐标系
    env = Quat2EulerWrapper(env) # 四元数转欧拉角
    env = SERLObsWrapper(env) # 图像观测处理
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None) # 动作分块


    # 获取图像键列表
    image_keys = [key for key in env.observation_space.keys() if key != "state"]

    # Create random keys
    rng, sampling_rng = jax.random.split(rng)

    if FLAGS.actor:
        # initialize the reward wrapper
        if FLAGS.reward_classifier_ckpt_path is None:
             raise ValueError("reward_classifier_ckpt_path must be specified for actor")

        # Use ClassifierRewardWrapper
        # Assuming 'image_right' corresponds to the arm camera used for training
        env = ClassifierRewardWrapper(env, classifier_ckpt_path=FLAGS.reward_classifier_ckpt_path, reward_image_key="image_right")

        # Apply Fixed Gripper Wrapper immediately to override agent actions
        # 95% Open corresponds to raw value -0.9
        # (Linear map: 1.0=Closed, -1.0=Open -> 1 - 2*0.95 = -0.9)
        env = FixedGripperWrapper(env, fixed_value=-0.9)
    
    # Move RecordEpisodeStatistics here (outermost) to capture rewards from ForwardReachRewardWrapper
    env = RecordEpisodeStatistics(env) # 记录统计信息

    # rng, sampling_rng = jax.random.split(rng) # Moved up
    # 创建 DrQ Agent
    # make_drq_agent 是一个工厂函数，它不仅初始化网络结构（Actor, Critic, Encoder），
    # 还会初始化优化器状态（Optimizer State）。此时 Agent 还只存在于 CPU 内存中。
    agent: DrQAgent = make_drq_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),   # 提供一个观测样本，让网络自动推断输入层形状
        sample_action=env.action_space.sample(),     # 提供一个动作样本，让网络推断输出层形状
        image_keys=image_keys,                       # 告诉 Agent 哪些观测是图像（需要经过 CNN 编码器）
        encoder_type=FLAGS.encoder_type,             # 选择编码器架构（如 "resnet-pretrained"）
    )

    # 在设备间复制 Agent (用于多 GPU 训练，这里通常是单 GPU)
    # jax.device_put: 将数据从 CPU 内存移动到 GPU 显存
    # sharding.replicate(): 指定分片策略为“复制”。如果有多个 GPU，Agent 会在每个 GPU 上都拷贝一份（数据并行）。
    # jax.tree_util.tree_map(jnp.array, agent): 这是一个防御性编程。
    # 有时候 Python 的原生类型（int/float）混在 Agent 结构里会导致 device_put 报错，
    # 这里强制把 Agent 里的所有叶子节点都转成 JAX 数组 (jnp.array)。
    agent: DrQAgent = jax.device_put(
        jax.tree_util.tree_map(jnp.array, agent), sharding
    )

    # 如果存在 Checkpoint，则恢复
    if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path):
        latest_ckpt = checkpoints.latest_checkpoint(FLAGS.checkpoint_path)
        if latest_ckpt:
            print(f"Restoring checkpoint from {latest_ckpt}")
            restored_state = checkpoints.restore_checkpoint(FLAGS.checkpoint_path, agent.state)
            
            try:
                ckpt_step = int(latest_ckpt.split('_')[-1])
                print(f"Restored agent step: {int(restored_state.step)}, Checkpoint step: {ckpt_step}")
                agent = agent.replace(state=restored_state)
            except ValueError:
                print("Warning: Could not parse step from checkpoint path")
            
            print(f"Resumed from internal step {int(agent.state.step)}")
        else:
            print("No checkpoint found, starting from scratch.")

    if FLAGS.learner:
        # Learner 模式：初始化 Replay Buffer 和 Demo Buffer
        
        # 将随机数生成器也放到设备上，并复制。
        # 这样在多设备训练时，每个设备都能独立生成随机数（用于从 Buffer 采样等操作）。
        sampling_rng = jax.device_put(sampling_rng, device=sharding)
        
        # 1. 初始化在线 Replay Buffer (存放 Actor 刚刚采集的新鲜数据)
        # MemoryEfficient: 这个类通常会对图像数据做优化（例如存为 uint8 格式，只在训练时转为 float32），以节省内存。
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=FLAGS.replay_buffer_capacity, # 容量通常较大 (如 20万)
            image_keys=image_keys,
        )
        
        # 2. 初始化演示 Demo Buffer (存放离线录制的专家数据)
        # 这是一个独立的 Buffer，用于 RLPD (RL with Prior Data) 算法。
        # 训练时，我们会从这两个 Buffer 各取一半数据 (50/50) 来更新网络。
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=10000,                        # 容量较小，只需装下所有 Demo 即可
            image_keys=image_keys,
        )

        # 加载 Demo 数据
        if FLAGS.demo_path:
            # 检查文件是否存在
            if not os.path.exists(FLAGS.demo_path):
                raise FileNotFoundError(f"File {FLAGS.demo_path} not found")

            with open(FLAGS.demo_path, "rb") as f:
                trajs = pkl.load(f)
                for traj in trajs:
                    traj_len = len(traj["actions"])
                    for i in range(traj_len):
                        # --- 适配 RelativeFrame 和 Quat2EulerWrapper ---
                        # 1. 获取重置位姿 (轨迹的第一帧)
                        # 假设 Demo 数据结构: state 是 (N, 16) 或类似
                        # 我们需要提取 7 维位姿 (XYZ + Quat)
                        
                        # 辅助函数：从状态向量中获取位姿
                        def get_pose(s, arm):
                            if arm == "left": return s[:7]
                            elif arm == "right": return s[8:15] # 8-14 是位姿 (7), 15 是夹爪
                            return s[:7] # 默认
                            
                        def get_gripper(s, arm):
                            if arm == "left": return s[7:8]
                            elif arm == "right": return s[15:16]
                            return s[7:8]
                        
                        # 获取该轨迹的初始位姿 (Reset Pose)
                        reset_state_vec = traj["observations"]["state"][0]
                        reset_pose_abs = get_pose(reset_state_vec, FLAGS.arm)
                        
                        # 计算 T_reset_inv
                        T_reset = construct_homogeneous_matrix(reset_pose_abs)
                        T_reset_inv = np.linalg.inv(T_reset)
                        
                        # 处理当前步
                        curr_state_vec = traj["observations"]["state"][i]
                        next_state_vec = traj["next_observations"]["state"][i]
                        
                        curr_pose_abs = get_pose(curr_state_vec, FLAGS.arm)
                        next_pose_abs = get_pose(next_state_vec, FLAGS.arm)
                        
                        curr_gripper = get_gripper(curr_state_vec, FLAGS.arm)
                        next_gripper = get_gripper(next_state_vec, FLAGS.arm)
                        
                        # 2. 计算相对位姿
                        def to_relative(pose_abs, T_inv):
                            T_curr = construct_homogeneous_matrix(pose_abs)
                            T_rel = T_inv @ T_curr
                            p_rel = T_rel[:3, 3]
                            q_rel = R.from_matrix(T_rel[:3, :3]).as_quat()
                            return np.concatenate([p_rel, q_rel])
                            
                        curr_pose_rel = to_relative(curr_pose_abs, T_reset_inv)
                        next_pose_rel = to_relative(next_pose_abs, T_reset_inv)
                        
                        # 3. 转换为欧拉角 (Quat2EulerWrapper)
                        def to_euler(pose_quat):
                            xyz = pose_quat[:3]
                            quat = pose_quat[3:]
                            rpy = R.from_quat(quat).as_euler("xyz")
                            return np.concatenate([xyz, rpy])
                            
                        curr_pose_euler = to_euler(curr_pose_rel) # (6,)
                        next_pose_euler = to_euler(next_pose_rel) # (6,)
                        
                        # 4. 添加虚拟速度 (6,) - 因为环境可能需要速度输入
                        vel = np.zeros((6,), dtype=np.float32)
                        
                        # 5. 构建最终状态字典
                        # Env (SERLObsWrapper) sorts keys: gripper_pose(1), tcp_pose(6), tcp_vel(6)
                        # So we must concatenate: [gripper, pose, vel]
                        final_state = np.concatenate([curr_gripper, curr_pose_euler, vel]) # (13,)
                        final_next_state = np.concatenate([next_gripper, next_pose_euler, vel]) # (13,)
                        
                        # 6. Action Normalization & Selection
                        # Demo data from record_demo_rl.py is ALREADY in Body Frame (Physical Units).
                        # Environment expects Normalized Actions in [-1, 1].
                        # So we MUST Normalize: Physical / Scale = Normalized
                        
                        action = traj["actions"][i]
                        if FLAGS.arm == "left":
                            arm_action = action[:7]
                        elif FLAGS.arm == "right":
                            arm_action = action[7:]
                        else:
                            arm_action = action[:7] # default
                            
                        action_pos_physical = arm_action[:3]
                        action_rot_physical = arm_action[3:6]
                        gripper_physical = arm_action[6]
                        
                        # Scale Factors from DefaultOpenArmConfig
                        SCALE_POS = 0.01
                        SCALE_ROT = 0.05
                        
                        action_pos_norm = action_pos_physical / SCALE_POS
                        action_rot_norm = action_rot_physical / SCALE_ROT
                        
                        # Gripper: val = 0.5236 * (norm - 1.0)  =>  norm = (val / 0.5236) + 1.0
                        gripper_norm = (gripper_physical / 0.5236) + 1.0
                        gripper_norm = np.clip(gripper_norm, -1.0, 1.0)
                        
                        final_action = np.concatenate([action_pos_norm, action_rot_norm, [gripper_norm]])
                        
                        # --- 奖励重塑 ---
                        # 成功演示：最后 10 步为成功 (1.0)，其他为路径 (0.0)
                        # 我们假设 rl_success_demos.pkl 仅包含成功演示
                        is_success_step = i >= (traj_len - 10)
                        reward_val = 1.0 if is_success_step else 0.0
                        
                        transition = {
                            "observations": {
                                "state": final_state,
                                **{k: v[i] for k, v in traj["observations"].items() if k != "state"}
                            },
                            "next_observations": {
                                "state": final_next_state,
                                **{k: v[i] for k, v in traj["next_observations"].items() if k != "state"}
                            },
                            "actions": final_action,
                            "rewards": np.array(reward_val, dtype=np.float32), # 覆盖奖励
                            "masks": np.array(1.0, dtype=np.float32), # 确保 Mask 为 1.0
                            "dones": traj["dones"][i],
                        }
                        demo_buffer.insert(transition)
            print(f"demo buffer size: {len(demo_buffer)}")
        else:
            print("WARNING: No demo path provided. Demo buffer will be empty.")

        # 启动 Learner 循环
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,
        )

    elif FLAGS.actor:
        # Actor 模式
        sampling_rng = jax.device_put(sampling_rng, sharding)
        data_store = QueuedDataStore(2000)  # Actor 上的队列大小
        # 启动 Actor 循环
        print_green("starting actor loop")
        actor(agent, data_store, env, sampling_rng)

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
