"""OpenArm pick-and-place task config for rl-serl.

Assembles the OpenArm wrapper stack and, when classifier=True, attaches the
vision reward classifier via MultiCameraBinaryRewardClassifierWrapper.

Reward design:
  - single top camera (image_primary) only
  - classifier consumes the SAME center-cropped image_primary used by the policy
    (NetworkPrimaryImageCropWrapper runs BEFORE the classifier wrapper)
  - classifier_keys = ["image_primary"]
"""
from gymnasium.wrappers import RecordEpisodeStatistics

from openarm_env.envs.openarm_env import DefaultOpenArmConfig
from openarm_env.envs.local_openarm_env import LocalOpenArmEnv
from openarm_env.camera.local_camera import load_deployment_camera_config
from openarm_env.envs.wrappers import (
    DualRelativeFrame,
    Quat2EulerWrapper,
    NetworkPrimaryImageCropWrapper,
    DualSpacemouseIntervention,
    OpenArmPolicyObsAdapter,
)
from openarm_env.envs.reward_wrappers import MultiCameraBinaryRewardClassifierWrapper

from rl_launcher.wrappers import SERLObsWrapper, ChunkingWrapper

from experiments.config import DefaultTrainingConfig
from experiments.artifacts import task_classifier_ckpt_dir


# Image streams fed to the policy encoder (three cameras).
TRAINING_IMAGE_KEYS = ["image_primary", "image_left", "image_right"]
PROPRIO_KEYS = ["tcp_pose", "tcp_vel", "gripper_pose"]
# Reward classifier uses only the top camera.
CLASSIFIER_KEYS = ["image_primary"]

# Default network-crop params.
NETWORK_PRIMARY_CROP_RATIO = 0.3
NETWORK_PRIMARY_CROP_Y_OFFSET = 0.0

# Classifier configuration (used by both classifier training and online RL)
CLASSIFIER_THRESHOLD = 0.85
CLASSIFIER_SUCCESS_TAIL_FRAMES = 30
CLASSIFIER_FAILURE_HEAD_FRAMES = 30

# Default classifier checkpoint location used by train/eval render and reward reference.
DEFAULT_CLASSIFIER_CKPT = str(task_classifier_ckpt_dir("openarm_pickplace"))


class EnvConfig(DefaultOpenArmConfig):
    """OpenArm bimanual env config: three local cameras."""

    CAMERAS = load_deployment_camera_config()


class TrainConfig(DefaultTrainingConfig):
    image_keys = TRAINING_IMAGE_KEYS
    classifier_keys = CLASSIFIER_KEYS
    proprio_keys = PROPRIO_KEYS
    setup_mode = "dual-arm-learned-gripper"
    encoder_type = "resnet-pretrained"

    # Teleop / servo params.
    network_crop_primary = True
    network_primary_crop_ratio = NETWORK_PRIMARY_CROP_RATIO
    network_primary_crop_y_offset = NETWORK_PRIMARY_CROP_Y_OFFSET

    # Classifier configuration (used by both training and online RL)
    classifier_threshold = CLASSIFIER_THRESHOLD
    classifier_success_tail_frames = CLASSIFIER_SUCCESS_TAIL_FRAMES
    classifier_failure_head_frames = CLASSIFIER_FAILURE_HEAD_FRAMES
    classifier_ckpt_path = DEFAULT_CLASSIFIER_CKPT

    def get_environment(self, env_mode="real", classifier=False):
        if env_mode not in ("real", "virtual"):
            raise ValueError(f"env_mode must be 'real' or 'virtual', got {env_mode!r}")

        env = LocalOpenArmEnv(
            env_mode=env_mode,
            hz=self.hz,
            config=EnvConfig(),
            max_episode_length=int(round(self.max_episode_seconds * self.hz)),
        )

        env = DualRelativeFrame(env)
        env = Quat2EulerWrapper(env)
        if self.network_crop_primary:
            env = NetworkPrimaryImageCropWrapper(
                env,
                crop_ratio=self.network_primary_crop_ratio,
                y_offset_ratio=self.network_primary_crop_y_offset,
            )
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        if env_mode == "real":
            policy_obs_adapter = OpenArmPolicyObsAdapter(env)
            env = DualSpacemouseIntervention(
                env,
                control_hz=self.teleop_control_hz,
                policy_obs_adapter=policy_obs_adapter,
            )

        if classifier:
            # Load classifier and reward function from base class.
            # image_primary here is ALREADY center-cropped (wrapper above),
            # matching how the classifier was trained.
            reward_func = self.load_reward_classifier(env)
            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)

        env = RecordEpisodeStatistics(env)
        return env
