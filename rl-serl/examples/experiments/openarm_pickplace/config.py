"""OpenArm pick-and-place task config for rl-serl.

Mirrors hil-serl/examples/experiments/ram_insertion/config.py. Assembles the
wrapper stack (migrated OpenArm wrappers + serl_launcher generic wrappers) and,
when classifier=True, attaches the vision reward classifier via
MultiCameraBinaryRewardClassifierWrapper.

Reward design (see rl-serl/REFACTOR_PLAN.md §4):
  - single top camera (image_primary) only
  - classifier consumes the SAME center-cropped image_primary used by the policy
    (NetworkPrimaryImageCropWrapper runs BEFORE the classifier wrapper)
  - classifier_keys = ["image_primary"]
"""
from gymnasium.wrappers import RecordEpisodeStatistics

from openarm_env.envs.openarm_env import DefaultOpenArmConfig
from openarm_env.envs.local_openarm_env import LocalOpenArmEnv
from openarm_env.envs.wrappers import (
    DualRelativeFrame,
    Quat2EulerWrapper,
    NetworkPrimaryImageCropWrapper,
    GripperPenaltyWrapper,
    DualSpacemouseIntervention,
)
from franka_env.envs.wrappers import MultiCameraBinaryRewardClassifierWrapper

from rl_launcher.wrappers import SERLObsWrapper, ChunkingWrapper

from experiments.config import DefaultTrainingConfig
from experiments.artifacts import task_classifier_ckpt_dir


# Image streams fed to the policy encoder (three cameras).
TRAINING_IMAGE_KEYS = ["image_primary", "image_left", "image_right"]
PROPRIO_KEYS = ["tcp_pose", "tcp_vel", "gripper_pose"]
# Reward classifier uses only the top camera.
CLASSIFIER_KEYS = ["image_primary"]

# Default network-crop params (carried from rl_deploy/train.py FLAGS).
NETWORK_PRIMARY_CROP_RATIO = 0.3
NETWORK_PRIMARY_CROP_Y_OFFSET = 0.0

# Classifier configuration (used by both classifier training and online RL)
CLASSIFIER_THRESHOLD = 0.85
CLASSIFIER_SUCCESS_TAIL_FRAMES = 30
CLASSIFIER_FAILURE_HEAD_FRAMES = 30

# Default classifier checkpoint location under this task folder.
DEFAULT_CLASSIFIER_CKPT = str(task_classifier_ckpt_dir("openarm_pickplace"))


class EnvConfig(DefaultOpenArmConfig):
    """OpenArm bimanual env config: three local cameras."""

    REALSENSE_CAMERAS = {
        "image_primary": "local-head",
        "image_left": "local-left",
        "image_right": "local-right",
    }


class TrainConfig(DefaultTrainingConfig):
    image_keys = TRAINING_IMAGE_KEYS
    classifier_keys = CLASSIFIER_KEYS
    proprio_keys = PROPRIO_KEYS
    setup_mode = "dual-arm-learned-gripper"
    encoder_type = "resnet-pretrained"

    # teleop / servo params (carried from rl_deploy/train.py FLAGS defaults)
    network_crop_primary = True
    network_primary_crop_ratio = NETWORK_PRIMARY_CROP_RATIO
    network_primary_crop_y_offset = NETWORK_PRIMARY_CROP_Y_OFFSET
    grasp_penalty = 0.0

    # Classifier configuration (used by both training and online RL)
    classifier_threshold = CLASSIFIER_THRESHOLD
    classifier_success_tail_frames = CLASSIFIER_SUCCESS_TAIL_FRAMES
    classifier_failure_head_frames = CLASSIFIER_FAILURE_HEAD_FRAMES
    classifier_ckpt_path = DEFAULT_CLASSIFIER_CKPT

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = LocalOpenArmEnv(
            fake_env=fake_env,
            save_video=save_video,
            hz=self.hz,
            config=EnvConfig(),
            max_episode_length=self.max_traj_length,
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
        env = GripperPenaltyWrapper(env, penalty=self.grasp_penalty)

        # SpaceMouse intervention only on the real actor (not fake/learner).
        if not fake_env:
            env = DualSpacemouseIntervention(env)

        if classifier:
            # Load classifier and reward function from base class.
            # image_primary here is ALREADY center-cropped (wrapper above),
            # matching how the classifier was trained.
            reward_func = self.load_reward_classifier(env)
            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)

        env = RecordEpisodeStatistics(env)
        return env
