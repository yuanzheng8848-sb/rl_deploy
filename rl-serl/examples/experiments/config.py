"""Default training configuration base class for rl-serl experiments.

Mirrors hil-serl/examples/experiments/config.py (DefaultTrainingConfig). Each
task defines a subclass under experiments/<task>/config.py implementing
get_environment(). Defaults below carry over the values used by the original
rl_deploy/train.py FLAGS so behavior is preserved.
"""
import os
from abc import abstractmethod
from typing import List, Callable

import jax
import jax.numpy as jnp


class DefaultTrainingConfig:
    """Default training configuration."""

    # Bimanual hybrid SAC (learned gripper, dual arm) — OpenArm is always bimanual.
    agent: str = "sac"
    setup_mode: str = "dual-arm-learned-gripper"

    # carried over from rl_deploy/train.py FLAGS defaults
    max_traj_length: int = 400
    batch_size: int = 256
    discount: float = 0.97
    critic_actor_ratio: int = 4

    max_steps: int = 100000
    replay_buffer_capacity: int = 50000

    random_steps: int = 50
    training_starts: int = 300
    steps_per_update: int = 50

    log_period: int = 10
    checkpoint_period: int = 200

    encoder_type: str = "resnet-pretrained"
    hz: int = 5

    # image / proprio keys
    image_keys: List[str] = None       # streams fed to the policy encoder
    classifier_keys: List[str] = None  # streams fed to the reward classifier
    classifier_success_tail_frames: int = 30  # positive samples: last N frames of success trajectories
    classifier_failure_head_frames: int = 30  # negative samples: first N frames of success trajectories
    classifier_threshold: float = 0.85  # sigmoid probability threshold for binary classification
    classifier_ckpt_path: str = None    # path to classifier checkpoint (set by subclass)
    proprio_keys: List[str] = None

    def load_reward_classifier(self, env) -> Callable:
        """Load the reward classifier and return a reward function.

        This method provides a unified interface for loading classifiers across all tasks.
        Subclasses can override this if they need custom behavior, but the default
        implementation should work for most cases.

        Args:
            env: The environment after all wrappers (including image crop) are applied.
                 The classifier will use env.observation_space to determine input shapes.

        Returns:
            A reward_func(obs) -> int that can be passed to
            MultiCameraBinaryRewardClassifierWrapper.
        """
        from rl_launcher.networks import load_classifier_func

        if not self.classifier_keys:
            raise ValueError(
                f"{self.__class__.__name__} must set classifier_keys to use reward classifier"
            )
        if not self.classifier_ckpt_path:
            raise ValueError(
                f"{self.__class__.__name__} must set classifier_ckpt_path to use reward classifier"
            )

        # Load classifier
        clf = load_classifier_func(
            key=jax.random.PRNGKey(0),
            sample=env.observation_space.sample(),
            image_keys=self.classifier_keys,
            checkpoint_path=os.path.abspath(self.classifier_ckpt_path),
        )

        threshold = self.classifier_threshold

        def reward_func(obs):
            """Binary reward function using the trained classifier.

            Returns 1 if the classifier predicts success (prob > threshold), 0 otherwise.
            The logits are squeezed to scalar to ensure int() conversion works.
            """
            sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
            logits = clf(obs)
            # Ensure scalar for int() conversion: (1, 1) or (1,) → scalar
            if logits.ndim > 0:
                logits = logits.squeeze()
            prob = sigmoid(logits)
            return int(prob > threshold)

        return reward_func

    @abstractmethod
    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        raise NotImplementedError
