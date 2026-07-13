"""Reward wrappers used by OpenArm training tasks."""

import time

import gymnasium as gym


class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    """Use a vision classifier as binary reward."""

    def __init__(self, env, reward_classifier_func, target_hz=None):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.target_hz = target_hz

    def compute_reward(self, obs):
        if self.reward_classifier_func is not None:
            return self.reward_classifier_func(obs)
        return 0, 0.0

    def step(self, action):
        start_time = time.time()
        obs, rew, terminated, truncated, info = self.env.step(action)
        idle = bool(info.get("intervention_idle", False))
        result = (0, 0.0) if idle else self.compute_reward(obs)
        if isinstance(result, tuple):
            rew, classifier_prob = result
        else:
            rew, classifier_prob = result, float(bool(result))
        terminated = False if idle else bool(terminated or rew)
        info["succeed"] = bool(rew)
        info["classifier_prob"] = float(classifier_prob)
        if self.target_hz is not None:
            time.sleep(max(0, 1 / self.target_hz - (time.time() - start_time)))
        return obs, rew, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["succeed"] = False
        return obs, info
