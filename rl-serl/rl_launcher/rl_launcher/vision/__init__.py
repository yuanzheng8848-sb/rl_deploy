from rl_launcher.vision.resnet_v1 import resnetv1_configs
from rl_launcher.vision.data_augmentations import batched_random_crop

encoders = dict()
encoders.update(resnetv1_configs)

__all__ = ["resnetv1_configs", "encoders", "batched_random_crop"]
