"""Experiment registry mapping exp_name to its TrainConfig.

"""
from experiments.openarm_pickplace.config import TrainConfig as OpenArmPickPlaceTrainConfig

CONFIG_MAPPING = {
    "openarm_pickplace": OpenArmPickPlaceTrainConfig,
}
