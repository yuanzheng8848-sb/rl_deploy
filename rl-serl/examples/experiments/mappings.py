"""Experiment registry — maps exp_name to its TrainConfig.

Mirrors hil-serl/examples/experiments/mappings.py.
"""
from experiments.openarm_pickplace.config import TrainConfig as OpenArmPickPlaceTrainConfig

CONFIG_MAPPING = {
    "openarm_pickplace": OpenArmPickPlaceTrainConfig,
}
