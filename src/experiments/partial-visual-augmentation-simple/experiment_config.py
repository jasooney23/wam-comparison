"""Shared environment settings for dataset generation and training."""

import copy
import math
import sys
import tomllib
from pathlib import Path

import gymnasium as gym

# Allow the experiment scripts to run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.common.environments.cartpole.observations import observation_layout
from src.common.models.world_model import WorldModel

ENV_ID = "CartPole-v1"


def build_world_model(modality, settings):
    """Build a world model with about 28,600 parameters for either modality."""
    if modality == "vector":
        return WorldModel(modality, settings, hidden_width=388)
    return WorldModel(modality, settings, image_channels=8)


def environment_settings(config):
    """Return the environment ID and observation settings for artifact checks."""
    return {"id": ENV_ID, **config["environment"]}


def validate(config, stage):
    """Reject invalid observations, weights, and training sizes early."""
    if observation_layout(config["environment"]) != (1, [0, 2]):
        raise ValueError("Expected one frame with cart position and pole angle")
    seeds = config[stage]["seeds"]
    if not seeds or any(
        type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds
    ):
        raise ValueError(
            f"{stage}.seeds must contain integers between zero and 2**32 - 1"
        )
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{stage}.seeds must be distinct")

    # Training sizes
    positive = {
        "run": ["cpu_threads"],
        "pretrain_eval": ["batch_size"],
    }
    if stage == "pretrain":
        positive.update(
            pretrain=[
                "transitions",
                "epochs",
                "eval_every",
                "batch_size",
                "world_lr",
                "max_grad_norm",
                "vector_weight",
                "image_weight",
            ],
            pretrain_eval=["transitions", "batch_size"],
        )
    else:
        positive.update(
            train=["episodes", "policy_lr"],
            ppo=["rollout_steps", "minibatch_size", "epochs", "max_grad_norm", "clip"],
            train_eval=["interval", "eval_episodes"],
        )
    for section, keys in positive.items():
        for key in keys:
            value = config[section][key]
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{section}.{key} must be finite and positive")
            if (
                key
                not in {
                    "world_lr",
                    "policy_lr",
                    "max_grad_norm",
                    "clip",
                    "vector_weight",
                    "image_weight",
                }
                and type(value) is not int
            ):
                raise ValueError(f"{section}.{key} must be an integer")

    if stage == "train":
        for key in ("gamma", "gae_lambda"):
            if not 0 <= config["ppo"][key] <= 1:
                raise ValueError(f"ppo.{key} must be between zero and one")
        for key in ("value_weight", "entropy_weight"):
            value = config["ppo"][key]
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"ppo.{key} must be finite and nonnegative")

    if stage != "train":
        return
    horizon = config["train_eval"]["horizon"]
    if (
        type(horizon) is not int
        or not 0 <= horizon < gym.spec(ENV_ID).max_episode_steps
    ):
        raise ValueError("train_eval.horizon must be an integer below max_steps")


def smoke_config(config):
    """Return a small real-training configuration for end-to-end checks."""
    config = copy.deepcopy(config)
    config["pretrain"]["seeds"] = [0]
    config["train"]["seeds"] = [0]
    config["pretrain"].update(transitions=32, epochs=1, batch_size=8)
    config["pretrain_eval"]["transitions"] = 16
    config["train"]["episodes"] = 2
    config["ppo"].update(rollout_steps=16, minibatch_size=8, epochs=1)
    config["train_eval"].update(interval=1, eval_episodes=1)
    return config


def load_config(path, stage, smoke=False):
    """Read and validate the configuration for one training stage."""
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    if smoke:
        config = smoke_config(config)
    validate(config, stage)
    return config
