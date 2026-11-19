"""Shared environment settings for dataset generation and training."""

import copy
import math
import sys
import tomllib
from pathlib import Path

# Allow the experiment scripts to run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.common.environments.cartpole.observations import (
    OBSERVATION_DEFAULTS,
    observation_layout,
)


def environment_settings(config, *, ood=False):
    """Build wrapper settings from the shared constants and parameter ranges.

    Args:
        config: Experiment configuration.
        ood: Whether to use evaluation ranges minus the training ranges.

    Returns:
        Environment settings accepted by HistoryStackWrapper.
    """
    environment = config["environment"]
    ranges = environment["ood"] if ood else config["train"]
    result = {
        "pole_mass": environment["pole_mass"],
        "max_steps": environment["max_steps"],
        **{
            key: environment.get(key, default)
            for key, default in OBSERVATION_DEFAULTS.items()
        },
        **{key: ranges[key] for key in ("cart_mass", "hinge_damping")},
    }
    observation_layout(result)
    if ood:
        result["excluded_ranges"] = {
            key: config["train"][key] for key in ("cart_mass", "hinge_damping")
        }
    return result


def matches_environment(saved, expected):
    """Check saved settings with observation defaults for older artifacts.

    Args:
        saved: Environment settings saved in the artifact.
        expected: Environment settings for this run.

    Returns:
        Whether the saved settings match this run.
    """
    return isinstance(saved, dict) and (OBSERVATION_DEFAULTS | saved) == expected


def validate_ranges(config):
    """Reject invalid ranges and empty OOD ranges after excluding training.

    Args:
        config: Experiment configuration.

    Raises:
        ValueError: A range is invalid or has no values outside training.
    """
    train = environment_settings(config)
    test = environment_settings(config, ood=True)
    for ranges in (train, test):
        for key in ("cart_mass", "hinge_damping"):
            bounds = ranges[key]
            if (
                len(bounds) != 2
                or not all(math.isfinite(x) for x in bounds)
                or bounds[0] > bounds[1]
                or bounds[0] < 0
                or (key == "cart_mass" and bounds[0] == 0)
            ):
                raise ValueError(f"Invalid {key} range")
    for key in ("cart_mass", "hinge_damping"):
        if train[key][0] <= test[key][0] <= test[key][1] <= train[key][1]:
            raise ValueError(f"OOD {key} has no values outside the training range")


def validate(config, stage):
    """Reject invalid ranges, weights, and training sizes early."""
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
        "environment": ["pole_mass", "max_steps"],
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
                    "pole_mass",
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

    # Shared ID and OOD ranges
    validate_ranges(config)
    if stage != "train":
        return
    horizon = config["train_eval"]["horizon"]
    if (
        type(horizon) is not int
        or not 0 <= horizon < config["environment"]["max_steps"]
    ):
        raise ValueError("train_eval.horizon must be an integer below max_steps")


def smoke_config(config):
    """Return a small real-training configuration for end-to-end checks."""
    config = copy.deepcopy(config)
    config["pretrain"]["seeds"] = [0]
    config["train"]["seeds"] = [0]
    frames, _ = observation_layout(config["environment"])
    config["environment"]["max_steps"] = max(30, frames)
    config["pretrain"].update(transitions=32, epochs=1, batch_size=8)
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
