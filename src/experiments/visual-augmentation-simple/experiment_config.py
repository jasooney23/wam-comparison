"""Shared physics settings for dataset generation and training."""

import math


def environment_settings(config, *, ood=False):
    """Build wrapper settings from the shared constants and parameter ranges.

    Args:
        config: Experiment configuration.
        ood: Whether to use evaluation ranges.

    Returns:
        Environment settings accepted by HistoryStackWrapper.
    """
    ranges = config["environment"]["ood"] if ood else config["train"]
    return {
        "pole_mass": config["environment"]["pole_mass"],
        "max_steps": config["environment"]["max_steps"],
        **{key: ranges[key] for key in ("cart_mass", "hinge_damping")},
    }


def validate_ranges(config):
    """Reject invalid ranges and any overlap between ID and OOD physics.

    Args:
        config: Experiment configuration.

    Raises:
        ValueError: A range is invalid or the ID and OOD boxes overlap.
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
    if all(
        max(train[key][0], test[key][0]) <= min(train[key][1], test[key][1])
        for key in ("cart_mass", "hinge_damping")
    ):
        raise ValueError("OOD ranges must be outside the training parameter ranges")
