"""Generate a fixed dataset of synthesized CartPole transitions."""

import argparse
import sys
import tomllib
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_config import environment_settings, validate_ranges

from src.common.environments.cartpole.observations import observation_layout
from src.common.environments.history_stack_wrapper import HistoryStackWrapper


def generate(env_settings, count, seed, fixed_params=None):
    """Sample states and simulate observation histories and one target step.

    Args:
        env_settings: Environment configuration.
        count: Number of pairs to generate.
        seed: Random sampling seed.
        fixed_params: Optional OOD cart mass and hinge damping.

    Returns:
        Observation histories, actions, and next frames as CPU tensors.
    """
    frames, _ = observation_layout(env_settings)
    if count <= 0 or env_settings["max_steps"] < frames:
        raise ValueError(
            "Dataset size must be positive and max_steps must be at least history_frames"
        )
    rng = np.random.default_rng(seed)
    env = HistoryStackWrapper(env_settings, images=True, fixed_params=fixed_params)
    vector_space = env.observation_space["vector"]
    image_space = env.observation_space["image"]
    arrays = {
        "vector": np.empty((count, *vector_space.shape), vector_space.dtype),
        "image": np.empty((count, *image_space.shape), image_space.dtype),
        "action": np.empty(count, np.int64),
        "next_vector": np.empty((count, *vector_space.shape[1:]), vector_space.dtype),
        "next_image": np.empty((count, *image_space.shape[1:]), image_space.dtype),
    }
    start = perf_counter()
    last_report = start
    print(f"Generating {count:,} samples...", flush=True)
    try:
        for i in range(count):
            # Reject episodes that end before the target step.
            while True:
                env.reset(seed=int(rng.integers(2**32)))
                base = env.unwrapped
                limits = np.array(
                    [base.x_threshold, 2.0, base.theta_threshold_radians, 2.0]
                )
                base.state = rng.uniform(-limits, limits)
                observation = env._observe(base.state, initial=True)

                # Fill the history with random actions.
                for _ in range(frames - 1):
                    observation, _, terminated, truncated, _ = env.step(
                        int(rng.integers(2))
                    )
                    if terminated or truncated:
                        break

                else:
                    break

            # Generate transition state
            action = int(rng.integers(2))
            future, _, _, _, _ = env.step(action)
            for key in ("vector", "image"):
                arrays[key][i] = observation[key]
                arrays[f"next_{key}"][i] = future[key][-1]
            arrays["action"][i] = action
            now = perf_counter()

            # print info
            if now - last_report >= 5 or i + 1 == count:
                print(
                    f"Generated {i + 1:,}/{count:,} samples "
                    f"({(i + 1) / count:.1%}); elapsed {now - start:.0f}s",
                    flush=True,
                )
                last_report = now
    finally:
        env.close()
    return {key: torch.from_numpy(value) for key, value in arrays.items()}


def main():
    """Generate the configured dataset without overwriting an existing file."""

    # clargs
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.toml")
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    with args.config.open("rb") as handle:
        config = tomllib.load(handle)
    if args.smoke:
        frames, _ = observation_layout(config["environment"])
        config["environment"]["max_steps"] = max(30, frames)
        config["pretrain"]["transitions"] = 32
    torch.set_num_threads(config["run"]["cpu_threads"])
    validate_ranges(config)
    path = args.output or Path(config["pretrain"]["dataset"])
    if path.exists():
        raise FileExistsError(f"Dataset already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    # Generate
    train_settings = environment_settings(config)
    test_settings = environment_settings(config, ood=True)
    data = generate(train_settings, config["pretrain"]["transitions"], args.seed)
    evaluation = generate(
        test_settings, config["pretrain_eval"]["transitions"], args.seed + 1
    )

    # save
    print(f"Saving dataset to {path}...", flush=True)
    torch.save(
        {
            "environment": train_settings,
            "training": data,
            "evaluation": evaluation,
            "evaluation_environment": test_settings,
        },
        path,
    )
    print(f"Saved {len(data['action']):,} samples to {path}")


if __name__ == "__main__":
    main()
