"""Generate a fixed dataset of synthesized CartPole transitions."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from cartpole_observation import CartPoleObservation
from experiment_config import environment_settings, load_config


def generate(count, seed):
    """Sample states and simulate one transition per state.

    Args:
        count: Number of pairs to generate.
        seed: Random sampling seed.

    Returns:
        Current observations, actions, and next observations as CPU tensors.
    """
    if count <= 0:
        raise ValueError("Dataset size must be positive")
    rng = np.random.default_rng(seed)
    env = CartPoleObservation()
    base = env.unwrapped
    limits = np.array([base.x_threshold, 2.0, base.theta_threshold_radians, 2.0])
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
            # Start each sample from an independently drawn state.
            env.reset()
            base.state = rng.uniform(-limits, limits)
            observation = env.observation(base.state)

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
    config = load_config(args.config, "pretrain", args.smoke)
    torch.set_num_threads(config["run"]["cpu_threads"])
    path = args.output or Path(config["pretrain"]["dataset"])
    if path.exists():
        raise FileExistsError(f"Dataset already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    # Generate
    train_seed, eval_seed = np.random.SeedSequence(args.seed).spawn(2)
    data = generate(config["pretrain"]["transitions"], train_seed)
    evaluation = generate(config["pretrain_eval"]["transitions"], eval_seed)

    # save
    print(f"Saving dataset to {path}...", flush=True)
    torch.save(
        {
            "environment": environment_settings(config),
            "training": data,
            "evaluation": evaluation,
        },
        path,
    )
    print(f"Saved {len(data['action']):,} samples to {path}")


if __name__ == "__main__":
    main()
