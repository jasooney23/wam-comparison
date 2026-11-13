"""Train paired CartPole agents and save episode-based comparison curves."""

# General
import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import tomllib
from pathlib import Path
from time import perf_counter

# math & torch
import numpy as np
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch.utils.tensorboard import SummaryWriter
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

# Keep the experiment executable despite its deliberately hyphenated directory.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib"))
from common.compute import ComputeCounter
from common.environments.history import History, collect
from common.models import Agent


def validate(config):
    """Reject invalid modes, ranges, weights, and training sizes early."""
    if config["training"]["mode"] not in {"frozen", "alternating", "joint"}:
        raise ValueError("training.mode must be frozen, alternating, or joint")
    if config["accounting"]["scope"] not in {"neural", "broader_estimate"}:
        raise ValueError("accounting.scope must be neural or broader_estimate")
    if not config["run"]["modalities"] or any(
        value not in {"vector", "image", "combined"}
        for value in config["run"]["modalities"]
    ):
        raise ValueError("run.modalities must contain vector, image, or combined")
    if not config["run"]["seeds"] or len(set(config["run"]["seeds"])) != len(
        config["run"]["seeds"]
    ):
        raise ValueError("Provide distinct run seeds")
    if any(
        type(seed) is not int or not 0 <= seed < 2**32
        for seed in config["run"]["seeds"]
    ):
        raise ValueError("Seeds must be integers between zero and 2**32 - 1")
    if len(set(config["run"]["modalities"])) != len(config["run"]["modalities"]):
        raise ValueError("Provide distinct modalities")

    # Training sizes
    positive = {
        "run": ["cpu_threads"],
        "environment": ["pole_mass", "max_steps"],
        "training": [
            "episodes",
            "pretrain_transitions",
            "pretrain_epochs",
            "batch_size",
            "world_lr",
            "policy_lr",
            "world_updates",
        ],
        "ppo": ["rollout_steps", "minibatch_size", "epochs", "max_grad_norm", "clip"],
        "evaluation": ["interval", "episodes", "prediction_transitions"],
    }
    for section, keys in positive.items():
        for key in keys:
            value = config[section][key]
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{section}.{key} must be finite and positive")
            if (
                key
                not in {"pole_mass", "world_lr", "policy_lr", "max_grad_norm", "clip"}
                and type(value) is not int
            ):
                raise ValueError(f"{section}.{key} must be an integer")

    # Loss weights
    for key in ("vector_weight", "image_weight", "joint_weight"):
        value = config["training"][key]
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"training.{key} must be finite and nonnegative")
    if config["training"]["vector_weight"] <= 0:
        raise ValueError(
            "A positive vector_weight is required for the shared prediction target"
        )
    for key in ("gamma", "gae_lambda"):
        if not 0 <= config["ppo"][key] <= 1:
            raise ValueError(f"ppo.{key} must be between zero and one")
    for key in ("value_weight", "entropy_weight"):
        value = config["ppo"][key]
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"ppo.{key} must be finite and nonnegative")

    # Physics ranges
    for key in ("cart_mass", "hinge_damping"):
        bounds = config["environment"][key]
        if (
            len(bounds) != 2
            or not all(math.isfinite(x) for x in bounds)
            or bounds[0] > bounds[1]
        ):
            raise ValueError(f"Invalid environment.{key} range")
        if bounds[0] < 0 or (key == "cart_mass" and bounds[0] == 0):
            raise ValueError(f"Invalid environment.{key} range")
    for pair in config["evaluation"]["settings"]:
        if (
            len(pair) != 2
            or not all(math.isfinite(x) for x in pair)
            or pair[0] <= 0
            or pair[1] < 0
        ):
            raise ValueError(
                "Evaluation settings require positive mass and nonnegative damping"
            )
    if not config["evaluation"]["settings"]:
        raise ValueError("At least one evaluation setting is required")


def smoke_config(config):
    """Return a small real-training configuration for end-to-end checks."""
    config = copy.deepcopy(config)
    config["run"].update(seeds=[0])
    config["environment"]["max_steps"] = 30
    config["training"].update(
        episodes=2, pretrain_transitions=32, pretrain_epochs=1, batch_size=8
    )
    config["ppo"].update(rollout_steps=16, minibatch_size=8, epochs=1)
    config["evaluation"].update(interval=1, episodes=1, prediction_transitions=8)
    return config


def to_device(data, indices, device, images, counter=None):
    """Select a CPU batch and transfer only required modalities to the device."""
    start = perf_counter()
    result = TensorDict(
        {key: value for key, value in data.items() if images or "image" not in key},
        batch_size=[len(data["action"])],
    )[indices].to(device)
    if counter is not None and counter.enabled:
        counter.data_seconds += perf_counter() - start
    return result


def observation_batch(observation, device):
    """Add the batch dimension to a single environment observation."""
    return TensorDict(observation, batch_size=[], device=device).unsqueeze(0)


def ppo_modules(agent, settings):
    """Wrap shared networks in TorchRL's actor, critic, and PPO objective."""
    inputs = ["vector", "image"] if agent.world.has_image else ["vector"]
    actor = ProbabilisticActor(
        TensorDictModule(agent, in_keys=inputs, out_keys=["latent", "logits"]),
        in_keys=["logits"],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )
    critic = TensorDictModule(agent.value, in_keys=["latent"], out_keys=["state_value"])
    loss = ClipPPOLoss(
        actor,
        critic,
        functional=False,
        clip_epsilon=settings["clip"],
        critic_coeff=settings["value_weight"],
        loss_critic_type="l2",
        entropy_bonus=settings["entropy_weight"] > 0,
        entropy_coeff=settings["entropy_weight"],
        normalize_advantage=False,
    )
    return actor, critic, loss


def prediction_loss(agent, batch, training, counter):
    """Compute prediction loss and count normalization and MSE arithmetic."""
    loss, vector, image = agent.world.loss(
        batch, training["vector_weight"], training["image_weight"]
    )

    # Count loss operations
    n = batch["vector"].shape[0]
    counter.add(n * (24 * int(agent.world.has_vector) + 12))
    if agent.world.has_image:
        counter.add(n * (4 * 64 * 64 + 5 * 64 * 64))
    return loss, vector, image


def checked_step(loss, optimizer, parameters, counter, max_norm):
    """Backpropagate finite losses and fail on nonfinite gradients."""
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training loss")
    parameters = list(parameters)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
    counter.optimizer(parameters)
    optimizer.step()


def pretrain(agent, data, optimizer, config, counter, writer, device):
    """Train the full world model on the shared random-action dataset."""
    settings = config["training"]
    counter.phase = "pretrain"

    # Train each epoch
    for epoch in range(settings["pretrain_epochs"]):
        order = torch.randperm(len(data["action"]))
        total = 0.0
        for indices in order.split(settings["batch_size"]):
            batch = to_device(data, indices, device, agent.world.has_image, counter)
            loss, _, _ = prediction_loss(agent, batch, settings, counter)
            checked_step(
                loss,
                optimizer,
                agent.world.parameters(),
                counter,
                config["ppo"]["max_grad_norm"],
            )
            total += loss.item() * len(indices)
        writer.add_scalar("pretrain/loss", total / len(order), epoch + 1)


def rollout(agent, actor, critic, env, config, seed, episodes, device, counter):
    """Collect policy data, finishing at an episode boundary for exact resume."""
    settings = config["ppo"]
    target = config["training"]["episodes"]
    next_evaluation = (episodes // config["evaluation"]["interval"] + 1) * config[
        "evaluation"
    ]["interval"]

    # Rollout buffers
    transitions, scores = [], []
    observation, _ = env.reset(seed=seed * 100_000 + episodes)
    score = 0.0

    # Collect transitions
    while True:
        batch = observation_batch(observation, device)
        with torch.no_grad():
            critic(actor(batch))
        counter.add(
            40
            + 24 * int(agent.world.has_vector)
            + (4 * 64 * 64 if agent.world.has_image else 0)
        )
        future, reward, terminated, truncated, _ = env.step(batch["action"].item())
        ended = terminated or truncated
        with torch.no_grad():
            future_batch = observation_batch(future, device)
            # Bootstrap time limits
            if terminated:
                next_value = torch.zeros_like(batch["state_value"])
            else:
                latent = agent.world.encode(
                    future_batch["vector"], future_batch.get("image", None)
                )
                next_value = agent.value(latent)
                counter.add(
                    24 * int(agent.world.has_vector)
                    + (4 * 64 * 64 if agent.world.has_image else 0)
                )

        # Store transition
        transition = batch.select(
            *observation, "action", "action_log_prob", "state_value"
        )
        transition["next_vector"] = future_batch["vector"][:, -1]
        if agent.world.has_image:
            transition["next_image"] = future_batch["image"][:, -1]
        transition["next"] = TensorDict(
            {
                "reward": [[reward]],
                "done": [[ended]],
                "terminated": [[terminated]],
                "state_value": next_value,
            },
            batch_size=[1],
            device=device,
        )
        transitions.append(transition.cpu())
        score += reward
        observation = future
        if ended:
            episodes += 1
            scores.append(score)
            if len(transitions) >= settings["rollout_steps"] or episodes >= min(
                target, next_evaluation
            ):
                break
            observation, _ = env.reset(seed=seed * 100_000 + episodes)
            score = 0.0

    # Build training batch
    data = torch.cat(transitions)
    GAE(
        gamma=settings["gamma"],
        lmbda=settings["gae_lambda"],
        value_network=None,
        average_gae=len(transitions) > 1,
    )(data)
    counter.add(25 * len(transitions))
    return data, episodes, scores


def update(
    agent, data, policy_optimizer, world_optimizer, config, counter, device, ppo_loss
):
    """Apply alternating or joint representation updates and clipped PPO."""
    training, ppo = config["training"], config["ppo"]
    mode = training["mode"]
    counter.phase = "ppo"

    # Update world model
    if mode == "alternating":
        for _ in range(training["world_updates"]):
            for indices in torch.randperm(len(data["action"])).split(
                training["batch_size"]
            ):
                batch = to_device(data, indices, device, agent.world.has_image, counter)
                loss, _, _ = prediction_loss(agent, batch, training, counter)
                checked_step(
                    loss,
                    world_optimizer,
                    agent.world.parameters(),
                    counter,
                    ppo["max_grad_norm"],
                )

    # PPO minibatches
    parameters = [p for group in policy_optimizer.param_groups for p in group["params"]]
    for _ in range(ppo["epochs"]):
        for indices in torch.randperm(len(data["action"])).split(ppo["minibatch_size"]):
            batch = to_device(data, indices, device, agent.world.has_image, counter)

            # PPO and prediction losses
            losses = ppo_loss(batch)
            loss = sum(
                value for key, value in losses.items() if key.startswith("loss_")
            )
            if mode == "joint":
                model_loss, _, _ = prediction_loss(agent, batch, training, counter)
                loss = loss + training["joint_weight"] * model_loss
            counter.add(
                len(indices)
                * (
                    100
                    + 24 * int(agent.world.has_vector)
                    + (4 * 64 * 64 if agent.world.has_image else 0)
                )
            )
            checked_step(
                loss, policy_optimizer, parameters, counter, ppo["max_grad_norm"]
            )


def evaluate(agent, actor, held_out, config, seed, device, counter):
    """Measure deterministic returns and prediction errors without training FLOPs."""
    metrics = {}
    with (
        counter.excluded(),
        torch.no_grad(),
        set_exploration_type(ExplorationType.MODE),
    ):
        for number, setting in enumerate(config["evaluation"]["settings"]):
            label = "familiar" if number == 0 else f"heldout_{number}"
            env = History(
                config["environment"], images=agent.world.has_image, fixed=setting
            )

            # Evaluate control
            scores = []
            try:
                for episode in range(config["evaluation"]["episodes"]):
                    observation, _ = env.reset(seed=1_000_000 + seed * 1000 + episode)
                    ended, score = False, 0.0
                    while not ended:
                        batch = observation_batch(observation, device)
                        actor(batch)
                        observation, reward, terminated, truncated, _ = env.step(
                            batch["action"].item()
                        )
                        score += reward
                        ended = terminated or truncated
                    scores.append(score)
            finally:
                env.close()
            metrics[f"return_{label}"] = float(np.mean(scores))
            metrics[f"return_std_{label}"] = float(np.std(scores))

            # Evaluate predictions
            vector_total, image_total, count = 0.0, 0.0, 0
            data = held_out[number]
            for indices in torch.arange(len(data["action"])).split(
                config["training"]["batch_size"]
            ):
                batch = to_device(data, indices, device, agent.world.has_image)
                _, vector, image = agent.world.loss(batch)
                vector_total += vector.item() * len(indices)
                image_total += image.item() * len(indices)
                count += len(indices)
            metrics[f"vector_mse_{label}"] = vector_total / count
            if agent.world.has_image:
                metrics[f"image_mse_{label}"] = image_total / count
    return metrics


def shared_data(config, seed, directory):
    """Collect or reload exactly shared data, rejecting incompatible caches."""
    path = directory / "data.pt"
    signature = {
        "environment": config["environment"],
        "training_size": config["training"]["pretrain_transitions"],
        "evaluation": config["evaluation"],
        "seed": seed,
    }

    # Reuse cached data
    if path.exists():
        payload = torch.load(path, weights_only=False, map_location="cpu")
        if payload["signature"] != signature:
            raise ValueError(
                f"Data configuration changed; choose a new output directory: {path}"
            )
        return payload

    # Collect shared datasets
    data, metadata = collect(
        config["environment"], config["training"]["pretrain_transitions"], seed
    )
    held_out = [
        collect(
            config["environment"],
            config["evaluation"]["prediction_transitions"],
            2_000_000 + seed * 1000 + i,
            fixed=setting,
        )[0]
        for i, setting in enumerate(config["evaluation"]["settings"])
    ]
    payload = {
        "signature": signature,
        "training": data,
        "held_out": held_out,
        "collection": metadata,
    }
    torch.save(payload, path)
    return payload


def save_checkpoint(
    path,
    agent,
    policy_optimizer,
    world_optimizer,
    counter,
    config,
    seed,
    modality,
    episodes,
    transitions,
    updates,
    training_seconds,
    environment_times,
):
    """Atomically save training state at an episode boundary."""
    state = {
        "ppo_backend": "torchrl",
        "agent": agent.state_dict(),
        "policy_optimizer": policy_optimizer.state_dict(),
        "world_optimizer": world_optimizer.state_dict(),
        "counter": counter.state_dict(),
        "config": config,
        "seed": seed,
        "modality": modality,
        "episodes": episodes,
        "transitions": transitions,
        "updates": updates,
        "training_seconds": training_seconds,
        "environment_times": environment_times,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }

    # Replace atomically
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def plot_run(directory):
    """Plot raw scores, prediction errors, and operation totals against episodes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (directory / "metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    x = [int(row["episode"]) for row in rows]
    figure, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    for key in rows[0]:
        if key.startswith("return_") and not key.startswith("return_std_"):
            axes[0].plot(x, [float(row[key]) for row in rows], marker=".", label=key)
        if key.startswith("vector_mse_"):
            axes[1].plot(x, [float(row[key]) for row in rows], marker=".", label=key)
    for key in ("neural_flops", "estimated_total_operations"):
        axes[2].plot(x, [float(row[key]) for row in rows], marker=".", label=key)

    # Label and save
    for axis, label in zip(
        axes,
        ("Mean evaluation return", "Normalized vector MSE", "Cumulative operations"),
    ):
        axis.set_ylabel(label)
        axis.legend(fontsize="small")
        axis.grid(alpha=0.3)
    axes[2].set_xlabel("Completed PPO episodes")
    figure.tight_layout()
    figure.savefig(directory / "curves.png")
    plt.close(figure)


def plot_comparison(output, mode):
    """Overlay paired arms with means and seed spread at common episode points."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for modality in ("vector", "image", "combined"):
        runs = []
        for path in sorted(output.glob(f"seed-*/{mode}/{modality}/metrics.csv")):
            with path.open(newline="") as handle:
                runs.append(
                    {int(row["episode"]): row for row in csv.DictReader(handle)}
                )
        if not runs:
            continue

        # Aggregate seeds
        episodes = sorted(set.intersection(*(set(run) for run in runs)))
        for axis, key in zip(
            axes, ("return_familiar", "vector_mse_familiar", "selected_operations")
        ):
            values = np.array(
                [[float(run[episode][key]) for episode in episodes] for run in runs]
            )
            mean, spread = values.mean(0), values.std(0)
            axis.plot(episodes, mean, marker=".", label=f"{modality} (n={len(runs)})")
            axis.fill_between(episodes, mean - spread, mean + spread, alpha=0.15)
    for axis, label in zip(
        axes,
        (
            "Familiar evaluation return",
            "Familiar vector MSE",
            "Selected cumulative operations",
        ),
    ):
        axis.set_ylabel(label)
        axis.legend()
        axis.grid(alpha=0.3)
    axes[-1].set_xlabel("Completed PPO episodes")
    figure.tight_layout()
    figure.savefig(output / f"comparison-{mode}.png")
    plt.close(figure)


def run_arm(config, seed, modality, payload, directory, resume=None, gui=False):
    """Train one modality and seed, optionally continuing a saved checkpoint."""
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "metrics.csv").exists() and resume is None:
        raise FileExistsError(
            f"Run already exists: {directory}; use --resume or a new output"
        )

    # Initialize model
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(config["run"]["device"])
    agent = Agent(modality, train_encoder=config["training"]["mode"] == "joint").to(
        device
    )
    actor, critic, ppo_loss = ppo_modules(agent, config["ppo"])
    data = payload["training"]
    agent.world.mean.copy_(data["next_vector"].mean(0).to(device))
    agent.world.std.copy_(
        data["next_vector"].std(0, unbiased=False).clamp_min(1e-6).to(device)
    )

    # Configure optimizers
    training = config["training"]
    world_optimizer = torch.optim.Adam(
        agent.world.parameters(), lr=training["world_lr"]
    )
    policy_groups = [
        {
            "params": list(agent.policy.parameters()) + list(agent.value.parameters()),
            "lr": training["policy_lr"],
        }
    ]
    if training["mode"] == "joint":
        policy_groups.append(
            {"params": agent.world.parameters(), "lr": training["world_lr"]}
        )
    policy_optimizer = torch.optim.Adam(policy_groups)

    # Tracking and environment
    counter = ComputeCounter(agent)
    episodes, transitions, updates, training_seconds = 0, 0, 0, 0.0
    env = History(
        config["environment"],
        images=agent.world.has_image,
        counter=counter,
        gui=gui,
    )
    writer = SummaryWriter(str(directory / "tensorboard"))
    try:
        if resume:
            state = torch.load(resume, map_location="cpu", weights_only=False)
            if state.get("ppo_backend") != "torchrl":
                raise ValueError(
                    "Checkpoint uses the previous PPO backend; start a new TorchRL run"
                )
            original, current = copy.deepcopy(state["config"]), copy.deepcopy(config)
            original["training"].pop("episodes")
            current["training"].pop("episodes")
            if (
                original != current
                or state["seed"] != seed
                or state["modality"] != modality
            ):
                raise ValueError(
                    "Resume requires the same config, seed, and modality; only episode limit may change"
                )
            if training["episodes"] < state["episodes"]:
                raise ValueError("Episode limit is below the saved episode counter")
            agent.load_state_dict(state["agent"])
            policy_optimizer.load_state_dict(state["policy_optimizer"])
            world_optimizer.load_state_dict(state["world_optimizer"])
            counter.load_state_dict(state["counter"])
            episodes, transitions, updates = (
                state["episodes"],
                state["transitions"],
                state["updates"],
            )
            training_seconds = state["training_seconds"]
            env.times.update(state["environment_times"])
            torch.set_rng_state(state["torch_rng"])
            np.random.set_state(state["numpy_rng"])
            random.setstate(state["python_rng"])
            if state["cuda_rng"]:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
        else:
            counter.phase = "collection"
            counter.add(payload["collection"]["estimated_operations"], extra=True)
            start = perf_counter()
            pretrain(agent, data, world_optimizer, config, counter, writer, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            training_seconds += perf_counter() - start

        # Save run settings
        if training["mode"] == "frozen":
            agent.world.requires_grad_(False)
        (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (directory / "collection.json").write_text(
            json.dumps(payload["collection"], indent=2) + "\n"
        )
        csv_path = directory / "metrics.csv"
        # Drop any rows written after the last durable checkpoint on resume.
        if resume and csv_path.exists():
            with csv_path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                fields, rows = (
                    reader.fieldnames,
                    [row for row in reader if int(row["episode"]) <= episodes],
                )
            with csv_path.open("w", newline="") as handle:
                output = csv.DictWriter(handle, fieldnames=fields)
                output.writeheader()
                output.writerows(rows)

        def record(scores):
            metrics = evaluate(
                agent, actor, payload["held_out"], config, seed, device, counter
            )
            row = dict(
                episode=episodes,
                transitions=transitions,
                updates=updates,
                neural_flops=counter.neural,
                estimated_total_operations=counter.neural + counter.extra,
                selected_operations=counter.neural
                + (
                    counter.extra
                    if config["accounting"]["scope"] == "broader_estimate"
                    else 0
                ),
                training_seconds=training_seconds,
                training_return=float(np.mean(scores)) if scores else float("nan"),
                data_seconds=counter.data_seconds,
                **env.times,
                **metrics,
            )
            exists = csv_path.exists()
            with csv_path.open("a", newline="") as handle:
                output = csv.DictWriter(handle, fieldnames=list(row))
                if not exists:
                    output.writeheader()
                output.writerow(row)
            for key, value in row.items():
                writer.add_scalar(key, value, episodes)
            writer.flush()
            checkpoint = directory / f"episode-{episodes:06d}.pt"
            save_checkpoint(
                checkpoint,
                agent,
                policy_optimizer,
                world_optimizer,
                counter,
                config,
                seed,
                modality,
                episodes,
                transitions,
                updates,
                training_seconds,
                env.times,
            )
            print(
                f"{training['mode']} seed={seed} {modality}: episode={episodes} "
                f"return={metrics['return_familiar']:.1f} neural_FLOPs={counter.neural:,}",
                flush=True,
            )

        if resume is None:
            record([])

        # Train policy
        while episodes < training["episodes"]:
            start = perf_counter()
            counter.phase = "ppo"
            batch, episodes, scores = rollout(
                agent, actor, critic, env, config, seed, episodes, device, counter
            )
            update(
                agent,
                batch,
                policy_optimizer,
                world_optimizer,
                config,
                counter,
                device,
                ppo_loss,
            )
            transitions += len(batch["action"])
            updates += 1
            if device.type == "cuda":
                torch.cuda.synchronize()
            training_seconds += perf_counter() - start
            if (
                episodes % config["evaluation"]["interval"] == 0
                or episodes == training["episodes"]
            ):
                record(scores)
        plot_run(directory)
    finally:
        writer.close()
        env.close()
        counter.close()


def main():
    """Load TOML configuration and run the requested paired seeds."""

    # clargs
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.toml")
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--gui", action="store_true", help="Show the training CartPole environment"
    )
    parser.add_argument("--mode", choices=["frozen", "alternating", "joint"])
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume one trusted local checkpoint with the same config",
    )
    args = parser.parse_args()

    # validate clargs
    with args.config.open("rb") as handle:
        config = tomllib.load(handle)
    if args.smoke:
        config = smoke_config(config)
    if args.mode:
        config["training"]["mode"] = args.mode
    if args.output:
        config["run"]["output"] = str(args.output)
    validate(config)

    # torch & CUDA setup
    torch.set_num_threads(config["run"]["cpu_threads"])
    device = torch.device(config["run"]["device"])

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable; sync CUDA dependencies or explicitly configure device='cpu'"
        )

    if device.type == "cuda":
        # Fail before collecting data if this installation cannot execute CUDA kernels.
        probe = torch.ones(4, device=device, requires_grad=True)
        probe.square().sum().backward()

    output = Path(config["run"]["output"])
    pairs = [
        (seed, modality)
        for seed in config["run"]["seeds"]
        for modality in config["run"]["modalities"]
    ]

    # Load training checkpoint
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state.get("ppo_backend") != "torchrl":
            raise ValueError(
                "Checkpoint uses the previous PPO backend; start a new TorchRL run"
            )
        pairs = [(state["seed"], state["modality"])]
    payload_seed, payload = None, None

    # run training multiple times for each WAM config (different input modalities)
    for seed, modality in pairs:
        shared = output / f"seed-{seed}"
        shared.mkdir(parents=True, exist_ok=True)
        if seed != payload_seed:
            payload = shared_data(config, seed, shared)
            payload_seed = seed
        directory = shared / config["training"]["mode"] / modality
        if args.resume and args.resume.resolve().parent != directory.resolve():
            raise ValueError(
                "Resume checkpoint must belong to the configured output directory"
            )

        # Train!
        run_arm(config, seed, modality, payload, directory, args.resume, args.gui)

    plot_comparison(output, config["training"]["mode"])


if __name__ == "__main__":
    main()
