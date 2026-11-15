"""Train paired CartPole agents and compare held-out results across training."""

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
from common.environments.history_stack_wrapper import (
    HistoryStackWrapper,
    random_rollouts,
)
from common.models import Agent

EXPERIMENT_VERSION = 9

STATE_NAMES = ("cart_position", "cart_velocity", "pole_angle")
EVALUATION_METRICS = ("return_ood", *(f"{name}_l2_ood" for name in STATE_NAMES))


def evaluation_metrics(horizon):
    """List the return and vector errors recorded for a rollout horizon."""
    if horizon == 0:
        return EVALUATION_METRICS
    return (
        "return_ood",
        *(
            f"{name}_l2_h{step}_ood"
            for name in STATE_NAMES
            for step in range(horizon + 1)
        ),
    )


def evaluation_metrics_from_rows(rows):
    """Find the metric columns in an evaluation CSV row."""
    keys = rows[0]
    if "cart_position_l2_ood" in keys:
        return EVALUATION_METRICS
    horizons = [
        int(key.rsplit("_h", 1)[1].split("_", 1)[0])
        for key in keys
        if key.startswith("cart_position_l2_h") and key.endswith("_ood")
    ]
    return evaluation_metrics(max(horizons))


def validate(config):
    """Reject invalid ranges, weights, and training sizes early."""
    if not config["run"]["modalities"] or any(
        value not in {"vector", "combined"} for value in config["run"]["modalities"]
    ):
        raise ValueError("run.modalities must contain vector or combined")
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
            "pretrain_transitions",
            "pretrain_epochs",
            "batch_size",
            "world_lr",
            "policy_lr",
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

    if (
        type(config["training"]["episodes"]) is not int
        or config["training"]["episodes"] <= 0
    ):
        raise ValueError("training.episodes must be a positive integer")

    # Loss weights
    for key in ("vector_weight", "image_weight"):
        value = config["training"][key]
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"training.{key} must be finite and positive")
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
        if all(
            config["environment"][key][0] <= value <= config["environment"][key][1]
            for key, value in zip(("cart_mass", "hinge_damping"), pair)
        ):
            raise ValueError(
                "Evaluation settings must be outside the training parameter ranges"
            )
    if not config["evaluation"]["settings"]:
        raise ValueError("At least one evaluation setting is required")
    horizon = config["evaluation"]["horizon"]
    if (
        type(horizon) is not int
        or not 0 <= horizon < config["environment"]["max_steps"]
    ):
        raise ValueError("evaluation.horizon must be an integer below max_steps")


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


def to_device(data, indices, device, world):
    """Select a CPU batch and transfer only required modalities to the device."""
    result = TensorDict(
        {
            key: value
            for key, value in data.items()
            if world.has_image or "image" not in key
        },
        batch_size=[len(data["vector"])],
    )[indices].to(device)
    return result


def observation_batch(observation, device):
    """Add the batch dimension to a single environment observation."""
    return TensorDict(observation, batch_size=[], device=device).unsqueeze(0)


def ppo_modules(agent, settings):
    """Wrap shared networks in TorchRL's actor, critic, and PPO objective."""
    inputs = {"vector": "vector"}
    if agent.world.has_image:
        inputs["image"] = "image"
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


def checked_step(loss, optimizer, parameters, max_norm):
    """Backpropagate finite losses and fail on nonfinite gradients."""
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training loss")
    parameters = list(parameters)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
    optimizer.step()


def pretrain(
    agent,
    data,
    optimizer,
    config,
    counter,
    writer,
    device,
    directory,
    seed,
    modality,
    held_out,
):
    """Train the full world model on the fixed synthesized dataset."""
    settings = config["training"]
    evaluations = []
    print(
        f"Pretraining world model: seed={seed} modality={modality}, "
        f"{settings['pretrain_epochs']} epochs",
        flush=True,
    )

    # Train each epoch
    for epoch in range(settings["pretrain_epochs"]):
        start = perf_counter()
        order = torch.randperm(len(data["action"]))
        total = 0.0
        for indices in order.split(settings["batch_size"]):
            batch = to_device(data, indices, device, agent.world)
            loss, _ = agent.world.loss(
                batch, settings["vector_weight"], settings["image_weight"]
            )
            checked_step(
                loss,
                optimizer,
                agent.world.parameters(),
                config["ppo"]["max_grad_norm"],
            )
            counter["world_updates"] += 1
            total += loss.item() * len(indices)
        if device.type == "cuda":
            torch.cuda.synchronize()
        counter["world_training_seconds"] += perf_counter() - start
        writer.add_scalar("pretrain/loss", total / len(order), epoch + 1)
        print(
            f"World model seed={seed} {modality}: "
            f"epoch {epoch + 1}/{settings['pretrain_epochs']}, "
            f"loss={total / len(order):.6f}, updates={counter['world_updates']:,}",
            flush=True,
        )
        if (epoch + 1) % 20 == 0 or epoch + 1 == settings["pretrain_epochs"]:
            metrics = evaluate_world(agent.world, held_out, config, device)
            evaluations.append(
                dict(
                    epoch=epoch + 1,
                    world_updates=counter["world_updates"],
                    world_training_seconds=counter["world_training_seconds"],
                    **metrics,
                )
            )
            with (directory / "world-metrics.csv").open("w", newline="") as handle:
                output = csv.DictWriter(handle, fieldnames=list(evaluations[0]))
                output.writeheader()
                output.writerows(evaluations)
            for key, value in metrics.items():
                writer.add_scalar(f"world_evaluation/{key}", value, epoch + 1)
            torch.save(
                {
                    "experiment_version": EXPERIMENT_VERSION,
                    "world": agent.world.state_dict(),
                    "modality": modality,
                    "seed": seed,
                    "epoch": epoch + 1,
                    "world_updates": counter["world_updates"],
                    "world_training_seconds": counter["world_training_seconds"],
                    "environment": config["environment"],
                },
                directory / f"world-epoch-{epoch + 1:03d}.pt",
            )


def rollout(agent, actor, critic, env, config, seed, episodes, device):
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

        # Store transition
        transition = batch.select(
            *observation, "action", "action_log_prob", "state_value"
        )
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
    return data, episodes, scores


def update(agent, data, policy_optimizer, config, counter, device, ppo_loss):
    """Update the policy and value heads using PPO."""
    ppo = config["ppo"]

    # PPO minibatches
    parameters = [p for group in policy_optimizer.param_groups for p in group["params"]]
    for _ in range(ppo["epochs"]):
        for indices in torch.randperm(len(data["action"])).split(ppo["minibatch_size"]):
            batch = to_device(data, indices, device, agent.world)

            losses = ppo_loss(batch)
            loss = sum(
                value for key, value in losses.items() if key.startswith("loss_")
            )
            checked_step(loss, policy_optimizer, parameters, ppo["max_grad_norm"])
            counter["agent_updates"] += 1


@torch.no_grad()
def evaluate_world(world, held_out, config, device):
    """Measure reconstruction and open-loop vector errors on shared OOD data.

    Args:
        world: World model to evaluate.
        held_out: One fixed dataset for each OOD parameter setting.
        config: Experiment configuration.
        device: Device used by the world model.

    Returns:
        Per-variable scalar L2 errors for each setting and their uniform mean.
    """
    metrics = {}
    horizon = config["evaluation"]["horizon"]
    for number, data in enumerate(held_out):
        totals = torch.zeros((horizon + 1, len(STATE_NAMES)), device=device)
        count = 0
        for indices in torch.arange(len(data["vector"])).split(
            config["training"]["batch_size"]
        ):
            batch = to_device(data, indices, device, world)
            latent = world.encode(batch["vector"], batch.get("image"))
            for step in range(horizon + 1):
                if step:
                    latent = world.predict(latent, batch["actions"][:, step - 1])
                prediction = world.decode(latent)["vector"][:, -1]
                prediction = prediction * world.std + world.mean
                target = (
                    batch["targets"][:, step] if horizon else batch["vector"][:, -1]
                )
                totals[step] += (prediction - target).abs().sum(0)
            count += len(indices)
        for step, errors in enumerate(totals / count):
            for name, error in zip(STATE_NAMES, errors):
                suffix = f"_h{step}" if horizon else ""
                metrics[f"{name}_l2{suffix}_heldout_{number}"] = error.item()

    # Average the equally sized datasets across OOD settings.
    for key in evaluation_metrics(horizon)[1:]:
        prefix = key.removesuffix("_ood")
        metrics[key] = float(
            np.mean([metrics[f"{prefix}_heldout_{i}"] for i in range(len(held_out))])
        )
    return metrics


def evaluate(agent, actor, held_out, config, seed, device):
    """Measure OOD return and vector rollout errors without training."""
    metrics = evaluate_world(agent.world, held_out, config, device)
    with (
        torch.no_grad(),
        set_exploration_type(ExplorationType.MODE),
    ):
        for number, setting in enumerate(config["evaluation"]["settings"]):
            label = f"heldout_{number}"
            env = HistoryStackWrapper(
                config["environment"],
                images=agent.world.has_image,
                fixed_params=setting,
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

    # Each OOD setting has the same number of control episodes.
    metrics["return_ood"] = float(
        np.mean([metrics[f"return_heldout_{i}"] for i in range(len(held_out))])
    )
    return metrics


def shared_data(config, seed, directory, *, load_training=True):
    """Load shared OOD data and, when needed, the fixed training dataset.

    Args:
        config: Experiment configuration.
        seed: Paired run seed.
        directory: Directory containing the held-out data cache.
        load_training: Whether this run will pretrain a world model.

    Returns:
        Held-out data and its signature, plus training data when requested.
    """
    path = directory / "data.pt"
    signature = {
        "experiment_version": EXPERIMENT_VERSION,
        "environment": config["environment"],
        "evaluation": config["evaluation"],
        "seed": seed,
    }

    # Checkpoint-only runs already contain their normalization buffers.
    training = {}
    if load_training:
        dataset_path = Path(config["training"]["dataset"])
        dataset_stat = dataset_path.stat()
        signature.update(
            training_size=config["training"]["pretrain_transitions"],
            dataset=str(dataset_path.resolve()),
            dataset_size=dataset_stat.st_size,
            dataset_modified=dataset_stat.st_mtime_ns,
        )
        dataset = torch.load(dataset_path, weights_only=False, map_location="cpu")
        if dataset["environment"] != config["environment"]:
            raise ValueError(
                "Dataset environment does not match the training configuration"
            )
        data = dataset["training"]
        if len(data["action"]) != config["training"]["pretrain_transitions"]:
            raise ValueError(
                "Dataset size does not match training.pretrain_transitions"
            )
        training["training"] = data

    # Cache only held-out data; the training dataset is shared by all seeds.
    if path.exists():
        payload = torch.load(path, weights_only=False, map_location="cpu")
        cached_signature = payload["signature"]
        if not load_training:
            cached_signature = {
                key: value
                for key, value in cached_signature.items()
                if key in signature
            }
        if cached_signature != signature:
            raise ValueError(
                f"Data configuration changed; choose a new output directory: {path}"
            )
        return {**payload, **training}

    held_out = [
        random_rollouts(
            config["environment"],
            config["evaluation"]["prediction_transitions"],
            2_000_000 + seed * 1000 + i,
            fixed_params=setting,
            horizon=config["evaluation"]["horizon"],
        )
        for i, setting in enumerate(config["evaluation"]["settings"])
    ]
    payload = {
        "signature": signature,
        "held_out": held_out,
    }
    torch.save(payload, path)
    return {**payload, **training}


def load_checkpoint(path):
    """Load a trusted checkpoint and reject older experiment definitions."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("ppo_backend") != "torchrl":
        raise ValueError(
            "Checkpoint uses the previous PPO backend; start a new TorchRL run"
        )
    if state.get("experiment_version") != EXPERIMENT_VERSION:
        raise ValueError(
            "Checkpoint uses a previous experiment definition; start a new run"
        )
    return state


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
    agent_environment_steps,
):
    """Atomically save training state at an episode boundary."""
    state = {
        "experiment_version": EXPERIMENT_VERSION,
        "ppo_backend": "torchrl",
        "agent": agent.state_dict(),
        "policy_optimizer": policy_optimizer.state_dict(),
        "world_optimizer": world_optimizer.state_dict(),
        "counter": dict(counter),
        "config": config,
        "seed": seed,
        "modality": modality,
        "episodes": episodes,
        "agent_environment_steps": agent_environment_steps,
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
    """Plot raw scores, prediction errors, and training counts against episodes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (directory / "metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    x = [int(row["episode"]) for row in rows]
    figure, axes = plt.subplots(5, 1, figsize=(9, 15), sharex=True)
    for key in rows[0]:
        if key.startswith("return_") and not key.startswith("return_std_"):
            axes[0].plot(x, [float(row[key]) for row in rows], marker=".", label=key)
        for axis, name in zip(axes[1:4], STATE_NAMES):
            if key == f"{name}_l2_ood" or (
                key.startswith(f"{name}_l2_h") and key.endswith("_ood")
            ):
                axis.plot(x, [float(row[key]) for row in rows], marker=".", label=key)
    for key in ("agent_environment_steps", "world_updates", "agent_updates"):
        axes[4].plot(x, [float(row[key]) for row in rows], marker=".", label=key)

    # Label and save
    for axis, label in zip(
        axes,
        (
            "Mean evaluation return",
            "Cart position L2",
            "Cart velocity L2",
            "Pole angle L2",
            "Cumulative counts",
        ),
    ):
        axis.set_ylabel(label)
        axis.legend(fontsize="small")
        axis.grid(alpha=0.3)
    axes[4].set_xlabel("Completed PPO episodes")
    figure.tight_layout()
    figure.savefig(directory / "curves.png")
    plt.close(figure)


def match_compute(runs, budget="agent_environment_steps"):
    """Interpolate metrics within all runs' shared compute range.

    Args:
        runs: Mapping from modality to lists of CSV row lists, one per seed.
        budget: Increasing compute column used to align runs.

    Returns:
        Rows with metric means and seed spread at common compute values.
        An empty list means the runs have no shared compute range yet.
    """
    series = [rows for seeds in runs.values() for rows in seeds]
    if not series:
        return []
    lower = max(float(rows[0][budget]) for rows in series)
    upper = min(float(rows[-1][budget]) for rows in series)
    if lower > upper:
        return []
    grid = np.unique(
        [
            lower,
            upper,
            *(
                float(row[budget])
                for rows in series
                for row in rows
                if lower <= float(row[budget]) <= upper
            ),
        ]
    )

    # Interpolate each seed before taking means; never extrapolate a curve.
    result = []
    for modality, seeds in runs.items():
        for metric in evaluation_metrics_from_rows(seeds[0]):
            values = np.array(
                [
                    np.interp(
                        grid,
                        [float(row[budget]) for row in rows],
                        [float(row[metric]) for row in rows],
                    )
                    for rows in seeds
                ]
            )
            for cost, mean, spread in zip(grid, values.mean(0), values.std(0)):
                result.append(
                    {
                        "modality": modality,
                        "metric": metric,
                        budget: cost,
                        "mean": mean,
                        "std": spread,
                        "seeds": len(seeds),
                        "method": "linear_interpolation",
                    }
                )
    return result


def plot_comparison(output, budget="agent_environment_steps"):
    """Save OOD comparison curves at common steps or total training time.

    Args:
        output: Directory containing paired runs.
        budget: Agent environment steps or total training seconds.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = {}
    for modality in ("vector", "combined"):
        for path in sorted(output.glob(f"seed-*/{modality}/metrics.csv")):
            with path.open(newline="") as handle:
                runs.setdefault(modality, []).append(list(csv.DictReader(handle)))
    label = {
        "agent_environment_steps": "Agent environment steps",
        "total_training_seconds": "Total training seconds (world model + agent)",
    }[budget]
    filename = (
        "comparison" if budget == "agent_environment_steps" else "comparison-time"
    )
    matched = match_compute(runs, budget)
    with (output / f"{filename}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "modality",
                "metric",
                budget,
                "mean",
                "std",
                "seeds",
                "method",
            ],
        )
        writer.writeheader()
        writer.writerows(matched)

    figure, axes = plt.subplots(4, 1, figsize=(9, 13), sharex=True)
    for modality in runs:
        for metric in dict.fromkeys(
            row["metric"] for row in matched if row["modality"] == modality
        ):
            axis = (
                axes[0]
                if metric == "return_ood"
                else next(
                    axes[index + 1]
                    for index, name in enumerate(STATE_NAMES)
                    if metric.startswith(f"{name}_l2")
                )
            )
            rows = [
                row
                for row in matched
                if row["modality"] == modality and row["metric"] == metric
            ]
            if not rows:
                continue
            x = np.array([row[budget] for row in rows])
            mean = np.array([row["mean"] for row in rows])
            spread = np.array([row["std"] for row in rows])
            axis.plot(
                x, mean, marker=".", label=f"{modality} {metric} (n={rows[0]['seeds']})"
            )
            axis.fill_between(x, mean - spread, mean + spread, alpha=0.15)
    for axis, label in zip(
        axes,
        (
            "Mean OOD return",
            "OOD cart position L2",
            "OOD cart velocity L2",
            "OOD pole angle L2",
        ),
    ):
        axis.set_ylabel(label)
        if matched:
            axis.legend()
        else:
            axis.text(
                0.5,
                0.5,
                "No shared compute range yet",
                ha="center",
                transform=axis.transAxes,
            )
        axis.grid(alpha=0.3)
    axes[-1].set_xlabel(f"{label} (interpolated within shared range)")
    figure.tight_layout()
    figure.savefig(output / f"{filename}.png")
    plt.close(figure)


def run_arm(
    config,
    seed,
    modality,
    payload,
    directory,
    resume=None,
    gui=False,
    world_checkpoint=None,
):
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
    agent = Agent(modality).to(device)
    actor, critic, ppo_loss = ppo_modules(agent, config["ppo"])
    if not resume and not world_checkpoint:
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
    policy_optimizer = torch.optim.Adam(
        list(agent.policy.parameters()) + list(agent.value.parameters()),
        lr=training["policy_lr"],
    )

    # Tracking and environment
    counter = {
        "world_updates": 0,
        "agent_updates": 0,
        "world_training_seconds": 0.0,
        "agent_training_seconds": 0.0,
    }
    episodes, agent_environment_steps = 0, 0
    env = HistoryStackWrapper(
        config["environment"],
        images=agent.world.has_image,
        gui=gui,
    )
    writer = SummaryWriter(str(directory / "tensorboard"))
    try:
        if resume:
            state = load_checkpoint(resume)
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
            counter = state["counter"]
            episodes = state["episodes"]
            agent_environment_steps = state["agent_environment_steps"]
            torch.set_rng_state(state["torch_rng"])
            np.random.set_state(state["numpy_rng"])
            random.setstate(state["python_rng"])
            if state["cuda_rng"]:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
        else:
            if world_checkpoint:
                state = torch.load(
                    world_checkpoint, map_location="cpu", weights_only=False
                )
                # Version 8 has the same world-model architecture and loss.
                if (
                    state.get("experiment_version") not in {8, EXPERIMENT_VERSION}
                    or state.get("modality") != modality
                    or state.get("environment") != config["environment"]
                ):
                    raise ValueError("World checkpoint is incompatible with this run")
                agent.world.load_state_dict(state["world"])
                counter["world_updates"] = state["world_updates"]
                counter["world_training_seconds"] = state["world_training_seconds"]
            else:
                pretrain(
                    agent,
                    data,
                    world_optimizer,
                    config,
                    counter,
                    writer,
                    device,
                    directory,
                    seed,
                    modality,
                    payload["held_out"],
                )

        # Save run settings
        agent.world.requires_grad_(False)
        (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
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
            metrics = evaluate(agent, actor, payload["held_out"], config, seed, device)
            row = dict(
                episode=episodes,
                agent_environment_steps=agent_environment_steps,
                total_training_seconds=(
                    counter["world_training_seconds"]
                    + counter["agent_training_seconds"]
                ),
                **counter,
                training_return=float(np.mean(scores)) if scores else float("nan"),
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
                agent_environment_steps,
            )
            print(
                f"seed={seed} {modality}: episode={episodes} "
                f"OOD_return={metrics['return_ood']:.1f} steps={agent_environment_steps:,}",
                flush=True,
            )

        if resume is None:
            record([])

        # Train policy
        while episodes < training["episodes"]:
            start = perf_counter()
            batch, episodes, scores = rollout(
                agent, actor, critic, env, config, seed, episodes, device
            )
            update(
                agent,
                batch,
                policy_optimizer,
                config,
                counter,
                device,
                ppo_loss,
            )
            agent_environment_steps += len(batch["action"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            counter["agent_training_seconds"] += perf_counter() - start
            if (
                episodes % config["evaluation"]["interval"] == 0
                or episodes == training["episodes"]
            ):
                record(scores)
        plot_run(directory)
    finally:
        writer.close()
        env.close()


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
    parser.add_argument(
        "--world-checkpoint",
        type=Path,
        help="Start a new agent from a world-model checkpoint",
    )
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
        # Check CUDA works
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
        state = load_checkpoint(args.resume)
        pairs = [(state["seed"], state["modality"])]
    if args.world_checkpoint and (args.resume or len(pairs) != 1):
        raise ValueError(
            "--world-checkpoint requires one seed and modality and cannot be combined with --resume"
        )
    payload_seed, payload = None, None

    # run training multiple times for each WAM config (different input modalities)
    for seed, modality in pairs:
        shared = output / f"seed-{seed}"
        shared.mkdir(parents=True, exist_ok=True)
        if seed != payload_seed:
            payload = shared_data(
                config,
                seed,
                shared,
                load_training=not (args.world_checkpoint or args.resume),
            )
            payload_seed = seed
        directory = shared / modality
        if args.resume and args.resume.resolve().parent != directory.resolve():
            raise ValueError(
                "Resume checkpoint must belong to the configured output directory"
            )

        # Train!
        run_arm(
            config,
            seed,
            modality,
            payload,
            directory,
            args.resume,
            args.gui,
            args.world_checkpoint,
        )

    plot_comparison(output)
    plot_comparison(output, "total_training_seconds")


if __name__ == "__main__":
    main()
