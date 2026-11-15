"""Train paired CartPole agents and evaluate on sampled OOD rollouts."""

# General
import argparse
import copy
import csv
import json
import math
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
from src.common.environments.history_stack_wrapper import HistoryStackWrapper
from src.common.models.agent import Agent

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_config import environment_settings, validate_ranges

EXPERIMENT_VERSION = 10

TENSORBOARD_STATE_NAMES = ("cartPos", "cartVel", "poleAngle")


class MetricWriter:
    """Write training scalars to TensorBoard and one CSV per run.

    Args:
        directory: Directory for this seed and modality.
    """

    def __init__(self, directory):
        self.path = directory / "metrics.csv"
        self.logdir = directory / "tensorboard"
        self.writer = SummaryWriter(str(self.logdir))
        self.rows = 0
        if self.path.exists():
            with self.path.open(newline="") as handle:
                self.rows = sum(1 for _ in csv.DictReader(handle))

    def add_csv(self, metric, value, step):
        """Append a metric using its own phase's step counter.

        Args:
            metric: Metric tag, including its phase prefix.
            value: Scalar metric value.
            step: Epoch, episode, or optimizer update count.
        """
        exists = self.path.exists()
        with self.path.open("a", newline="") as handle:
            output = csv.writer(handle)
            if not exists:
                output.writerow(("metric", "step", "value"))
            output.writerow((metric, step, float(value)))
        self.rows += 1

    def add_scalar(self, metric, value, step):
        """Record the same scalar in TensorBoard and CSV.

        Args:
            metric: TensorBoard tag.
            value: Scalar metric value.
            step: Epoch, episode, or optimizer update count.
        """
        self.writer.add_scalar(metric, value, step)
        self.add_csv(metric, value, step)

    def restore(self, rows):
        """Restore logs to the checkpoint, including their different step axes.

        Args:
            rows: Number of CSV rows saved in the checkpoint.

        Raises:
            ValueError: The CSV has fewer rows than the checkpoint requires.
        """
        records = []
        if self.path.exists():
            with self.path.open(newline="") as handle:
                records = list(csv.DictReader(handle))
        if len(records) < rows:
            raise ValueError("The metrics CSV is missing checkpoint records")
        records = records[:rows]
        with self.path.open("w", newline="") as handle:
            output = csv.DictWriter(handle, fieldnames=("metric", "step", "value"))
            output.writeheader()
            output.writerows(records)
        # Rebuild event files from durable CSV rows to avoid stale resumed steps.
        self.writer.close()
        for path in self.logdir.glob("events.out.tfevents.*"):
            path.unlink()
        self.writer = SummaryWriter(str(self.logdir))
        for row in records:
            if row["metric"].startswith(("pretrain/wm/", "train/ppo/", "train/wm/")):
                self.writer.add_scalar(
                    row["metric"], float(row["value"]), int(row["step"])
                )
        self.rows = rows

    def flush(self):
        """Flush pending TensorBoard events; CSV rows are written immediately."""
        self.writer.flush()

    def close(self):
        """Close TensorBoard event writing."""
        self.writer.close()


# ============= VALIDATION =============
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
        "pretrain": ["transitions", "epochs", "eval_every", "batch_size", "world_lr"],
        "train": ["episodes", "policy_lr"],
        "pretrain_eval": ["transitions", "batch_size"],
        "ppo": ["rollout_steps", "minibatch_size", "epochs", "max_grad_norm", "clip"],
        "train_eval": ["interval", "eval_episodes"],
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
    for key in ("vector_weight", "image_weight"):
        value = config["pretrain"][key]
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"pretrain.{key} must be finite and positive")
    for key in ("gamma", "gae_lambda"):
        if not 0 <= config["ppo"][key] <= 1:
            raise ValueError(f"ppo.{key} must be between zero and one")
    for key in ("value_weight", "entropy_weight"):
        value = config["ppo"][key]
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"ppo.{key} must be finite and nonnegative")

    # Shared ID and OOD ranges
    validate_ranges(config)
    horizon = config["train_eval"]["horizon"]
    if (
        type(horizon) is not int
        or not 0 <= horizon < config["environment"]["max_steps"]
    ):
        raise ValueError("train_eval.horizon must be an integer below max_steps")


def smoke_config(config):
    """Return a small real-training configuration for end-to-end checks."""
    config = copy.deepcopy(config)
    config["run"].update(seeds=[0])
    config["environment"]["max_steps"] = 30
    config["pretrain"].update(transitions=32, epochs=1, batch_size=8)
    config["train"]["episodes"] = 2
    config["ppo"].update(rollout_steps=16, minibatch_size=8, epochs=1)
    config["train_eval"].update(interval=1, eval_episodes=1)
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


# ============= AGENT TRAINING =============
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


@torch.no_grad()
def reconstruction_mae(world, data, config, device):
    """Measure current-state reconstruction MAE in physical units.

    Args:
        world: World model to evaluate.
        data: Observation histories from a dataset or policy rollout.
        config: Experiment configuration.
        device: Model device.

    Returns:
        One mean absolute error per state variable.
    """
    totals = torch.zeros(3, device=device)
    count = len(data["vector"])
    for indices in torch.arange(count).split(config["pretrain_eval"]["batch_size"]):
        batch = to_device(data, indices, device, world)
        latent = world.encode(batch["vector"], batch.get("image"))
        prediction = world.decode(latent)["vector"][:, -1] * world.std + world.mean
        totals += (prediction - batch["vector"][:, -1]).abs().sum(0)
    return dict(zip(TENSORBOARD_STATE_NAMES, (totals / count).tolist()))


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
    test_data,
):
    """Train the full world model on the fixed synthesized dataset."""
    settings = config["pretrain"]
    print(
        f"Pretraining world model: seed={seed} modality={modality}, "
        f"{settings['epochs']} epochs",
        flush=True,
    )

    # Train each epoch
    for epoch in range(settings["epochs"]):
        start = perf_counter()
        order = torch.randperm(len(data["action"]))
        total = 0.0
        components = {"reconstruction_loss": 0.0, "dynamics_loss": 0.0}
        for indices in order.split(settings["batch_size"]):
            batch = to_device(data, indices, device, agent.world)
            loss, losses = agent.world.loss(
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
            for key in components:
                components[key] += losses[key].item() * len(indices)
        if device.type == "cuda":
            torch.cuda.synchronize()
        counter["world_training_seconds"] += perf_counter() - start
        metrics = {
            "loss_total": total / len(order),
            "loss_rec": components["reconstruction_loss"] / len(order),
            "loss_dyn": components["dynamics_loss"] / len(order),
            "wall_time": counter["world_training_seconds"],
            "grad_steps": counter["world_updates"],
        }
        evaluating = (epoch + 1) % settings["eval_every"] == 0 or epoch + 1 == settings[
            "epochs"
        ]
        if evaluating:
            for label, observations in (("train", data), ("test", test_data)):
                metrics.update(
                    {
                        f"mae_{name}_{label}": value
                        for name, value in reconstruction_mae(
                            agent.world, observations, config, device
                        ).items()
                    }
                )
        for key, value in metrics.items():
            writer.add_scalar(f"pretrain/wm/{key}", value, epoch + 1)
        print(
            f"World model seed={seed} {modality}: "
            f"epoch {epoch + 1}/{settings['epochs']}, "
            f"loss={total / len(order):.6f}, updates={counter['world_updates']:,}",
            flush=True,
        )

        # Save evaluated world models.
        if evaluating:
            torch.save(
                {
                    "experiment_version": EXPERIMENT_VERSION,
                    "world": agent.world.state_dict(),
                    "modality": modality,
                    "seed": seed,
                    "epoch": epoch + 1,
                    "world_updates": counter["world_updates"],
                    "world_training_seconds": counter["world_training_seconds"],
                    "environment": environment_settings(config),
                },
                directory / f"world-epoch-{epoch + 1:03d}.pt",
            )


def rollout(agent, actor, critic, env, config, seed, episodes, device):
    """Collect policy data, finishing at an episode boundary for exact resume."""
    settings = config["ppo"]
    target = config["train"]["episodes"]
    next_evaluation = (episodes // config["train_eval"]["interval"] + 1) * config[
        "train_eval"
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
def evaluate(agent, actor, config, seed, device):
    """Measure return and WM errors on the same sampled OOD policy episodes.

    Args:
        agent: Agent with a frozen world model.
        actor: Policy used to select evaluation actions.
        config: Experiment configuration.
        seed: Paired run seed.
        device: Model device.

    Returns:
        Episode return statistics and per-variable MAE at each horizon.
        Each horizon uses every available window within episode boundaries.
    """
    horizon = config["train_eval"]["horizon"]
    totals = torch.zeros((horizon + 1, 3), device=device)
    counts = torch.zeros(horizon + 1, device=device)
    scores = []
    env = HistoryStackWrapper(
        environment_settings(config, ood=True), images=agent.world.has_image
    )
    try:
        with set_exploration_type(ExplorationType.MODE):
            for episode in range(config["train_eval"]["eval_episodes"]):
                observation, _ = env.reset(seed=1_000_000 + seed * 1000 + episode)
                pending = []
                ended, score = False, 0.0
                while not ended:
                    batch = observation_batch(observation, device)
                    actor(batch)
                    # Compare previous predictions to this real observation.
                    pending.append((batch["latent"], 0))
                    for latent, step in pending:
                        prediction = agent.world.decode(latent)["vector"][:, -1]
                        prediction = prediction * agent.world.std + agent.world.mean
                        totals[step] += (
                            (prediction - batch["vector"][:, -1]).abs().sum(0)
                        )
                        counts[step] += 1
                    action = batch["action"]
                    pending = [
                        (agent.world.predict(latent, action), step + 1)
                        for latent, step in pending
                        if step < horizon
                    ]
                    observation, reward, terminated, truncated, _ = env.step(
                        action.item()
                    )
                    score += reward
                    ended = terminated or truncated
                # Include the final observation, without crossing the reset.
                target = torch.as_tensor(observation["vector"][-1], device=device)
                for latent, step in pending:
                    prediction = agent.world.decode(latent)["vector"][:, -1]
                    totals[step] += (
                        (prediction * agent.world.std + agent.world.mean - target)
                        .abs()
                        .sum(0)
                    )
                    counts[step] += 1
                scores.append(score)
    finally:
        env.close()
    metrics = {
        "return_ood": float(np.mean(scores)),
        "return_std_ood": float(np.std(scores)),
    }
    for step in range(horizon + 1):
        for name, error in zip(
            TENSORBOARD_STATE_NAMES, totals[step] / counts[step].clamp_min(1)
        ):
            # A short episode can leave a long horizon without any valid window.
            metrics[f"mae_{name}_h{step}_ood"] = (
                error.item() if counts[step] else float("nan")
            )
        metrics[f"windows_h{step}"] = counts[step].item()
    return metrics


def shared_data(config, seed, directory, *, load_training=True):
    """Load the fixed ID/OOD pairs only when pretraining is requested.

    Args:
        config: Experiment configuration.
        seed: Paired run seed, unused because the dataset is shared.
        directory: Run directory, unused because no rollout cache is needed.
        load_training: Whether this run will pretrain a world model.

    Returns:
        ID training pairs and OOD pretraining evaluation pairs, or an empty dict.

    Raises:
        ValueError: Dataset physics or pair counts do not match the config.
    """
    if not load_training:
        return {}
    dataset = torch.load(
        config["pretrain"]["dataset"], weights_only=False, map_location="cpu"
    )
    if dataset.get("environment") != environment_settings(config) or dataset.get(
        "evaluation_environment"
    ) != environment_settings(config, ood=True):
        raise ValueError("Dataset environment changed; regenerate the ID/OOD dataset")
    for key, section in (("training", "pretrain"), ("evaluation", "pretrain_eval")):
        if (
            key not in dataset
            or len(dataset[key]["action"]) != config[section]["transitions"]
        ):
            raise ValueError(f"Dataset size does not match {section}.transitions")
    return {
        "training": dataset["training"],
        "pretrain_evaluation": dataset["evaluation"],
    }


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
    metric_rows,
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
        "metric_rows": metric_rows,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }

    # Replace atomically
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def run_training(
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
    training = config["train"]
    world_optimizer = torch.optim.Adam(
        agent.world.parameters(), lr=config["pretrain"]["world_lr"]
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
        environment_settings(config),
        images=agent.world.has_image,
        gui=gui,
    )
    writer = MetricWriter(directory)
    try:
        if resume:
            # Load & verify checkpoint
            state = load_checkpoint(resume)
            original, current = copy.deepcopy(state["config"]), copy.deepcopy(config)
            original["train"].pop("episodes")
            current["train"].pop("episodes")
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
            writer.restore(state["metric_rows"])
            episodes = state["episodes"]
            agent_environment_steps = state["agent_environment_steps"]
            torch.set_rng_state(state["torch_rng"])
            np.random.set_state(state["numpy_rng"])
            random.setstate(state["python_rng"])
            if state["cuda_rng"]:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
        else:
            if world_checkpoint:
                # Use saved pretrained WM
                state = torch.load(
                    world_checkpoint, map_location="cpu", weights_only=False
                )
                if (
                    state.get("experiment_version") != EXPERIMENT_VERSION
                    or state.get("modality") != modality
                    or state.get("environment") != environment_settings(config)
                ):
                    raise ValueError("World checkpoint is incompatible with this run")
                agent.world.load_state_dict(state["world"])
                counter["world_updates"] = state["world_updates"]
                counter["world_training_seconds"] = state["world_training_seconds"]
            else:
                # pretrain WM
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
                    payload["pretrain_evaluation"],
                )

        # Save run settings
        agent.world.requires_grad_(False)
        (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")

        def record():
            """Record joint rollout evaluation and save a durable checkpoint."""
            metrics = evaluate(agent, actor, config, seed, device)
            for key, value in metrics.items():
                writer.add_csv(f"eval/{key}", value, episodes)
            for key, value in {
                **counter,
                "agent_environment_steps": agent_environment_steps,
                "total_training_seconds": counter["world_training_seconds"]
                + counter["agent_training_seconds"],
            }.items():
                writer.add_csv(f"compute/{key}", value, episodes)
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
                writer.rows,
            )
            print(
                f"seed={seed} {modality}: episode={episodes} "
                f"OOD_return={metrics['return_ood']:.1f} steps={agent_environment_steps:,}",
                flush=True,
            )

        if resume is None:
            record()

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
            for episode, score in enumerate(scores, episodes - len(scores) + 1):
                writer.add_scalar("train/ppo/episode_return", score, episode)
            # One observation evaluation per rollout update, indexed by gradient steps.
            for name, value in reconstruction_mae(
                agent.world, batch, config, device
            ).items():
                writer.add_scalar(
                    f"train/wm/mae_{name}", value, counter["agent_updates"]
                )
            if (
                episodes % config["train_eval"]["interval"] == 0
                or episodes == training["episodes"]
            ):
                record()
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
        run_training(
            config,
            seed,
            modality,
            payload,
            directory,
            args.resume,
            args.gui,
            args.world_checkpoint,
        )


if __name__ == "__main__":
    main()
