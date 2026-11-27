"""Train paired PPO agents using completed, frozen world models."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from cartpole_observation import CartPoleObservation
from experiment_config import build_world_model, environment_settings, load_config
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from training_utils import (
    EXPERIMENT_VERSION,
    MODALITIES,
    MetricWriter,
    check_output,
    checked_step,
    prepare_run,
    reconstruction_mae,
    run_directory,
    save_snapshot,
    seed_run,
    setup_device,
    to_device,
)

from src.common.models.agent import Agent


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


def observation_batch(observation, device):
    """Add the batch dimension to a single environment observation."""
    return TensorDict(observation, batch_size=[], device=device).unsqueeze(0)


def rollout(agent, actor, critic, env, config, seed, episodes, device):
    """Collect policy data through an episode boundary without crossing evaluation intervals."""
    settings = config["ppo"]
    target = config["train"]["episodes"]
    next_evaluation = (episodes // config["train_eval"]["interval"] + 1) * config[
        "train_eval"
    ]["interval"]

    # Rollout buffers
    transitions, scores = [], []
    observation, _ = env.reset(seed=2 * (seed * 100_000 + episodes))
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
            observation, _ = env.reset(seed=2 * (seed * 100_000 + episodes))
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
    """Measure return and WM errors on held-out policy episodes.

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
    state_names = agent.world.state_names
    totals = torch.zeros((horizon + 1, len(state_names)), device=device)
    counts = torch.zeros(horizon + 1, device=device)
    scores = []
    env = CartPoleObservation(images=agent.world.has_image)
    try:
        with set_exploration_type(ExplorationType.MODE):
            for episode in range(config["train_eval"]["eval_episodes"]):
                observation, _ = env.reset(seed=2 * (seed * 1000 + episode) + 1)
                pending = []
                ended, score = False, 0.0
                while not ended:
                    batch = observation_batch(observation, device)
                    actor(batch)
                    # Compare previous predictions to this real observation.
                    pending.append((batch["latent"], 0))
                    for latent, step in pending:
                        prediction = agent.world.decode(latent)["vector"][:, -1]
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
                    totals[step] += (prediction - target).abs().sum(0)
                    counts[step] += 1
                scores.append(score)
    finally:
        env.close()
    metrics = {
        "return": float(np.mean(scores)),
        "return_std": float(np.std(scores)),
    }
    for step in range(horizon + 1):
        for name, error in zip(state_names, totals[step] / counts[step].clamp_min(1)):
            # A short episode can leave a long horizon without any valid window.
            metrics[f"mae_{name}_h{step}"] = (
                error.item() if counts[step] else float("nan")
            )
        metrics[f"windows_h{step}"] = counts[step].item()
    return metrics


def load_world_models(config, source):
    """Validate every requested final WM pair before any agent training starts.

    Args:
        config: Agent configuration; train.seeds selects matching WM seeds.
        source: Root directory containing pretrained world-model runs.

    Returns:
        Checkpoint paths and validated states keyed by seed and modality.

    Raises:
        FileNotFoundError: A required final world model is missing.
        ValueError: A world model is incomplete or incompatible.
    """
    checkpoints = {}
    settings = environment_settings(config)
    for seed in config["train"]["seeds"]:
        for modality in MODALITIES:
            path = run_directory(source, seed, modality) / "world-final.pt"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Required final world model is missing: {path}"
                )
            state = torch.load(path, map_location="cpu", weights_only=False)
            if (
                state.get("experiment_version") != EXPERIMENT_VERSION
                or state.get("modality") != modality
                or state.get("seed") != seed
                or state.get("environment") != settings
                or state.get("epoch")
                != state.get("config", {}).get("pretrain", {}).get("epochs")
                or "epoch" not in state
            ):
                raise ValueError(
                    f"World checkpoint is incomplete or incompatible: {path}"
                )
            try:
                world = build_world_model(modality, settings)
                world.load_state_dict(state["world"])
                if any(
                    not torch.isfinite(value).all()
                    for value in world.state_dict().values()
                ):
                    raise ValueError("Nonfinite world-model weights")
                for key in ("world_updates", "world_training_seconds"):
                    if not np.isfinite(state[key]) or state[key] < 0:
                        raise ValueError(f"Invalid {key}")
            except (KeyError, RuntimeError, ValueError) as error:
                raise ValueError(
                    f"Invalid world checkpoint: {path}: {error}"
                ) from error
            checkpoints[seed, modality] = (path.resolve(), state)
    return checkpoints


def load_agent_checkpoint(directory, config, seed, modality):
    """Load the latest agent checkpoint and check training settings."""
    paths = [
        path
        for path in directory.glob("episode-*.pt")
        if path.stem.removeprefix("episode-").isdigit()
    ]
    if not paths:
        raise FileNotFoundError(f"No agent checkpoint in {directory}")
    path = max(paths, key=lambda path: int(path.stem.removeprefix("episode-")))
    state = torch.load(path, map_location="cpu", weights_only=True)
    previous = state["config"]
    if (
        state["experiment_version"] != EXPERIMENT_VERSION
        or state["ppo_backend"] != "torchrl"
        or state["seed"] != seed
        or state["modality"] != modality
        or state["episodes"] != int(path.stem.removeprefix("episode-"))
        or any(
            previous[key] != config[key] for key in ("environment", "ppo", "train_eval")
        )
        or previous["train"]["policy_lr"] != config["train"]["policy_lr"]
        or state["episodes"] > config["train"]["episodes"]
    ):
        raise ValueError(f"Agent checkpoint is incompatible with this run: {path}")
    return state


def run_training(
    config, seed, modality, checkpoint, directory, gui=False, *, saved=None
):
    """Train one agent using an already validated, frozen world model.

    Args:
        config: Experiment configuration.
        seed: Agent seed, matching the source WM seed.
        modality: Vector or combined observations.
        checkpoint: Source path and state returned by load_world_models.
        directory: Output directory for this agent run.
        gui: Whether to display the training environment.
        saved: Agent checkpoint to resume, if any.
    """
    if saved is None:
        prepare_run(directory, config)
    seed_run(seed)
    device = torch.device(config["run"]["device"])
    source, state = checkpoint
    world = build_world_model(modality, config["environment"])
    agent = Agent(modality, head_seed=seed, world=world).to(device)
    agent.world.load_state_dict(state["world"])
    agent.world.requires_grad_(False)
    actor, critic, ppo_loss = ppo_modules(agent, config["ppo"])
    policy_optimizer = torch.optim.Adam(
        list(agent.policy.parameters()) + list(agent.value.parameters()),
        lr=config["train"]["policy_lr"],
    )
    counter = {
        "world_updates": state["world_updates"],
        "world_training_seconds": state["world_training_seconds"],
        "agent_updates": 0,
        "agent_training_seconds": 0.0,
    }
    episodes, agent_environment_steps = 0, 0
    if saved is not None:
        agent.load_state_dict(saved["agent"])
        source = Path(saved["world_checkpoint"])
        episodes = saved["episodes"]
        agent_environment_steps = saved["agent_environment_steps"]
        counter = saved["counter"]
        if "policy_optimizer" in saved:
            policy_optimizer.load_state_dict(saved["policy_optimizer"])
        else:
            print(
                f"{directory}: resuming legacy checkpoint with fresh Adam and RNG",
                flush=True,
            )

    env = CartPoleObservation(images=agent.world.has_image, gui=gui)
    writer = None
    try:
        resume_at = (episodes, counter["agent_updates"]) if saved is not None else None
        writer = MetricWriter(directory, resume_at=resume_at)

        def record():
            """Record held-out evaluation and save an inspectable agent snapshot."""
            metrics = evaluate(agent, actor, config, seed, device)
            for key, value in metrics.items():
                writer.add_scalar(f"eval/{key}", value, episodes)
            for key, value in {
                **counter,
                "agent_environment_steps": agent_environment_steps,
                "total_training_seconds": counter["world_training_seconds"]
                + counter["agent_training_seconds"],
            }.items():
                writer.add_scalar(f"compute/{key}", value, episodes)
            writer.flush()
            save_snapshot(
                directory / f"episode-{episodes:06d}.pt",
                {
                    "experiment_version": EXPERIMENT_VERSION,
                    "ppo_backend": "torchrl",
                    "agent": agent.state_dict(),
                    "policy_optimizer": policy_optimizer.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device)
                    if device.type == "cuda"
                    else None,
                    "config": config,
                    "seed": seed,
                    "modality": modality,
                    "episodes": episodes,
                    "agent_environment_steps": agent_environment_steps,
                    "counter": dict(counter),
                    "world_checkpoint": str(source),
                    "world_seed": state["seed"],
                },
            )
            print(
                f"seed={seed} {modality}: episode={episodes} "
                f"eval_return={metrics['return']:.1f} steps={agent_environment_steps:,}",
                flush=True,
            )

        if saved is None:
            record()
        elif "torch_rng" in saved:
            torch.set_rng_state(saved["torch_rng"])
            if device.type == "cuda" and saved["cuda_rng"] is not None:
                torch.cuda.set_rng_state(saved["cuda_rng"], device)
        # Train only the policy and value heads.
        while episodes < config["train"]["episodes"]:
            start = perf_counter()
            batch, episodes, scores = rollout(
                agent, actor, critic, env, config, seed, episodes, device
            )
            update(agent, batch, policy_optimizer, config, counter, device, ppo_loss)
            agent_environment_steps += len(batch["action"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            counter["agent_training_seconds"] += perf_counter() - start
            for episode, score in enumerate(scores, episodes - len(scores) + 1):
                writer.add_scalar("train/ppo/episode_return", score, episode)
            for name, value in reconstruction_mae(
                agent.world, batch, config, device
            ).items():
                writer.add_scalar(
                    f"train/wm/mae_{name}", value, counter["agent_updates"]
                )
            if (
                episodes % config["train_eval"]["interval"] == 0
                or episodes == config["train"]["episodes"]
            ):
                record()
    finally:
        if writer is not None:
            writer.close()
        env.close()


def main():
    """Train both agent modalities after checking every required WM pair."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.toml")
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the latest agent checkpoints",
    )
    parser.add_argument(
        "--output", type=Path, help="Override the agent output directory"
    )
    parser.add_argument(
        "--world-models", type=Path, help="Select a pretrained world-model directory"
    )
    parser.add_argument(
        "--gui", action="store_true", help="Show the training environment"
    )
    args = parser.parse_args()
    config = load_config(args.config, "train", args.smoke)
    setup_device(config)
    output = args.output or Path(config["run"]["output"]) / "agents"
    source = args.world_models or Path(config["run"]["output"]) / "world-models"
    if not args.resume:
        check_output(output, config["train"]["seeds"])
    checkpoints = load_world_models(config, source)
    directories = {key: run_directory(output, *key) for key in checkpoints}
    saved_runs = {
        key: load_agent_checkpoint(directory, config, *key)
        for key, directory in directories.items()
        if args.resume and directory.exists()
    }
    for (seed, modality), checkpoint in checkpoints.items():
        saved = saved_runs.get((seed, modality))
        if saved is not None and saved["episodes"] == config["train"]["episodes"]:
            print(
                f"seed={seed} {modality}: already completed {saved['episodes']} episodes"
            )
            continue
        run_training(
            config,
            seed,
            modality,
            checkpoint,
            directories[seed, modality],
            args.gui,
            saved=saved,
        )


if __name__ == "__main__":
    main()
