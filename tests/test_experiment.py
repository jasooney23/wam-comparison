"""Behavior checks for physics, paired training stages, and measurement."""

import csv
import importlib
import importlib.util
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium.envs.classic_control.cartpole import CartPoleEnv as UpstreamCartPole
from tensordict import TensorDict
from torch import nn
from torchrl.objectives.value import GAE

from src.common.environments.cartpole import CartPoleEnv
from src.common.environments.history_stack_wrapper import (
    HistoryStackWrapper,
    random_rollouts,
)
from src.common.models.agent import Agent
from src.common.models.common.basic import MLP
from src.common.models.world_model import WorldModel

EXPERIMENT = (
    Path(__file__).resolve().parents[1] / "src/experiments/visual-augmentation-simple"
)
sys.path.insert(0, str(EXPERIMENT))
training = importlib.import_module("train_agent")
pretraining = importlib.import_module("pretrain_world_model")
settings = importlib.import_module("experiment_config")

torch.set_num_threads(1)


@pytest.fixture
def config(tmp_path, request):
    with (EXPERIMENT / "config.toml").open("rb") as handle:
        result = settings.smoke_config(tomllib.load(handle))
    result["run"].update(device="cpu", output=str(tmp_path))
    result["environment"]["max_steps"] = 5
    result["environment"].update(getattr(request, "param", {}))
    result["pretrain_eval"]["transitions"] = 8
    return result


@pytest.mark.parametrize(
    "config, frames, vector_size",
    [
        ({}, 4, 3),
        ({"history_frames": 1, "max_steps": 1, "input_pole_vel": True}, 1, 4),
    ],
    indirect=["config"],
)
def test_synthesized_dataset_is_reproducible(config, frames, vector_size):
    spec = importlib.util.spec_from_file_location(
        "dataset_generator", EXPERIMENT / "generate_dataset.py"
    )
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    first = generator.generate(training.environment_settings(config), 4, 17)
    second = generator.generate(training.environment_settings(config), 4, 17)
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    assert first["vector"].shape == (4, frames, vector_size)
    assert first["image"].shape == (4, frames, 32, 32)
    assert first["next_image"].shape == (4, 32, 32)
    assert first["next_vector"].shape == (4, vector_size)
    assert set(first["action"].tolist()) <= {0, 1}


def test_zero_damping_matches_upstream():
    local, upstream = CartPoleEnv(), UpstreamCartPole()
    np.testing.assert_array_equal(local.reset(seed=42)[0], upstream.reset(seed=42)[0])
    for action in [0, 1, 1, 0, 1, 0]:
        a, b = local.step(action), upstream.step(action)
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:] == b[1:]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cart_mass": 0},
        {"pole_mass": -1},
        {"hinge_damping": -1},
        {"cart_mass": float("nan")},
    ],
)
def test_invalid_physics_fails(kwargs):
    with pytest.raises(ValueError):
        CartPoleEnv(**kwargs)


def test_damping_satisfies_coupled_equations():
    env = CartPoleEnv(cart_mass=0.7, pole_mass=0.2, hinge_damping=0.03)
    env.reset(seed=0)
    x, velocity, angle, angular_velocity = (0.1, 0.3, 0.05, 0.4)
    env.state = np.array([x, velocity, angle, angular_velocity])
    env.step(1)
    x_acc = (env.state[1] - velocity) / env.tau
    angle_acc = (env.state[3] - angular_velocity) / env.tau
    m, length = env.masspole, env.length
    assert env.total_mass == pytest.approx(0.9)
    assert env.polemass_length == pytest.approx(0.1)
    assert env.total_mass * x_acc + m * length * np.cos(
        angle
    ) * angle_acc == pytest.approx(
        env.force_mag + m * length * np.sin(angle) * angular_velocity**2
    )
    assert 4 / 3 * m * length**2 * angle_acc + m * length * np.cos(
        angle
    ) * x_acc == pytest.approx(
        m * env.gravity * length * np.sin(angle) - env.hinge_damping * angular_velocity
    )


def test_history_resets_parameters_and_truncates(config):
    env = HistoryStackWrapper(training.environment_settings(config), images=False)
    try:
        observation, info = env.reset(seed=5)
        assert env.observation_space.contains(observation)
        np.testing.assert_array_equal(
            observation["vector"], np.repeat(observation["vector"][:1], 4, axis=0)
        )
        mass = env.unwrapped.masscart
        assert (
            training.environment_settings(config)["cart_mass"][0]
            <= mass
            <= training.environment_settings(config)["cart_mass"][1]
        )
        for _ in range(5):
            _, _, _terminated, truncated, _ = env.step(0)
            assert env.unwrapped.masscart == mass
        assert truncated
        repeated, repeated_info = env.reset(seed=5)
        np.testing.assert_array_equal(repeated["vector"], observation["vector"])
        assert info == repeated_info
        assert env.unwrapped.total_mass == pytest.approx(mass + 0.1)
    finally:
        env.close()


def test_render_has_no_parameter_cues_and_preserves_angle(config):
    a = HistoryStackWrapper(
        training.environment_settings(config), fixed_params=(0.25, 0.0)
    )
    b = HistoryStackWrapper(
        training.environment_settings(config), fixed_params=(2.0, 0.01)
    )
    try:
        first, _ = a.reset(seed=0)
        second, _ = b.reset(seed=0)
        assert a.observation_space.contains(first)
        np.testing.assert_array_equal(first["image"], second["image"])
        a.unwrapped.state[2] += 0.02
        changed = a._observe(np.asarray(a.unwrapped.state))
        assert np.any(changed["image"][-1] != first["image"][-1])
    finally:
        a.close()
        b.close()


def test_registered_wrapper_passes_gym_checks(config):
    from gymnasium.utils.env_checker import check_env

    env = CartPoleEnv(cart_mass=1.2, hinge_damping=0.002)
    check_env(env, skip_render_check=True)


@pytest.mark.parametrize("modality", ["vector", "combined"])
def test_model_shapes_and_gradients(modality):
    model = WorldModel(modality)
    batch = {
        "vector": torch.randn(3, 4, 3),
        "image": torch.randint(256, (3, 4, 32, 32), dtype=torch.uint8),
        "action": torch.tensor([0, 1, 0]),
        "next_vector": torch.randn(3, 3),
        "next_image": torch.randint(256, (3, 32, 32), dtype=torch.uint8),
    }
    assert model.encode(batch["vector"], batch["image"]).shape == (3, 16)
    predictions = model(batch["vector"], batch["image"], batch["action"])
    assert predictions["vector"].shape == (3, 4, 3)
    if model.has_image:
        assert predictions["image"].shape == (3, 4, 32, 32)
    batch = {
        key: value
        for key, value in batch.items()
        if model.has_image or "image" not in key
    }
    loss, metrics = model.loss(batch)
    torch.testing.assert_close(
        loss, metrics["reconstruction_loss"] + metrics["dynamics_loss"]
    )
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_mlp_explicit_hidden_layers_and_default_linear_model():
    model = MLP(3, 2, [32])
    layers = [layer for layer in model if isinstance(layer, nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in layers] == [
        (3, 32),
        (32, 2),
    ]
    for hidden_layers in (None, []):
        linear = MLP(3, 2, hidden_layers)
        assert sum(isinstance(layer, nn.Linear) for layer in linear) == 1
        assert linear(torch.zeros(4, 3)).shape == (4, 2)


def test_gae_bootstraps_truncations_without_reset_leakage():
    data = TensorDict(
        {
            "state_value": [[2.0], [10.0]],
            "next": {
                "reward": [[1.0], [1.0]],
                "state_value": [[3.0], [99.0]],
                "done": [[True], [True]],
                "terminated": [[False], [True]],
            },
        },
        batch_size=[2],
    )
    GAE(gamma=0.9, lmbda=0.95, value_network=None)(data)
    torch.testing.assert_close(data["advantage"], torch.tensor([[1.7], [-9.0]]))
    torch.testing.assert_close(data["value_target"], torch.tensor([[3.7], [1.0]]))


@pytest.mark.parametrize("modality", ["vector", "combined"])
def test_agent_updates_only_policy_parameters(config, modality):
    torch.manual_seed(0)
    data = random_rollouts(training.environment_settings(config), 8, 0)
    agent = Agent(modality)
    actor, critic, ppo_loss = training.ppo_modules(agent, config["ppo"])
    with torch.no_grad():
        data = critic(actor(TensorDict(data, batch_size=[8])))
    data["advantage"] = torch.ones(8, 1)
    data["value_target"] = torch.ones(8, 1)
    agent.world.requires_grad_(False)
    counter = {"world_updates": 0, "agent_updates": 0}
    policy_parameters = list(agent.policy.parameters()) + list(agent.value.parameters())
    policy_optimizer = torch.optim.Adam(policy_parameters, lr=0.001)
    before_world = [p.clone() for p in agent.world.parameters()]
    before_policy = [p.clone() for p in agent.policy.parameters()]
    training.update(agent, data, policy_optimizer, config, counter, "cpu", ppo_loss)
    world_changed = any(
        not torch.equal(old, new)
        for old, new in zip(before_world, agent.world.parameters())
    )
    assert not world_changed
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before_policy, agent.policy.parameters())
    )
    assert counter["agent_updates"] > 0
    assert counter["world_updates"] == 0


@pytest.mark.parametrize("modality", ["vector", "combined"])
def test_ppo_and_world_losses_have_separate_gradients(config, modality):
    agent = Agent(modality)
    actor, critic, ppo_loss = training.ppo_modules(agent, config["ppo"])
    data = random_rollouts(training.environment_settings(config), 4, 0)
    data = training.to_device(data, torch.arange(4), "cpu", agent.world)
    with torch.no_grad():
        critic(actor(data))
    data["advantage"] = torch.ones(4, 1)
    data["value_target"] = torch.ones(4, 1)

    # No prediction loss contributes to either gradient.
    losses = ppo_loss(data)
    losses["loss_objective"].backward(retain_graph=True)
    assert all(p.grad is None for p in agent.world.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in agent.policy.parameters()
    )
    agent.zero_grad(set_to_none=True)
    losses["loss_critic"].backward()
    assert all(p.grad is None for p in agent.world.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in agent.value.parameters()
    )

    agent.zero_grad(set_to_none=True)
    agent.world.loss(data)[0].backward()
    assert all(p.grad is None for p in agent.policy.parameters())
    assert all(p.grad is None for p in agent.value.parameters())
    assert agent.world.projection.weight.grad.abs().sum() > 0


def test_single_transition_rollout_bootstraps_time_limit(config):
    config["environment"]["max_steps"] = 1
    config["train"]["episodes"] = 1
    agent = Agent("vector")
    actor, critic, _ = training.ppo_modules(agent, config["ppo"])
    env = HistoryStackWrapper(training.environment_settings(config), images=False)
    try:
        data, episodes, scores = training.rollout(
            agent, actor, critic, env, config, 0, 0, "cpu"
        )
    finally:
        env.close()
    assert episodes == 1 and scores == [1.0]
    assert data["next", "done"].item()
    assert not data["next", "terminated"].item()
    expected = (
        data["next", "reward"] + config["ppo"]["gamma"] * data["next", "state_value"]
    )
    torch.testing.assert_close(data["value_target"], expected)
    assert torch.isfinite(data["advantage"]).all()


def write_dataset(config, tmp_path):
    data = random_rollouts(training.environment_settings(config), 32, 0)
    path = tmp_path / "dataset.pt"
    torch.save(
        {
            "environment": training.environment_settings(config),
            "training": data,
            "evaluation": random_rollouts(
                training.environment_settings(config, ood=True),
                config["pretrain_eval"]["transitions"],
                1,
            ),
            "evaluation_environment": training.environment_settings(config, ood=True),
        },
        path,
    )
    config["pretrain"]["dataset"] = str(path)


def test_dataset_rejects_changed_physics(config, tmp_path):
    write_dataset(config, tmp_path)
    pretraining.load_dataset(config)
    config["train"]["cart_mass"] = [0.6, 1.4]
    with pytest.raises(ValueError, match="Dataset environment"):
        pretraining.load_dataset(config)


def test_dataset_uses_legacy_observation_defaults(config, tmp_path):
    write_dataset(config, tmp_path)
    path = config["pretrain"]["dataset"]
    dataset = torch.load(path, weights_only=False)
    for section in ("environment", "evaluation_environment"):
        for key in settings.OBSERVATION_DEFAULTS:
            del dataset[section][key]
    torch.save(dataset, path)
    pretraining.load_dataset(config)

    # Swap inputs while preserving the vector shape.
    config["environment"].update(input_cart_vel=False, input_pole_vel=True)
    with pytest.raises(ValueError, match="Dataset environment"):
        pretraining.load_dataset(config)


@pytest.fixture
def world_pairs(config, tmp_path):
    write_dataset(config, tmp_path)
    payload = pretraining.load_dataset(config)
    source = tmp_path / "world-models"
    for modality in training.MODALITIES:
        pretraining.run_pretraining(
            config, 0, modality, payload, training.run_directory(source, 0, modality)
        )
    return source


@pytest.mark.parametrize(
    "config",
    [{}, {"history_frames": 2, "input_cart_vel": False, "input_pole_vel": True}],
    indirect=True,
)
def test_paired_agent_training_needs_no_dataset(config, tmp_path, world_pairs):
    Path(config["pretrain"]["dataset"]).unlink()
    # Neither the source seed list nor WM optimizer settings govern agent training.
    config["pretrain"].update(seeds=[99], epochs=0, world_lr=0)
    settings.validate(config, "train")
    checkpoints = training.load_world_models(config, world_pairs)
    initial_heads = {}
    for (seed, modality), checkpoint in checkpoints.items():
        directory = training.run_directory(tmp_path / "agents", seed, modality)
        training.run_training(config, seed, modality, checkpoint, directory)
        source, world = checkpoint
        initial = torch.load(directory / "episode-000000.pt", weights_only=False)
        initial_heads[modality] = {
            key: value
            for key, value in initial["agent"].items()
            if key.startswith(("policy.", "value."))
        }
        result = torch.load(directory / "episode-000002.pt", weights_only=False)
        for key, value in world["world"].items():
            torch.testing.assert_close(
                result["agent"][f"world.{key}"], value, rtol=0, atol=0
            )
        assert any(
            not torch.equal(value, result["agent"][key])
            for key, value in initial["agent"].items()
            if key.startswith("policy.")
        )
        assert result["counter"]["world_updates"] == world["world_updates"]
        assert (
            result["counter"]["world_training_seconds"]
            == world["world_training_seconds"]
        )
        assert result["counter"]["agent_updates"] > 0
        assert result["counter"]["agent_training_seconds"] > 0
        assert result["world_checkpoint"] == str(source)
        assert result["world_seed"] == seed
        metric_rows = (directory / "metrics.csv").read_text()
        assert ("mae_poleVel_h0_ood" in metric_rows) == config["environment"][
            "input_pole_vel"
        ]
        assert ("mae_cartVel_h0_ood" in metric_rows) == config["environment"][
            "input_cart_vel"
        ]
        assert (
            not {"policy_optimizer", "world_optimizer", "torch_rng", "metric_rows"}
            & result.keys()
        )
        assert [p.name for p in sorted(directory.glob("episode-*.pt"))] == [
            "episode-000000.pt",
            "episode-000001.pt",
            "episode-000002.pt",
        ]
        assert_tensorboard_matches_csv(directory)
        assert_tensorboard_matches_csv(
            training.run_directory(world_pairs, seed, modality)
        )
        with pytest.raises(FileExistsError):
            training.run_training(config, seed, modality, checkpoint, directory)

    torch.testing.assert_close(
        initial_heads["vector"], initial_heads["combined"], rtol=0, atol=0
    )


@pytest.mark.parametrize("missing", ["combined", "later_seed"])
def test_missing_pair_prevents_all_agent_training(
    config, tmp_path, world_pairs, monkeypatch, missing
):
    if missing == "combined":
        (world_pairs / "seed-0/combined/world-final.pt").unlink()
    else:
        config["train"]["seeds"] = [0, 1]

    def reject_training(*args, **kwargs):
        raise AssertionError("No agent may start before every pair is checked")

    monkeypatch.setattr(training, "run_training", reject_training)
    monkeypatch.setattr(training, "load_config", lambda *args: config)
    output = tmp_path / "agents"
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_agent.py", "--world-models", str(world_pairs), "--output", str(output)],
    )
    with pytest.raises(FileNotFoundError, match="Required final world model"):
        training.main()
    assert not output.exists()


@pytest.mark.parametrize(
    "damage",
    ["seed", "modality", "version", "environment", "epoch", "weights", "nonfinite"],
)
def test_incompatible_pair_prevents_training(config, world_pairs, damage, monkeypatch):
    path = world_pairs / "seed-0/combined/world-final.pt"
    state = torch.load(path, weights_only=False)
    if damage == "seed":
        state["seed"] = 1
    elif damage == "modality":
        state["modality"] = "vector"
    elif damage == "version":
        state["experiment_version"] = -1
    elif damage == "environment":
        state["environment"]["pole_mass"] = 99
    elif damage == "epoch":
        state["epoch"] = 0
    elif damage == "weights":
        state["world"].pop("mean")
    else:
        state["world"]["mean"].fill_(float("nan"))
    torch.save(state, path)
    monkeypatch.setattr(training, "load_config", lambda *args: config)
    monkeypatch.setattr(
        sys, "argv", ["train_agent.py", "--world-models", str(world_pairs)]
    )
    with pytest.raises(ValueError, match="checkpoint"):
        training.main()
    assert not (Path(config["run"]["output"]) / "agents").exists()


def test_pretraining_failure_does_not_publish_final(config, tmp_path, monkeypatch):
    write_dataset(config, tmp_path)
    payload = pretraining.load_dataset(config)
    directory = tmp_path / "failed"

    def fail_evaluation(*args):
        raise RuntimeError("Evaluation failed")

    monkeypatch.setattr(pretraining, "next_step_mae", fail_evaluation)
    with pytest.raises(RuntimeError, match="Evaluation failed"):
        pretraining.run_pretraining(config, 0, "vector", payload, directory)
    assert not (directory / "world-final.pt").exists()
    with pytest.raises(FileExistsError):
        pretraining.run_pretraining(config, 0, "vector", payload, directory)


def test_output_preflight_checks_later_pairs(tmp_path):
    (tmp_path / "seed-1/combined").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        training.check_output(tmp_path, [0, 1])
    assert not (tmp_path / "seed-0").exists()


@pytest.mark.parametrize("stage", ["pretrain", "train"])
@pytest.mark.parametrize("seeds", [[], [0, 0], [-1], [2**32], [True], [0.5]])
def test_invalid_stage_seeds_fail(config, stage, seeds):
    config[stage]["seeds"] = seeds
    with pytest.raises(ValueError, match="seeds"):
        settings.validate(config, stage)


def test_pretraining_does_not_validate_agent_settings(config):
    config["train"]["seeds"] = []
    config.pop("ppo")
    config.pop("train_eval")
    settings.validate(config, "pretrain")


def assert_tensorboard_matches_csv(directory):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = EventAccumulator(
        str(directory / "tensorboard"), size_guidance={"scalars": 0}
    ).Reload()
    with (directory / "metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    tags = {row["metric"] for row in rows}
    assert set(events.Tags()["scalars"]) == tags
    for tag in tags:
        records = [row for row in rows if row["metric"] == tag]
        scalars = events.Scalars(tag)
        assert [event.step for event in scalars] == [
            int(row["step"]) for row in records
        ]
        assert [event.value for event in scalars] == pytest.approx(
            [float(row["value"]) for row in records], nan_ok=True
        )


def test_pretraining_mae_uses_action_and_next_frame(config):
    world = WorldModel("vector")
    data = {"vector": torch.zeros(3, 4, 3), "action": torch.tensor([0, 1, 1])}
    data["next_vector"] = data["action"][:, None].repeat(1, 3).float()
    world.encode = lambda vector, image: torch.zeros(len(vector), 16)
    world.predict = lambda latent, action: action[:, None].expand(-1, 16).float()
    world.decode = lambda latent: {
        "vector": latent[:, :3, None].transpose(1, 2).expand(-1, 4, -1)
    }
    config["pretrain_eval"]["batch_size"] = 2
    assert list(pretraining.next_step_mae(world, data, config, "cpu").values()) == [
        0.0,
        0.0,
        0.0,
    ]
    data["next_vector"] += torch.tensor([1.0, 2.0, 3.0])
    assert list(pretraining.next_step_mae(world, data, config, "cpu").values()) == [
        1.0,
        2.0,
        3.0,
    ]


def test_pretraining_evaluates_periodic_and_final_checkpoints(config, tmp_path):
    config["pretrain"].update(epochs=11, batch_size=4)
    data = random_rollouts(training.environment_settings(config), 4, 0)
    test = random_rollouts(training.environment_settings(config, ood=True), 2, 20)
    directory = tmp_path / "world"
    pretraining.run_pretraining(
        config, 0, "vector", {"training": data, "pretrain_evaluation": test}, directory
    )
    with (directory / "metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert_tensorboard_matches_csv(directory)
    mae = {
        (row["metric"], int(row["step"])): float(row["value"])
        for row in rows
        if row["metric"].endswith("_test")
    }
    assert len([row for row in rows if row["metric"] == "pretrain/wm/loss_total"]) == 11
    world = WorldModel("vector")
    assert len(mae) == 3 * 2 * len(world.state_names)
    for epoch in (5, 10, 11):
        state = torch.load(
            directory / f"world-epoch-{epoch:03d}.pt", weights_only=False
        )
        world.load_state_dict(state["world"])
        for horizon, measure in enumerate(
            (pretraining.reconstruction_mae, pretraining.next_step_mae)
        ):
            for name, value in measure(world, test, config, "cpu").items():
                assert mae[f"pretrain/wm/mae_{name}_{horizon}_test", epoch] == (
                    pytest.approx(value)
                )
        assert state["world_updates"] == epoch
    assert [p.name for p in sorted(directory.glob("world-epoch-*.pt"))] == [
        "world-epoch-005.pt",
        "world-epoch-010.pt",
        "world-epoch-011.pt",
    ]
    final = torch.load(directory / "world-final.pt", weights_only=False)
    assert final["epoch"] == 11
    for key, value in state["world"].items():
        torch.testing.assert_close(final["world"][key], value, rtol=0, atol=0)


def test_losses_average_modalities_and_use_shifted_history():
    model = WorldModel("combined")
    for parameter in model.parameters():
        nn.init.zeros_(parameter)
    model.mean.fill_(1)
    model.std.fill_(2)
    batch = {
        "vector": torch.tensor([1.0, 3.0, 5.0, 7.0]).view(1, 4, 1).expand(2, 4, 3),
        "next_vector": torch.full((2, 3), 9.0),
        "image": torch.tensor([0, 0, 0, 255], dtype=torch.uint8)
        .view(1, 4, 1, 1)
        .expand(2, 4, 32, 32),
        "next_image": torch.full((2, 32, 32), 128, dtype=torch.uint8),
        "action": torch.tensor([0, 1]),
    }
    loss, metrics = model.loss(batch, vector_weight=2, image_weight=3)
    # Vector targets are [0, 1, 2, 3] and [1, 2, 3, 4]; images decode to 0.5.
    image_dynamics = (3 * 0.25 + (0.5 - 128 / 255) ** 2) / 4
    assert metrics["vector_reconstruction_loss"].item() == pytest.approx(3.5)
    assert metrics["vector_dynamics_loss"].item() == pytest.approx(7.5)
    assert metrics["image_reconstruction_loss"].item() == pytest.approx(0.25)
    assert metrics["image_dynamics_loss"].item() == pytest.approx(image_dynamics)
    assert metrics["reconstruction_loss"].item() == pytest.approx(2 * 3.5 + 3 * 0.25)
    assert loss.item() == pytest.approx(2 * (3.5 + 7.5) + 3 * (0.25 + image_dynamics))


def test_reconstruction_trains_encoder_and_decoder_without_dynamics(config):
    model = WorldModel("combined")
    data = random_rollouts(training.environment_settings(config), 4, 0)
    _, metrics = model.loss(data)
    metrics["reconstruction_loss"].backward()
    assert all(p.grad is None for p in model.dynamics.parameters())
    for module in (
        model.vector_encoder,
        model.image_encoder,
        model.vector_decoder,
        model.image_decoder,
    ):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
        )


@pytest.mark.parametrize("key", ["cart_mass", "hinge_damping"])
def test_evaluation_rejects_training_parameters(config, key):
    config["environment"]["ood"][key] = config["train"][key][:]
    with pytest.raises(ValueError, match="outside the training"):
        settings.validate(config, "train")


@pytest.mark.parametrize("bounds", [[0.25, 1.5], [0.25, 0.75], [0.75, 1.5], [1.5, 2]])
def test_ood_sampling_excludes_training(config, bounds):
    config["train"]["cart_mass"] = [0.5, 1.0]
    config["environment"]["ood"]["cart_mass"] = bounds
    settings.validate_ranges(config)
    env = HistoryStackWrapper(
        settings.environment_settings(config, ood=True), images=False
    )
    try:
        samples = [env.reset(seed=seed)[1]["cart_mass"] for seed in range(200)]
    finally:
        env.close()
    assert all(bounds[0] <= value <= bounds[1] for value in samples)
    assert all(value < 0.5 or value > 1.0 for value in samples)
    if bounds == [0.25, 1.5]:
        # The right interval is twice as wide as the left interval.
        assert 0.55 < np.mean(np.array(samples) > 1.0) < 0.8


@pytest.mark.parametrize("modality", ["vector", "combined"])
@pytest.mark.parametrize("horizon", [0, 4])
def test_joint_evaluation_uses_policy_rollouts(config, modality, horizon, monkeypatch):
    config["train_eval"]["horizon"] = horizon
    agent = Agent(modality)
    actor, _, _ = training.ppo_modules(agent, config["ppo"])
    before = {key: value.clone() for key, value in agent.state_dict().items()}
    traces = []

    class TracedEnvironment(HistoryStackWrapper):
        def reset(self, **kwargs):
            observation, info = super().reset(**kwargs)
            traces.append({"observations": [observation], "actions": []})
            assert (
                config["environment"]["ood"]["hinge_damping"][0]
                <= info["hinge_damping"]
                <= config["environment"]["ood"]["hinge_damping"][1]
            )
            return observation, info

        def step(self, action):
            result = super().step(action)
            traces[-1]["observations"].append(result[0])
            traces[-1]["actions"].append(action)
            return result

    monkeypatch.setattr(training, "HistoryStackWrapper", TracedEnvironment)
    if horizon == 0:

        def reject_dynamics(*args, **kwargs):
            raise AssertionError("Horizon zero must not advance dynamics")

        monkeypatch.setattr(agent.world, "predict", reject_dynamics)
    metrics = training.evaluate(agent, actor, config, 0, "cpu")
    trace = traces[0]
    observations, actions = trace["observations"], trace["actions"]
    assert metrics["return_ood"] == len(actions)
    with torch.no_grad():
        for step in range(horizon + 1):
            errors = []
            for start in range(min(len(actions), len(observations) - step)):
                batch = training.observation_batch(observations[start], "cpu")
                latent = agent.world.encode(batch["vector"], batch.get("image"))
                for action in actions[start : start + step]:
                    latent = agent.world.predict(latent, torch.tensor([action]))
                prediction = (
                    agent.world.decode(latent)["vector"][:, -1] * agent.world.std
                    + agent.world.mean
                )
                errors.append(
                    (
                        prediction
                        - torch.tensor(observations[start + step]["vector"][-1])
                    ).abs()
                )
            assert metrics[f"windows_h{step}"] == len(errors)
            for name, value in zip(
                ("cartPos", "cartVel", "poleAngle"), torch.cat(errors).mean(0)
            ):
                assert metrics[f"mae_{name}_h{step}_ood"] == pytest.approx(value.item())
    assert metrics == training.evaluate(agent, actor, config, 0, "cpu")
    for key, value in agent.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_dataset_requires_evaluation_metadata(config, tmp_path):
    write_dataset(config, tmp_path)
    dataset = torch.load(config["pretrain"]["dataset"], weights_only=False)
    dataset.pop("evaluation_environment")
    torch.save(dataset, config["pretrain"]["dataset"])
    with pytest.raises(ValueError, match="Dataset environment changed"):
        pretraining.load_dataset(config)
