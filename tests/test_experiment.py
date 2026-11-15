"""Behavior checks for physics, training, measurement, and resume."""

import csv
import importlib.util
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium.envs.classic_control.cartpole import CartPoleEnv as UpstreamCartPole
from tensordict import TensorDict
from torch import nn
from torchrl.objectives.value import GAE

from common.environments.cartpole import CartPoleEnv
from common.environments.history_stack_wrapper import (
    HistoryStackWrapper,
    random_rollouts,
)
from common.models import MLP, Agent, WorldModel

EXPERIMENT = (
    Path(__file__).resolve().parents[1] / "src/experiments/visual-augmentation-simple"
)
SPEC = importlib.util.spec_from_file_location(
    "visual_training", EXPERIMENT / "train.py"
)
training = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(training)
torch.set_num_threads(1)


@pytest.fixture
def config(tmp_path):
    with (EXPERIMENT / "config.toml").open("rb") as handle:
        result = training.smoke_config(tomllib.load(handle))
    result["run"].update(device="cpu", output=str(tmp_path))
    result["environment"]["max_steps"] = 5
    result["evaluation"]["settings"] = [[1.0, 0.0025]]
    return result


def test_synthesized_dataset_is_reproducible(config):
    spec = importlib.util.spec_from_file_location(
        "dataset_generator", EXPERIMENT / "generate_dataset.py"
    )
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    first = generator.generate(config["environment"], 4, 17)
    second = generator.generate(config["environment"], 4, 17)
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    assert first["image"].shape == (4, 4, 64, 64)
    assert first["next_vector"].shape == (4, 3)
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
    env = HistoryStackWrapper(config["environment"], images=False)
    try:
        observation, info = env.reset(seed=5)
        assert env.observation_space.contains(observation)
        np.testing.assert_array_equal(
            observation["vector"], np.repeat(observation["vector"][:1], 4, axis=0)
        )
        mass = env.unwrapped.masscart
        assert (
            config["environment"]["cart_mass"][0]
            <= mass
            <= config["environment"]["cart_mass"][1]
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
    a = HistoryStackWrapper(config["environment"], fixed_params=(0.25, 0.0))
    b = HistoryStackWrapper(config["environment"], fixed_params=(2.0, 0.01))
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
        "image": torch.randint(256, (3, 4, 64, 64), dtype=torch.uint8),
        "action": torch.tensor([0, 1, 0]),
        "next_vector": torch.randn(3, 3),
        "next_image": torch.randint(256, (3, 64, 64), dtype=torch.uint8),
    }
    assert model.encode(batch["vector"], batch["image"]).shape == (3, 16)
    predictions = model(batch["vector"], batch["image"], batch["action"])
    assert predictions["vector"].shape == (3, 4, 3)
    if model.has_image:
        assert predictions["image"].shape == (3, 4, 64, 64)
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


def test_mlp_default_and_explicit_linear_model():
    model = MLP(3, 2)
    layers = [layer for layer in model if isinstance(layer, nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in layers] == [
        (3, 32),
        (32, 2),
    ]
    linear = MLP(3, 2, [])
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
    data = random_rollouts(config["environment"], 8, 0)
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


def test_resume_matches_uninterrupted_training(config, tmp_path):
    data = random_rollouts(config["environment"], 16, 0)
    held_out = random_rollouts(
        config["environment"],
        8,
        20,
        fixed_params=(1.0, 0.0025),
        horizon=config["evaluation"]["horizon"],
    )
    payload = {"training": data, "held_out": [held_out]}
    directory = tmp_path / "arm"
    training.run_arm(config, 0, "vector", payload, directory)
    complete = torch.load(directory / "episode-000002.pt", weights_only=False)
    payload.pop("training")
    training.run_arm(
        config, 0, "vector", payload, directory, directory / "episode-000001.pt"
    )
    resumed = torch.load(directory / "episode-000002.pt", weights_only=False)
    for key in complete["agent"]:
        torch.testing.assert_close(
            complete["agent"][key], resumed["agent"][key], rtol=0, atol=0
        )
    for key in ("world_updates", "agent_updates", "world_training_seconds"):
        assert complete["counter"][key] == resumed["counter"][key]
    assert complete["counter"]["agent_training_seconds"] > 0
    assert resumed["counter"]["agent_training_seconds"] > 0
    assert complete["agent_environment_steps"] == resumed["agent_environment_steps"]
    assert (directory / "curves.png").is_file()
    with (directory / "metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert int(rows[0]["agent_environment_steps"]) == 0
    assert float(rows[0]["agent_training_seconds"]) == 0
    assert float(rows[0]["world_training_seconds"]) > 0
    assert int(rows[-1]["agent_environment_steps"]) > 0
    assert float(rows[-1]["agent_training_seconds"]) > 0
    for row in rows:
        assert float(row["total_training_seconds"]) == pytest.approx(
            float(row["world_training_seconds"]) + float(row["agent_training_seconds"])
        )
    assert "transitions" not in rows[0]
    assert "training_seconds" not in rows[0]

    # Reject checkpoints from the custom backend.
    complete.pop("ppo_backend")
    legacy = directory / "legacy.pt"
    torch.save(complete, legacy)
    with pytest.raises(ValueError, match="previous PPO backend"):
        training.run_arm(config, 0, "vector", payload, directory, legacy)


@pytest.mark.parametrize("modality", ["vector", "combined"])
def test_ppo_and_world_losses_have_separate_gradients(config, modality):
    agent = Agent(modality)
    actor, critic, ppo_loss = training.ppo_modules(agent, config["ppo"])
    data = random_rollouts(config["environment"], 4, 0)
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
    config["training"]["episodes"] = 1
    agent = Agent("vector")
    actor, critic, _ = training.ppo_modules(agent, config["ppo"])
    env = HistoryStackWrapper(config["environment"], images=False)
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
    data = random_rollouts(config["environment"], 32, 0)
    path = tmp_path / "dataset.pt"
    torch.save(
        {
            "environment": config["environment"],
            "training": data,
        },
        path,
    )
    config["training"]["dataset"] = str(path)


def test_old_dataset_collection_field_is_ignored(config, tmp_path):
    write_dataset(config, tmp_path)
    path = tmp_path / "dataset.pt"
    dataset = torch.load(path, weights_only=False)
    dataset["collection"] = {"collection_seconds": 123.0}
    torch.save(dataset, path)
    payload = training.shared_data(config, 0, tmp_path)
    assert "collection" not in payload


def test_cached_data_rejects_changed_physics(config, tmp_path):
    write_dataset(config, tmp_path)
    training.shared_data(config, 0, tmp_path)
    config["environment"]["cart_mass"] = [0.6, 1.4]
    with pytest.raises(ValueError, match="Dataset environment"):
        training.shared_data(config, 0, tmp_path)


@pytest.mark.parametrize("modality, version", [("vector", 8), ("combined", 9)])
def test_checkpoint_only_training_needs_no_dataset(
    config, tmp_path, monkeypatch, modality, version
):
    write_dataset(config, tmp_path)
    payload = training.shared_data(config, 0, tmp_path)
    source = tmp_path / "source"
    training.run_arm(config, 0, modality, payload, source)
    checkpoint = source / "world-epoch-001.pt"
    saved = torch.load(checkpoint, weights_only=False)
    saved["experiment_version"] = version
    torch.save(saved, checkpoint)

    # Both cached and freshly collected evaluation data work without training data.
    config["training"]["dataset"] = str(tmp_path / "missing.pt")
    cached = training.shared_data(config, 0, tmp_path, load_training=False)
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    payload = training.shared_data(config, 0, fresh, load_training=False)
    assert "training" not in payload and "training" not in cached
    for key, value in cached["held_out"][0].items():
        torch.testing.assert_close(value, payload["held_out"][0][key])

    def reject_pretraining(*args, **kwargs):
        raise AssertionError("Checkpoint-only training must not pretrain")

    monkeypatch.setattr(training, "pretrain", reject_pretraining)
    destination = tmp_path / "new-agent"
    training.run_arm(
        config, 1, modality, payload, destination, world_checkpoint=checkpoint
    )
    result = torch.load(destination / "episode-000002.pt", weights_only=False)
    for key, value in saved["world"].items():
        torch.testing.assert_close(
            result["agent"][f"world.{key}"], value, rtol=0, atol=0
        )
    assert result["counter"]["world_updates"] == saved["world_updates"]
    assert (
        result["counter"]["world_training_seconds"] == saved["world_training_seconds"]
    )
    assert result["counter"]["agent_updates"] > 0


def test_pretraining_evaluates_periodic_and_final_checkpoints(config, tmp_path):
    config["training"].update(pretrain_epochs=21, batch_size=4)
    data = random_rollouts(config["environment"], 4, 0)
    held_out = [
        random_rollouts(
            config["environment"],
            2,
            20,
            fixed_params=config["evaluation"]["settings"][0],
            horizon=4,
        )
    ]
    agent = Agent("vector")
    optimizer = torch.optim.Adam(agent.world.parameters(), lr=0.001)
    counter = {"world_updates": 0, "world_training_seconds": 0.0}
    with training.SummaryWriter(str(tmp_path / "tensorboard")) as writer:
        training.pretrain(
            agent,
            data,
            optimizer,
            config,
            counter,
            writer,
            torch.device("cpu"),
            tmp_path,
            0,
            "vector",
            held_out,
        )
    with (tmp_path / "world-metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["epoch"]) for row in rows] == [20, 21]
    for row in rows:
        epoch = int(row["epoch"])
        state = torch.load(tmp_path / f"world-epoch-{epoch:03d}.pt", weights_only=False)
        agent.world.load_state_dict(state["world"])
        expected = training.evaluate_world(agent.world, held_out, config, "cpu")
        for key, value in expected.items():
            assert float(row[key]) == pytest.approx(value)
        assert int(row["world_updates"]) == state["world_updates"] == epoch
        assert float(row["world_training_seconds"]) == state["world_training_seconds"]


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
        .expand(2, 4, 64, 64),
        "next_image": torch.full((2, 64, 64), 128, dtype=torch.uint8),
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
    data = random_rollouts(config["environment"], 4, 0)
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


@pytest.mark.parametrize("pair", [[1.0, 0.0], [1.01, 0.0001]])
def test_evaluation_rejects_training_parameters(config, pair):
    config["evaluation"]["settings"] = [pair]
    with pytest.raises(ValueError, match="outside the training"):
        training.validate(config)


@pytest.mark.parametrize("modality", ["vector", "combined"])
def test_evaluation_is_repeatable_and_leaves_training_unchanged(
    config, modality, monkeypatch
):
    config["evaluation"]["horizon"] = 0
    agent = Agent(modality)
    actor, _, _ = training.ppo_modules(agent, config["ppo"])
    held_out = random_rollouts(
        config["environment"], 5, 20, fixed_params=config["evaluation"]["settings"][0]
    )
    before = {key: value.clone() for key, value in agent.state_dict().items()}

    def reject_dynamics(*args, **kwargs):
        raise AssertionError("Horizon zero must not advance the dynamics")

    monkeypatch.setattr(agent.world, "predict", reject_dynamics)
    metrics = training.evaluate(agent, actor, [held_out], config, 0, "cpu")
    assert metrics == training.evaluate(agent, actor, [held_out], config, 0, "cpu")
    for key, value in agent.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert metrics["return_ood"] == 5
    with torch.no_grad():
        latent = agent.world.encode(held_out["vector"], held_out["image"])
        prediction = agent.world.decode(latent)["vector"][:, -1]
        expected = (
            (
                prediction * agent.world.std
                + agent.world.mean
                - held_out["vector"][:, -1]
            )
            .abs()
            .mean(0)
        )
    for name, value in zip(training.STATE_NAMES, expected):
        assert metrics[f"{name}_l2_ood"] == pytest.approx(value.item())
    assert not any("loss" in key for key in metrics)
    assert not any("familiar" in key for key in metrics)


def test_horizon_windows_and_metrics(config):
    config["evaluation"]["horizon"] = 4
    data = random_rollouts(
        config["environment"],
        5,
        20,
        fixed_params=config["evaluation"]["settings"][0],
        horizon=4,
    )
    assert data["actions"].shape == (5, 4)
    assert data["targets"].shape == (5, 5, 3)
    torch.testing.assert_close(data["targets"][:, 0], data["vector"][:, -1])

    agent = Agent("vector")
    actor, _, _ = training.ppo_modules(agent, config["ppo"])
    metrics = training.evaluate(agent, actor, [data], config, 0, "cpu")
    for step in range(5):
        for name in training.STATE_NAMES:
            assert f"{name}_l2_h{step}_ood" in metrics
    with torch.no_grad():
        latent = agent.world.encode(data["vector"])
        for step in range(5):
            if step:
                latent = agent.world.predict(latent, data["actions"][:, step - 1])
            prediction = agent.world.decode(latent)["vector"][:, -1]
            expected = (
                (
                    prediction * agent.world.std
                    + agent.world.mean
                    - data["targets"][:, step]
                )
                .abs()
                .mean(0)
            )
            for name, error in zip(training.STATE_NAMES, expected):
                assert metrics[f"{name}_l2_h{step}_ood"] == pytest.approx(error.item())
    assert not any("image" in key or "loss" in key for key in metrics)

    # Horizon zero uses the same reconstruction as the start of a longer rollout.
    config["evaluation"]["horizon"] = 0
    immediate = training.evaluate_world(agent.world, [data], config, "cpu")
    for name in training.STATE_NAMES:
        assert immediate[f"{name}_l2_ood"] == metrics[f"{name}_l2_h0_ood"]


@pytest.mark.parametrize(
    "budget", ["agent_environment_steps", "total_training_seconds"]
)
def test_equal_compute_comparison_uses_only_shared_range(tmp_path, budget):
    def rows(costs, returns):
        return [
            {
                budget: cost,
                "return_ood": score,
                "cart_position_l2_ood": score / 10,
                "cart_velocity_l2_ood": score / 5,
                "pole_angle_l2_ood": score / 20,
            }
            for cost, score in zip(costs, returns)
        ]

    runs = {
        "vector": [rows([10, 30, 110], [1, 3, 11]), rows([20, 40, 120], [3, 5, 13])],
        "combined": [rows([50, 80, 130], [10, 16, 26])],
    }
    matched = training.match_compute(runs, budget)
    assert min(row[budget] for row in matched) == 50
    assert max(row[budget] for row in matched) == 110
    vector = next(
        row
        for row in matched
        if row["modality"] == "vector" and row["metric"] == "return_ood"
    )
    assert vector["mean"] == pytest.approx(5.5)
    assert vector["std"] == pytest.approx(0.5)
    assert vector["seeds"] == 2
    assert (
        training.match_compute(
            {"vector": [rows([1, 2], [1, 2])], "combined": [rows([3, 4], [3, 4])]},
            budget,
        )
        == []
    )

    # Exercise the nonempty CSV and plot path as well as the interpolation.
    for modality, seeds in runs.items():
        for seed, data in enumerate(seeds):
            directory = tmp_path / f"seed-{seed}" / modality
            directory.mkdir(parents=True)
            with (directory / "metrics.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    training.plot_comparison(tmp_path, budget)
    filename = (
        "comparison" if budget == "agent_environment_steps" else "comparison-time"
    )
    with (tmp_path / f"{filename}.csv").open(newline="") as handle:
        exported = list(csv.DictReader(handle))
    assert len(exported) == len(matched)
    assert all(row["method"] == "linear_interpolation" for row in exported)
    assert (tmp_path / f"{filename}.png").is_file()


def test_comparison_accepts_rollout_metrics(tmp_path):
    for modality in ("vector", "combined"):
        directory = tmp_path / "seed-0" / modality
        directory.mkdir(parents=True)
        row = {"agent_environment_steps": 10, "return_ood": 5}
        row.update({key: 1.0 for key in training.evaluation_metrics(4)[1:]})
        with (directory / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    training.plot_comparison(tmp_path)
    with (tmp_path / "comparison.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["metric"] for row in rows} == set(training.evaluation_metrics(4))


def test_old_experiment_artifacts_are_rejected(config, tmp_path):
    write_dataset(config, tmp_path)
    payload = training.shared_data(config, 0, tmp_path)
    payload["signature"].pop("experiment_version")
    torch.save(payload, tmp_path / "data.pt")
    with pytest.raises(ValueError, match="Data configuration changed"):
        training.shared_data(config, 0, tmp_path)
    torch.save({"ppo_backend": "torchrl"}, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="previous experiment definition"):
        training.load_checkpoint(tmp_path / "old.pt")
