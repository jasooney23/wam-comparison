"""Behavior checks for physics, training modes, measurement, and resume."""

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
from common.models import Agent, WorldModel

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


@pytest.mark.parametrize("modality", ["vector", "image", "combined"])
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
    if model.has_vector:
        assert predictions["vector"].shape == (3, 4, 3)
    if model.has_image:
        assert predictions["image"].shape == (3, 4, 64, 64)
    batch = {
        key: value
        for key, value in batch.items()
        if (model.has_vector or "vector" not in key)
        and (model.has_image or "image" not in key)
    }
    loss, metrics = model.loss(batch)
    torch.testing.assert_close(
        loss, metrics["reconstruction_loss"] + metrics["dynamics_loss"]
    )
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


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


@pytest.mark.parametrize("mode", ["frozen", "alternating", "joint"])
@pytest.mark.parametrize("modality", ["vector", "image", "combined"])
def test_training_mode_updates_only_intended_parameters(config, mode, modality):
    torch.manual_seed(0)
    config["training"]["mode"] = mode
    data, _ = random_rollouts(config["environment"], 8, 0)
    agent = Agent(modality)
    actor, critic, ppo_loss = training.ppo_modules(agent, config["ppo"])
    with torch.no_grad():
        data = critic(actor(TensorDict(data, batch_size=[8])))
    data["advantage"] = torch.ones(8, 1)
    data["value_target"] = torch.ones(8, 1)
    if mode == "frozen":
        agent.world.requires_grad_(False)
    counter = {"world_updates": 0, "agent_updates": 0}
    world_optimizer = torch.optim.Adam(agent.world.parameters(), lr=0.001)
    policy_parameters = list(agent.policy.parameters()) + list(agent.value.parameters())
    policy_optimizer = torch.optim.Adam(policy_parameters, lr=0.001)
    before_world = [p.clone() for p in agent.world.parameters()]
    before_policy = [p.clone() for p in agent.policy.parameters()]
    training.update(
        agent, data, policy_optimizer, world_optimizer, config, counter, "cpu", ppo_loss
    )
    world_changed = any(
        not torch.equal(old, new)
        for old, new in zip(before_world, agent.world.parameters())
    )
    assert world_changed == (mode != "frozen")
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before_policy, agent.policy.parameters())
    )
    assert counter["agent_updates"] > 0
    assert (counter["world_updates"] > 0) == (mode != "frozen")


def test_resume_matches_uninterrupted_training(config, tmp_path):
    config["training"]["mode"] = "joint"
    data, metadata = random_rollouts(config["environment"], 16, 0)
    held_out, _ = random_rollouts(
        config["environment"], 8, 20, fixed_params=(1.0, 0.0025)
    )
    payload = {"training": data, "held_out": [held_out], "collection": metadata}
    directory = tmp_path / "arm"
    training.run_arm(config, 0, "vector", payload, directory)
    complete = torch.load(directory / "episode-000002.pt", weights_only=False)
    training.run_arm(
        config, 0, "vector", payload, directory, directory / "episode-000001.pt"
    )
    resumed = torch.load(directory / "episode-000002.pt", weights_only=False)
    for key in complete["agent"]:
        torch.testing.assert_close(
            complete["agent"][key], resumed["agent"][key], rtol=0, atol=0
        )
    assert complete["counter"] == resumed["counter"]
    assert complete["transitions"] == resumed["transitions"]
    assert (directory / "curves.png").is_file()

    # Reject checkpoints from the custom backend.
    complete.pop("ppo_backend")
    legacy = directory / "legacy.pt"
    torch.save(complete, legacy)
    with pytest.raises(ValueError, match="previous PPO backend"):
        training.run_arm(config, 0, "vector", payload, directory, legacy)


@pytest.mark.parametrize("modality", ["vector", "image", "combined"])
def test_ppo_and_world_losses_have_separate_gradients(config, modality):
    agent = Agent(modality)
    actor, critic, ppo_loss = training.ppo_modules(agent, config["ppo"])
    data, _ = random_rollouts(config["environment"], 4, 0)
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


def test_bad_config_fails_before_collection(config):
    config["training"]["mode"] = "unknown"
    with pytest.raises(ValueError, match="training.mode"):
        training.validate(config)


def test_cached_data_rejects_changed_physics(config, tmp_path):
    training.shared_data(config, 0, tmp_path)
    config["environment"]["cart_mass"] = [0.6, 1.4]
    with pytest.raises(ValueError, match="Data configuration changed"):
        training.shared_data(config, 0, tmp_path)


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
    data, _ = random_rollouts(config["environment"], 4, 0)
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


@pytest.mark.parametrize("modality", ["vector", "image", "combined"])
def test_evaluation_is_repeatable_and_leaves_training_unchanged(config, modality):
    agent = Agent(modality)
    actor, _, _ = training.ppo_modules(agent, config["ppo"])
    held_out, _ = random_rollouts(
        config["environment"], 5, 20, fixed_params=config["evaluation"]["settings"][0]
    )
    before = {key: value.clone() for key, value in agent.state_dict().items()}
    metrics = training.evaluate(agent, actor, [held_out], config, 0, "cpu")
    assert metrics == training.evaluate(agent, actor, [held_out], config, 0, "cpu")
    for key, value in agent.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert metrics["return_ood"] == 5
    _, expected = agent.world.loss(held_out)
    for key, value in expected.items():
        assert metrics[f"{key}_ood"] == pytest.approx(value.item())
    assert not any("familiar" in key for key in metrics)


def test_equal_compute_comparison_uses_only_shared_range(tmp_path):
    def rows(costs, returns):
        return [
            {
                "transitions": cost,
                "return_ood": score,
                "reconstruction_loss_ood": score / 10,
                "dynamics_loss_ood": score / 5,
            }
            for cost, score in zip(costs, returns)
        ]

    runs = {
        "vector": [rows([10, 30, 110], [1, 3, 11]), rows([20, 40, 120], [3, 5, 13])],
        "combined": [rows([50, 80, 130], [10, 16, 26])],
    }
    matched = training.match_compute(runs)
    assert min(row["transitions"] for row in matched) == 50
    assert max(row["transitions"] for row in matched) == 110
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
            {"vector": [rows([1, 2], [1, 2])], "combined": [rows([3, 4], [3, 4])]}
        )
        == []
    )

    # Exercise the nonempty CSV and plot path as well as the interpolation.
    for modality, seeds in runs.items():
        for seed, data in enumerate(seeds):
            directory = tmp_path / f"seed-{seed}" / "joint" / modality
            directory.mkdir(parents=True)
            with (directory / "metrics.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    training.plot_comparison(tmp_path, "joint")
    with (tmp_path / "comparison-joint.csv").open(newline="") as handle:
        exported = list(csv.DictReader(handle))
    assert len(exported) == len(matched)
    assert all(row["method"] == "linear_interpolation" for row in exported)
    assert (tmp_path / "comparison-joint.png").is_file()


def test_old_experiment_artifacts_are_rejected(config, tmp_path):
    payload = training.shared_data(config, 0, tmp_path)
    payload["signature"].pop("experiment_version")
    torch.save(payload, tmp_path / "data.pt")
    with pytest.raises(ValueError, match="Data configuration changed"):
        training.shared_data(config, 0, tmp_path)
    torch.save({"ppo_backend": "torchrl"}, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="previous experiment definition"):
        training.load_checkpoint(tmp_path / "old.pt")
