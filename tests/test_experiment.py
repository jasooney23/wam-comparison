"""Behavior checks for physics, training modes, accounting, and resume."""

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

from common.compute import ComputeCounter
from common.environments.cartpole import CartPoleEnv
from common.environments.history import History, collect
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
    env = History(config["environment"], images=False)
    try:
        observation, info = env.reset(seed=5)
        assert env.observation_space.contains(observation)
        np.testing.assert_array_equal(
            observation["vector"], np.repeat(observation["vector"][:1], 4, axis=0)
        )
        mass = env.unwrapped.masscart
        assert 0.5 <= mass <= 1.5
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
    a = History(config["environment"], fixed=(0.25, 0.0))
    b = History(config["environment"], fixed=(2.0, 0.01))
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
    vector, image = model(batch["vector"], batch["image"], batch["action"])
    assert vector.shape == (3, 3)
    assert (image is None) == (modality == "vector")
    loss, _, _ = model.loss(batch)
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_accounting_linear_backward_and_exclusion():
    model = nn.Linear(3, 2)
    counter = ComputeCounter(model)
    model(torch.ones(4, 3)).sum().backward()
    # Forward: 48 multiply-add operations + 8 biases. Weight/bias backward: same.
    assert counter.neural == 112
    before = counter.state_dict().copy()
    with counter.excluded():
        model(torch.ones(4, 3)).sum().backward()
    assert counter.state_dict() == before
    counter.optimizer(model.parameters())
    assert counter.extra == 15 * 8
    counter.close()


def test_accounting_convolution_and_input_gradient():
    model = nn.Conv2d(1, 2, 3, bias=False)
    counter = ComputeCounter(model)
    model(torch.ones(1, 1, 4, 4, requires_grad=True)).sum().backward()
    assert counter.neural == 2 * (1 * 2 * 2 * 2) * 9 * 3
    counter.close()


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
    data, _ = collect(config["environment"], 8, 0)
    agent = Agent(modality, train_encoder=mode == "joint")
    actor, critic, ppo_loss = training.ppo_modules(agent, config["ppo"])
    with torch.no_grad():
        data = critic(actor(TensorDict(data, batch_size=[8])))
    data["advantage"] = torch.ones(8, 1)
    data["value_target"] = torch.ones(8, 1)
    if mode == "frozen":
        agent.world.requires_grad_(False)
    counter = ComputeCounter(agent)
    world_optimizer = torch.optim.Adam(agent.world.parameters(), lr=0.001)
    policy_parameters = list(agent.policy.parameters()) + list(agent.value.parameters())
    if mode == "joint":
        policy_parameters += list(agent.world.parameters())
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
    assert counter.neural > 0
    counter.close()


def test_resume_matches_uninterrupted_training(config, tmp_path):
    config["training"]["mode"] = "joint"
    data, metadata = collect(config["environment"], 16, 0)
    held_out, _ = collect(config["environment"], 8, 20, fixed=(1.0, 0.0025))
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
    for key in ("neural", "extra", "phases"):
        assert complete["counter"][key] == resumed["counter"][key]
    assert complete["transitions"] == resumed["transitions"]
    assert (directory / "curves.png").is_file()

    # Reject checkpoints from the custom backend.
    complete.pop("ppo_backend")
    legacy = directory / "legacy.pt"
    torch.save(complete, legacy)
    with pytest.raises(ValueError, match="previous PPO backend"):
        training.run_arm(config, 0, "vector", payload, directory, legacy)


def test_joint_ppo_loss_reaches_encoder(config):
    agent = Agent("vector", train_encoder=True)
    actor, critic, ppo_loss = training.ppo_modules(agent, config["ppo"])
    data = TensorDict({"vector": torch.randn(4, 4, 3)}, batch_size=[4])
    with torch.no_grad():
        critic(actor(data))
    data["advantage"] = torch.ones(4, 1)
    data["value_target"] = torch.ones(4, 1)

    # No prediction loss contributes to either gradient.
    losses = ppo_loss(data)
    losses["loss_objective"].backward(retain_graph=True)
    assert agent.world.projection.weight.grad.abs().sum() > 0
    agent.zero_grad(set_to_none=True)
    losses["loss_critic"].backward()
    assert agent.world.projection.weight.grad.abs().sum() > 0


def test_single_transition_rollout_bootstraps_time_limit(config):
    config["environment"]["max_steps"] = 1
    config["training"]["episodes"] = 1
    agent = Agent("vector")
    actor, critic, _ = training.ppo_modules(agent, config["ppo"])
    counter = ComputeCounter(agent)
    env = History(config["environment"], images=False, counter=counter)
    try:
        data, episodes, scores = training.rollout(
            agent, actor, critic, env, config, 0, 0, "cpu", counter
        )
    finally:
        env.close()
        counter.close()
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


def test_accounting_transpose_convolution():
    model = nn.ConvTranspose2d(2, 3, 4, 2, 1, bias=False)
    counter = ComputeCounter(model)
    model(torch.ones(1, 2, 2, 2)).sum().backward()
    # 8 input elements * 3 output channels * 16 kernel positions * 2,
    # repeated for the weight gradient (input has no requested gradient).
    assert counter.neural == 8 * 3 * 16 * 2 * 2
    counter.close()


def test_cached_data_rejects_changed_physics(config, tmp_path):
    training.shared_data(config, 0, tmp_path)
    config["environment"]["cart_mass"] = [0.6, 1.4]
    with pytest.raises(ValueError, match="Data configuration changed"):
        training.shared_data(config, 0, tmp_path)
