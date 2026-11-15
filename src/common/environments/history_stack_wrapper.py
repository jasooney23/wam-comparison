"""Four-frame CartPole observations and shared trajectory collection."""

from collections import deque

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torch.nn import functional as F

from .cartpole import CartPoleEnv


class HistoryStackWrapper(gym.Wrapper):
    """Stack the past G states as the observation.

    Args:
        settings: Environment config with sampling ranges and episode limit.
        images: Whether to render image observations.
        fixed_params: Optional fixed_params cart mass and damping for evaluation.
        gui: Whether to show the environment in a window.
    """

    def __init__(self, settings, images=True, fixed_params=None, gui=False):
        base = CartPoleEnv(
            render_mode="human" if gui else "rgb_array" if images else None,
            pole_mass=settings["pole_mass"],
        )
        super().__init__(gym.wrappers.TimeLimit(base, settings["max_steps"]))
        self.settings, self.images, self.fixed_params = (
            settings,
            images,
            fixed_params,
        )
        self.vectors, self.frames = deque(maxlen=4), deque(maxlen=4)
        observation_spaces = {"vector": spaces.Box(-np.inf, np.inf, (4, 3), np.float32)}
        if images:
            observation_spaces["image"] = spaces.Box(0, 255, (4, 64, 64), np.uint8)
        self.observation_space = spaces.Dict(observation_spaces)

    def _observe(self, state, initial=False):
        vector = state[:3].copy()
        frame = None

        # Get image as grayscale
        if self.images:
            img_rgb = (
                self.unwrapped._last_frame
                if self.unwrapped.render_mode == "human"
                else self.env.render()
            )
            img_gray = np.dot(
                img_rgb.astype(np.float32), np.array([0.299, 0.587, 0.114], np.float32)
            )
            frame = F.interpolate(
                torch.from_numpy(img_gray)[None, None], size=(64, 64), mode="area"
            )
            frame = frame[0, 0].round().clamp(0, 255).numpy().astype(np.uint8)

        # Construct stacked obs
        for _ in range(4 if initial else 1):
            self.vectors.append(vector)
            if self.images:
                self.frames.append(frame)
        result = {"vector": np.stack(self.vectors)}
        if self.images:
            result["image"] = np.stack(self.frames)

        return result

    def reset(self, *, seed=None, options=None):
        """Reset history and sample physics once for the new episode."""
        state, info = self.env.reset(seed=seed, options=options)
        base = self.unwrapped

        # sample env params
        mass, damping = (
            self.fixed_params
            if self.fixed_params is not None
            else (
                base.np_random.uniform(*self.settings["cart_mass"]),
                base.np_random.uniform(*self.settings["hinge_damping"]),
            )
        )

        base.masscart, base.hinge_damping = mass, damping
        base.total_mass = mass + base.masspole
        base.polemass_length = base.masspole * base.length
        info.update(cart_mass=mass, hinge_damping=damping)
        self.vectors.clear()
        self.frames.clear()

        return self._observe(state, initial=True), info

    def step(self, action):
        """Step the parameterized physics and append the resulting observation."""
        state, reward, terminated, truncated, info = self.env.step(action)
        return self._observe(state), reward, terminated, truncated, info


def random_rollouts(env_settings, count, seed, fixed_params=None, horizon=0):
    """Collect a bunch of random transitions/windows from env.

    Args:
        env_settings: Environment config.
        count: Number of transitions/windows.
        seed: Data collection seed.
        fixed_params: Optional evaluation parameter pair.
        horizon: Future steps per window; zero keeps the one-step format.

    Returns:
        CPU tensors.
    """
    if count <= 0 or horizon < 0 or horizon >= env_settings["max_steps"]:
        raise ValueError("Count must be positive and horizon below max_steps")
    env = HistoryStackWrapper(env_settings, images=True, fixed_params=fixed_params)
    rng = np.random.default_rng(seed)
    arrays = {
        "vector": np.empty((count, 4, 3), np.float32),
        "image": np.empty((count, 4, 64, 64), np.uint8),
    }
    if horizon:
        arrays["actions"] = np.empty((count, horizon), np.int64)
        arrays["targets"] = np.empty((count, horizon + 1, 3), np.float32)
    else:
        arrays["action"] = np.empty(count, np.int64)
        arrays["next_vector"] = np.empty((count, 3), np.float32)
        arrays["next_image"] = np.empty((count, 64, 64), np.uint8)
    observation, _ = env.reset(seed=seed)
    observations, actions = [observation], []
    collected = 0
    steps = max(1, horizon)

    # collect rollouts
    try:
        while collected < count:
            action = int(rng.integers(2))
            future, _, terminated, truncated, _ = env.step(action)
            actions.append(action)
            observations.append(future)
            if len(actions) == steps:
                arrays["vector"][collected] = observations[0]["vector"]
                arrays["image"][collected] = observations[0]["image"]
                if horizon:
                    arrays["actions"][collected] = actions
                    arrays["targets"][collected] = [
                        item["vector"][-1] for item in observations
                    ]
                else:
                    arrays["action"][collected] = action
                    arrays["next_vector"][collected] = future["vector"][-1]
                    arrays["next_image"][collected] = future["image"][-1]
                collected += 1
                observations.pop(0)
                actions.pop(0)
            if terminated or truncated:
                if collected < count:
                    observation, _ = env.reset()
                    observations, actions = [observation], []
    finally:
        env.close()
    return {key: torch.from_numpy(value) for key, value in arrays.items()}
