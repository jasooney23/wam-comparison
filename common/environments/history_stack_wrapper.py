"""Four-frame CartPole observations and shared trajectory collection."""

from collections import deque
from time import perf_counter

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
        counter: Optional training operation counter.
        gui: Whether to show the environment in a window.
    """

    def __init__(
        self, settings, images=True, fixed_params=None, counter=None, gui=False
    ):
        base = CartPoleEnv(
            render_mode="human" if gui else "rgb_array" if images else None,
            pole_mass=settings["pole_mass"],
        )
        super().__init__(gym.wrappers.TimeLimit(base, settings["max_steps"]))
        self.settings, self.images, self.fixed_params, self.counter = (
            settings,
            images,
            fixed_params,
            counter,
        )
        self.vectors, self.frames = deque(maxlen=4), deque(maxlen=4)
        self.times = {
            "render_seconds": 0.0,
            "preprocess_seconds": 0.0,
            "environment_seconds": 0.0,
        }
        observation_spaces = {"vector": spaces.Box(-np.inf, np.inf, (4, 3), np.float32)}
        if images:
            observation_spaces["image"] = spaces.Box(0, 255, (4, 64, 64), np.uint8)
        self.observation_space = spaces.Dict(observation_spaces)

    def _observe(self, state, initial=False):
        vector = state[:3].copy()
        frame = None

        # Get image as grayscale
        if self.images:
            start = perf_counter()
            img_rgb = (
                self.unwrapped._last_frame
                if self.unwrapped.render_mode == "human"
                else self.env.render()
            )
            self.times["render_seconds"] += perf_counter() - start
            start = perf_counter()
            img_gray = np.dot(
                img_rgb.astype(np.float32), np.array([0.299, 0.587, 0.114], np.float32)
            )
            frame = F.interpolate(
                torch.from_numpy(img_gray)[None, None], size=(64, 64), mode="area"
            )
            frame = frame[0, 0].round().clamp(0, 255).numpy().astype(np.uint8)
            self.times["preprocess_seconds"] += perf_counter() - start
            if self.counter:
                # RGB weighted sum plus area-resize accumulation and scaling.
                self.counter.add(
                    8 * img_rgb.shape[0] * img_rgb.shape[1] + 64 * 64, extra=True
                )

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
        start = perf_counter()
        state, reward, terminated, truncated, info = self.env.step(action)
        self.times["environment_seconds"] += perf_counter() - start
        if self.counter:
            self.counter.add(40, extra=True)
        return self._observe(state), reward, terminated, truncated, info


def random_rollouts(settings, n_rollouts, seed, fixed_params=None):
    """Collect a bunch of random rollouts from env.

    Args:
        settings: Environment config.
        count: Number of transitions.
        seed: Data collection seed.
        fixed_params: Optional evaluation parameter pair.

    Returns:
        CPU tensors and collection timing metadata.
    """
    env = HistoryStackWrapper(settings, images=True, fixed_params=fixed_params)
    rng = np.random.default_rng(seed)
    arrays = {
        "vector": np.empty((n_rollouts, 4, 3), np.float32),
        "image": np.empty((n_rollouts, 4, 64, 64), np.uint8),
        "action": np.empty(n_rollouts, np.int64),
        "next_vector": np.empty((n_rollouts, 3), np.float32),
        "next_image": np.empty((n_rollouts, 64, 64), np.uint8),
    }
    start = perf_counter()
    observation, _ = env.reset(seed=seed)
    frames = 1

    # Generate rollouts
    try:
        for i in range(n_rollouts):
            action = int(rng.integers(2))
            future, _, terminated, truncated, _ = env.step(action)
            frames += 1
            for key in ("vector", "image"):
                arrays[key][i] = observation[key]
                arrays[f"next_{key}"][i] = future[key][-1]
            arrays["action"][i] = action
            observation = future
            if (terminated or truncated) and i + 1 < n_rollouts:
                observation, _ = env.reset()
                frames += 1
    finally:
        env.close()

    metadata = dict(
        env.times,
        collection_seconds=perf_counter() - start,
        estimated_operations=40 * n_rollouts + frames * (8 * 600 * 400 + 64 * 64),
    )
    return {key: torch.from_numpy(value) for key, value in arrays.items()}, metadata
