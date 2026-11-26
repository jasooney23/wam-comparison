"""Single-frame observations from default Gymnasium CartPole."""

import gymnasium as gym
import numpy as np
import torch
from experiment_config import ENV_ID
from gymnasium import spaces
from torch.nn import functional as F

from src.common.environments.cartpole.observations import IMAGE_SIZE


class CartPoleObservation(gym.ObservationWrapper):
    """Expose cart position, pole angle, and optionally the current render.

    Args:
        images: Include a grayscale image alongside the vector.
        gui: Display the Gymnasium render in a window.
    """

    def __init__(self, images=True, gui=False):
        render_mode = "rgb_array" if images else "human" if gui else None
        env = gym.make(ENV_ID, render_mode=render_mode)
        if gui and images:
            env = gym.wrappers.HumanRendering(env)
        super().__init__(env)
        self.images = images
        self.observation_space = spaces.Dict(
            {"vector": spaces.Box(-np.inf, np.inf, (1, 2), np.float32)}
        )
        if images:
            self.observation_space["image"] = spaces.Box(
                0, 255, (1, IMAGE_SIZE, IMAGE_SIZE), np.uint8
            )

    def observation(self, state):
        """Convert one state to model inputs with a single-frame dimension."""
        result = {"vector": state[None, [0, 2]].astype(np.float32)}
        if self.images:
            rgb = self.unwrapped.render()
            weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
            gray = rgb.astype(np.float32) @ weights
            image = F.interpolate(
                torch.from_numpy(gray)[None, None],
                size=IMAGE_SIZE,
                mode="area",
            )
            result["image"] = image[0].round().clamp(0, 255).numpy().astype(np.uint8)
        return result
