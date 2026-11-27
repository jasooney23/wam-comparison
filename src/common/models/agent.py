"""Agent with policy and value heads."""

import torch
from torch import nn

from .common.basic import MLP
from .world_model import WorldModel


class Agent(nn.Module):
    """World-model features with categorical policy and scalar value heads.

    Args:
        modality: World-model observation modality.
        settings: Optional CartPole observation settings.
        head_seed: Optional PyTorch seed for initializing both heads.
        world: Optional world model to use in place of the default model.
    """

    def __init__(
        self, modality: str, settings=None, *, head_seed: int | None = None, world=None
    ):
        super().__init__()
        self.world = WorldModel(modality, settings) if world is None else world

        # Match head weights across world-model modalities.
        if head_seed is not None:
            torch.manual_seed(head_seed)
        self.policy = MLP(16, 2, [32])
        self.value = MLP(16, 1, [32])

    def forward(self, vector=None, image=None):
        """Return features and logits for TorchRL's categorical actor."""
        with torch.no_grad():
            latent = self.world.encode(vector, image)
        return latent, self.policy(latent)
