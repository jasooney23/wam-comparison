"""Agent with policy and value heads."""

import torch
from torch import nn

from .common.basic import MLP
from .world_model import WorldModel


class Agent(nn.Module):
    """World-model features with categorical policy and scalar value heads.

    Args:
        modality: World-model observation modality.
    """

    def __init__(self, modality: str):
        super().__init__()
        self.world = WorldModel(modality)
        self.policy = MLP(16, 2, [32])
        self.value = MLP(16, 1, [32])

    def forward(self, vector=None, image=None):
        """Return features and logits for TorchRL's categorical actor."""
        with torch.no_grad():
            latent = self.world.encode(vector, image)
        return latent, self.policy(latent)
