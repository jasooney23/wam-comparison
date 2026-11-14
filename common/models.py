"""Small networks shared by the experiment's training modes."""

from itertools import pairwise

import torch
from torch import nn


class MLP(nn.Sequential):
    """A feedforward network with configurable hidden layers.

    Args:
        inputs: Input feature count.
        outputs: Output feature count.
        hidden_layers: Hidden layer sizes; defaults to one layer of 32.
        hidden_act_fn: Activation module class for hidden layers.
        output_act_fn: Activation module class for the output.
    """

    def __init__(
        self,
        inputs: int,
        outputs: int,
        hidden_layers: list[int] | None = None,
        hidden_act_fn=nn.ReLU,
        output_act_fn=nn.Identity,
    ):
        sizes = [
            inputs,
            *(hidden_layers if hidden_layers is not None else []),
            outputs,
        ]
        layers = []
        for index, (input_size, output_size) in enumerate(pairwise(sizes)):
            layers.extend(
                (
                    nn.Linear(input_size, output_size),
                    (output_act_fn if index == len(sizes) - 2 else hidden_act_fn)(),
                )
            )
        super().__init__(*layers)


class CNN(nn.Sequential):
    """A convolutional network with configurable hidden channel counts.

    Args:
        inputs: Input channel count.
        outputs: Output channel count.
        hidden_layers: Hidden layer channel counts.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding.
        hidden_act_fn: Activation module class for hidden layers.
        output_act_fn: Activation module class for the output.
    """

    def __init__(
        self,
        inputs: int,
        outputs: int,
        hidden_layers: list[int],
        kernel_size=3,
        stride=1,
        padding=0,
        hidden_act_fn=nn.ReLU,
        output_act_fn=nn.Identity,
    ):
        sizes = [inputs, *hidden_layers, outputs]
        layers = []
        for index, (input_size, output_size) in enumerate(pairwise(sizes)):
            layers.extend(
                (
                    nn.Conv2d(input_size, output_size, kernel_size, stride, padding),
                    (output_act_fn if index == len(sizes) - 2 else hidden_act_fn)(),
                )
            )
        super().__init__(*layers)


class TransposeCNN(nn.Sequential):
    """A transposed convolutional network with configurable channel counts.

    Args:
        inputs: Input channel count.
        outputs: Output channel count.
        hidden_layers: Hidden layer channel counts.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding.
        hidden_act_fn: Activation module class for hidden layers.
        output_act_fn: Activation module class for the output.
    """

    def __init__(
        self,
        inputs: int,
        outputs: int,
        hidden_layers: list[int],
        kernel_size=3,
        stride=1,
        padding=0,
        hidden_act_fn=nn.ReLU,
        output_act_fn=nn.Identity,
    ):
        sizes = [inputs, *hidden_layers, outputs]
        layers = []
        for index, (input_size, output_size) in enumerate(pairwise(sizes)):
            layers.extend(
                (
                    nn.ConvTranspose2d(
                        input_size, output_size, kernel_size, stride, padding
                    ),
                    (output_act_fn if index == len(sizes) - 2 else hidden_act_fn)(),
                )
            )
        super().__init__(*layers)


class WorldModel(nn.Module):
    """Encode four observations and predict the next observation given an action.

    Args:
        modality: One of vector, image, or combined.
    """

    def __init__(self, modality: str):
        super().__init__()
        if modality not in {"vector", "image", "combined"}:
            raise ValueError(f"Unknown modality: {modality}")
        self.modality = modality
        self.has_vector = modality != "image"
        self.has_image = modality != "vector"
        if self.has_vector:
            self.vector_encoder = MLP(12, 16, [32])
            self.vector_decoder = nn.Sequential(
                MLP(16, 12, [32]), nn.Unflatten(1, (4, 3))
            )
        if self.has_image:
            self.image_encoder = nn.Sequential(
                *CNN(
                    4,
                    32,
                    [16, 32],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_act_fn=nn.ReLU,
                ),
                nn.Flatten(),
                nn.Linear(32 * 8 * 8, 16),
            )
            self.image_decoder = nn.Sequential(
                MLP(16, 32 * 8 * 8, [32]),
                nn.Unflatten(1, (32, 8, 8)),
                *TransposeCNN(
                    32,
                    4,
                    [32, 16],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_act_fn=nn.Sigmoid,
                ),
            )
        self.projection = nn.Linear(16 * (self.has_vector + self.has_image), 16)
        self.dynamics = MLP(16 + 2, 16, [32])  # z = 16, cartpole act = 2
        self.register_buffer("mean", torch.zeros(3))
        self.register_buffer("std", torch.ones(3))

    def encode(self, vector=None, image=None):
        """Encode a batch of four-frame histories."""
        features = []
        if self.has_vector:
            features.append(
                self.vector_encoder(((vector - self.mean) / self.std).flatten(1))
            )
        if self.has_image:
            features.append(self.image_encoder(image.float() / 255))
        return self.projection(torch.cat(features, dim=1))

    def predict(self, latent, action):
        """Predict the next latent from the current latent and action."""
        action = torch.nn.functional.one_hot(action.long(), 2).float()
        return self.dynamics(torch.cat((latent, action), dim=1))

    def decode(self, latent):
        """Reconstruct four-frame histories for the active modalities."""
        return {
            name: getattr(self, f"{name}_decoder")(latent)
            for name, enabled in (
                ("vector", self.has_vector),
                ("image", self.has_image),
            )
            if enabled
        }

    def forward(self, vector, image, action):
        """Predict the next four-frame observation from the current history."""
        return self.decode(self.predict(self.encode(vector, image), action))

    def loss(self, batch, vector_weight=1.0, image_weight=1.0):
        """Sum reconstruction and dynamics MSE, averaging each modality separately.

        Args:
            batch: Histories, actions, and the next single frame for each modality.
            vector_weight: Weight for vector MSE in each loss.
            image_weight: Weight for image MSE in each loss.

        Returns:
            Total loss and separate reconstruction, dynamics, and modality losses.
        """
        latent = self.encode(batch.get("vector"), batch.get("image"))
        outputs = {
            "reconstruction": self.decode(latent),
            "dynamics": self.decode(self.predict(latent, batch["action"])),
        }
        metrics = {}
        for loss_kind, predictions in outputs.items():
            errors = []
            for out_mode, prediction in predictions.items():
                target = batch[out_mode]

                if loss_kind == "dynamics":
                    # target is next frame
                    target = torch.cat(
                        (target[:, 1:], batch[f"next_{out_mode}"][:, None]), 1
                    )

                if out_mode == "vector":
                    target = (target - self.mean) / self.std
                    weight = vector_weight
                else:
                    target = target.float() / 255
                    weight = image_weight

                # MSE for all modalities
                # average over each loss kind and output modality
                error = (prediction - target).square().mean()
                metrics[f"{out_mode}_{loss_kind}_loss"] = error
                errors.append(weight * error)

            metrics[f"{loss_kind}_loss"] = sum(errors)

        return metrics["reconstruction_loss"] + metrics["dynamics_loss"], metrics


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
