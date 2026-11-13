"""Small networks shared by the experiment's training modes."""

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
            *(hidden_layers if hidden_layers is not None else [32]),
            outputs,
        ]
        layers = []
        for index, (input_size, output_size) in enumerate(zip(sizes, sizes[1:])):
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
        for index, (input_size, output_size) in enumerate(zip(sizes, sizes[1:])):
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
        for index, (input_size, output_size) in enumerate(zip(sizes, sizes[1:])):
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
            self.vector_encoder = MLP(12, 32)
        # A vector target is required even in image-only mode for common metrics.
        self.vector_decoder = MLP(16, 3)
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
                nn.Linear(32 * 8 * 8, 32),
            )
            self.image_decoder = nn.Sequential(
                MLP(16, 32 * 8 * 8),
                nn.Unflatten(1, (32, 8, 8)),
                *TransposeCNN(
                    32,
                    1,
                    [32, 16],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_act_fn=nn.Sigmoid,
                ),
            )
        self.projection = nn.Linear(32 * (self.has_vector + self.has_image), 16)
        self.dynamics = MLP(18, 16)
        self.register_buffer("mean", torch.zeros(3))
        self.register_buffer("std", torch.ones(3))

    def encode(self, vector, image):
        """Encode a batch of four-frame histories."""
        features = []
        if self.has_vector:
            features.append(
                self.vector_encoder(((vector - self.mean) / self.std).flatten(1))
            )
        if self.has_image:
            features.append(self.image_encoder(image.float() / 255))
        return self.projection(torch.cat(features, dim=1))

    def forward(self, vector, image, action):
        """Predict the next normalized vector and optional image."""
        latent = self.encode(vector, image)
        action = torch.nn.functional.one_hot(action.long(), 2).float()
        future = self.dynamics(torch.cat((latent, action), dim=1))
        predicted_image = self.image_decoder(future) if self.has_image else None
        return self.vector_decoder(future), predicted_image

    def loss(self, batch, vector_weight=1.0, image_weight=1.0):
        """Compute weighted next-vector and next-image mean squared errors."""
        vector, image = self(batch["vector"], batch.get("image", None), batch["action"])
        target = (batch["next_vector"] - self.mean) / self.std
        vector_error = (vector - target).square().mean()
        image_error = vector_error.new_zeros(())
        if self.has_image:
            image_error = (
                (image[:, 0] - batch["next_image"].float() / 255).square().mean()
            )
        return (
            vector_weight * vector_error + image_weight * image_error,
            vector_error,
            image_error,
        )


class Agent(nn.Module):
    """World-model features with categorical policy and scalar value heads.

    Args:
        modality: World-model observation modality.
        train_encoder: Allow policy and value losses to update the encoder.
    """

    def __init__(self, modality: str, train_encoder: bool = False):
        super().__init__()
        self.world = WorldModel(modality)
        self.policy = MLP(16, 2)
        self.value = MLP(16, 1)
        self.train_encoder = train_encoder

    def forward(self, vector, image=None):
        """Return features and logits for TorchRL's categorical actor."""
        with torch.set_grad_enabled(self.train_encoder and torch.is_grad_enabled()):
            latent = self.world.encode(vector, image)
        return latent, self.policy(latent)
