"""Small networks shared by the experiment's training modes."""

from itertools import pairwise

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
