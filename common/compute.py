"""Shape-based arithmetic accounting, independent of CUDA kernel selection."""

from contextlib import contextmanager

from torch import nn


class ComputeCounter:
    """Count nominal dense arithmetic in executed network layers.

    Args:
        model: Network whose layers should be instrumented.

    Linear and convolution multiply-adds count as two operations, including
    nominal padded kernel positions. Backward counts weight and, when needed,
    input gradients. Activation and bias work uses documented scalar estimates.
    This is an arithmetic convention, not a hardware instruction measurement.
    """

    def __init__(self, model):
        self.neural = 0
        self.extra = 0
        self.enabled = True
        self.phase = "pretrain"
        self.phases = {}
        self.data_seconds = 0.0
        self.handles = []
        for layer in model.modules():
            if isinstance(
                layer, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d, nn.ReLU, nn.Sigmoid)
            ):
                self.handles.append(layer.register_forward_hook(self._hook))

    def add(self, count, extra=False):
        """Record arithmetic in the current training phase."""
        if self.enabled:
            name = "extra" if extra else "neural"
            setattr(self, name, getattr(self, name) + int(count))
            key = f"{self.phase}_{name}"
            self.phases[key] = self.phases.get(key, 0) + int(count)

    def _hook(self, layer, inputs, output):
        if not self.enabled:
            return
        elements = output.numel()
        if isinstance(layer, nn.Linear):
            multiply_adds = 2 * elements * layer.in_features
        elif isinstance(layer, nn.ConvTranspose2d):
            multiply_adds = 2 * inputs[0].numel() * (layer.out_channels // layer.groups)
            multiply_adds *= layer.kernel_size[0] * layer.kernel_size[1]
        elif isinstance(layer, nn.Conv2d):
            multiply_adds = 2 * elements * (layer.in_channels // layer.groups)
            multiply_adds *= layer.kernel_size[0] * layer.kernel_size[1]
        else:
            multiply_adds = 0
        if isinstance(layer, (nn.ReLU, nn.Sigmoid)):
            self.add(elements * (1 if isinstance(layer, nn.ReLU) else 4))
            backward = elements * (1 if isinstance(layer, nn.ReLU) else 3)
        else:
            bias = elements if layer.bias is not None else 0
            self.add(multiply_adds + bias)
            backward = multiply_adds * (
                int(layer.weight.requires_grad) + int(inputs[0].requires_grad)
            )
            backward += (
                bias if layer.bias is not None and layer.bias.requires_grad else 0
            )
        if output.requires_grad:
            output.register_hook(lambda grad: self._backward(grad, backward))

    def _backward(self, grad, count):
        self.add(count)
        return grad

    def optimizer(self, parameters):
        """Estimate Adam arithmetic as 15 scalar operations per updated parameter."""
        self.add(
            15 * sum(p.numel() for p in parameters if p.grad is not None), extra=True
        )

    @contextmanager
    def excluded(self):
        """Exclude evaluation from all training counters."""
        previous = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = previous

    def state_dict(self):
        """Return counters needed to resume a run."""
        return {
            "neural": self.neural,
            "extra": self.extra,
            "phases": dict(self.phases),
            "data_seconds": self.data_seconds,
        }

    def load_state_dict(self, state):
        """Restore cumulative counters."""
        self.neural, self.extra, self.phases = (
            state["neural"],
            state["extra"],
            state["phases"],
        )
        self.data_seconds = state["data_seconds"]

    def close(self):
        """Remove instrumentation hooks."""
        for handle in self.handles:
            handle.remove()
