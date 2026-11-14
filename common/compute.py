"""Shape-based arithmetic accounting and elapsed-time compute estimates."""

from contextlib import contextmanager
from math import isfinite
from time import perf_counter

from torch import cuda, nn
from torch import device as torch_device


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


class ElapsedComputeCounter:
    """Estimate compute as elapsed seconds times a fixed hardware FLOP/s rate.

    This measures equivalent compute capacity, including idle time inside each
    measured block, rather than executed arithmetic. CUDA synchronization waits
    for all work on the selected device at each block boundary.

    The RTX 4050 Laptop GPU default is a derived FP32 peak estimate:
    2,560 CUDA cores * 2.370 GHz maximum boost * 2 operations per multiply-add
    = 12.1344 TFLOP/s. Actual clocks and throughput vary by laptop and workload.
    This rate excludes Tensor Core and sparse arithmetic.

    Sources:
        Core count and boost clock:
        https://www.nvidia.com/en-us/geforce/laptops/compare/
        Multiply-add counting convention:
        https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html

    Args:
        device: CPU or CUDA device on which the measured workload runs.
        flops_per_second: Positive finite hardware rate. Required for CPU and
            CUDA devices other than the RTX 4050 Laptop GPU. Override the default
            when using a different clock, precision, or measured throughput.
    """

    def __init__(self, device="cuda", flops_per_second=None):
        # Resolve the device and its rate once for the whole measurement.
        self.device = torch_device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Only CPU and CUDA devices are supported")
        if self.device.type == "cuda":
            if not cuda.is_available():
                raise ValueError("CUDA requested but unavailable")
            if self.device.index is None:
                self.device = torch_device("cuda", cuda.current_device())
            name = cuda.get_device_name(self.device)
            if (
                flops_per_second is None
                and name == "NVIDIA GeForce RTX 4050 Laptop GPU"
            ):
                flops_per_second = 2560 * 2.370e9 * 2
        if flops_per_second is None:
            raise ValueError(
                "An explicit flops_per_second rate is required for this hardware"
            )
        self.flops_per_second = float(flops_per_second)
        if not isfinite(self.flops_per_second) or self.flops_per_second <= 0:
            raise ValueError("flops_per_second must be positive and finite")
        self.elapsed_seconds = 0.0
        self._measuring = False

    @property
    def estimated_flops(self):
        """Return estimated FLOPs for completed measurement blocks."""
        return self.elapsed_seconds * self.flops_per_second

    def _synchronize(self):
        if self.device.type == "cuda":
            cuda.synchronize(self.device)

    @contextmanager
    def measure(self):
        """Accumulate elapsed time, including time spent in a failing block.

        Yields:
            ElapsedComputeCounter: This counter.

        Raises:
            RuntimeError: A measurement is already running on this counter.
        """
        if self._measuring:
            raise RuntimeError("Measurement blocks on the same counter cannot overlap")
        self._synchronize()
        start = perf_counter()
        self._measuring = True
        try:
            yield self
        finally:
            try:
                self._synchronize()
                self.elapsed_seconds += perf_counter() - start
            finally:
                self._measuring = False
