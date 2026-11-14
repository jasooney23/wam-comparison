"""Checks for elapsed-time compute estimates."""

from unittest.mock import Mock

import pytest
import torch

from common import compute
from common.compute import ElapsedComputeCounter


def test_elapsed_compute_accumulates_completed_blocks(monkeypatch):
    times = iter([10.0, 12.0, 20.0, 23.0])
    monkeypatch.setattr(compute, "perf_counter", lambda: next(times))
    synchronize = Mock()
    monkeypatch.setattr(compute.cuda, "synchronize", synchronize)
    counter = ElapsedComputeCounter("cpu", flops_per_second=100)

    for _ in range(2):
        with counter.measure() as measured:
            assert measured is counter

    assert counter.elapsed_seconds == 5
    assert counter.estimated_flops == 500
    synchronize.assert_not_called()


@pytest.mark.parametrize("rate", [0, -1, float("inf"), float("nan")])
def test_invalid_rate_fails(rate):
    with pytest.raises(ValueError, match="positive and finite"):
        ElapsedComputeCounter("cpu", rate)


def test_cpu_requires_explicit_rate():
    with pytest.raises(ValueError, match="explicit flops_per_second"):
        ElapsedComputeCounter("cpu")


def test_cuda_unavailable_fails(monkeypatch):
    monkeypatch.setattr(compute.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA requested but unavailable"):
        ElapsedComputeCounter("cuda", 100)


def test_cuda_rate_and_synchronization_order(monkeypatch):
    events = []
    times = iter([10.0, 12.0])
    monkeypatch.setattr(compute.cuda, "is_available", lambda: True)
    monkeypatch.setattr(compute.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(
        compute.cuda,
        "get_device_name",
        lambda device: "NVIDIA GeForce RTX 4050 Laptop GPU",
    )
    monkeypatch.setattr(
        compute.cuda, "synchronize", lambda device: events.append(device)
    )

    def clock():
        events.append("clock")
        return next(times)

    monkeypatch.setattr(compute, "perf_counter", clock)
    counter = ElapsedComputeCounter()
    with counter.measure():
        events.append("work")

    assert events == [
        torch.device("cuda:1"),
        "clock",
        "work",
        torch.device("cuda:1"),
        "clock",
    ]
    assert counter.flops_per_second == pytest.approx(12.1344e12)
    assert counter.estimated_flops == pytest.approx(24.2688e12)
    assert ElapsedComputeCounter(flops_per_second=100).flops_per_second == 100


def test_unknown_cuda_hardware_requires_rate(monkeypatch):
    monkeypatch.setattr(compute.cuda, "is_available", lambda: True)
    monkeypatch.setattr(compute.cuda, "get_device_name", lambda device: "Other GPU")
    with pytest.raises(ValueError, match="explicit flops_per_second"):
        ElapsedComputeCounter("cuda:0")
    assert ElapsedComputeCounter("cuda:0", 100).flops_per_second == 100


def test_failed_block_counts_time_and_allows_reuse(monkeypatch):
    times = iter([0.0, 2.0, 3.0, 4.0])
    monkeypatch.setattr(compute, "perf_counter", lambda: next(times))
    counter = ElapsedComputeCounter("cpu", 100)

    with pytest.raises(ValueError, match="work failed"), counter.measure():
        raise ValueError("work failed")
    with (
        counter.measure(),
        pytest.raises(RuntimeError, match="cannot overlap"),
        counter.measure(),
    ):
        pass

    assert counter.elapsed_seconds == 3
    assert counter.estimated_flops == 300
