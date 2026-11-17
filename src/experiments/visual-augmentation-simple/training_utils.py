"""Shared logging, model measurements, and run setup for both training stages."""

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter

# Allow direct execution from the experiment directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

EXPERIMENT_VERSION = 11
MODALITIES = ("vector", "combined")
TENSORBOARD_STATE_NAMES = ("cartPos", "cartVel", "poleAngle")


class MetricWriter:
    """Write every scalar to CSV and TensorBoard.

    Args:
        directory: New directory for one seed and modality.
    """

    def __init__(self, directory):
        self.handle = (directory / "metrics.csv").open("x", newline="")
        self.csv = csv.writer(self.handle)
        self.csv.writerow(("metric", "step", "value"))
        self.writer = SummaryWriter(str(directory / "tensorboard"))

    def add_scalar(self, metric, value, step):
        """Record a scalar with the same tag and step in both destinations."""
        self.writer.add_scalar(metric, value, step)
        self.csv.writerow((metric, step, float(value)))
        self.handle.flush()

    def flush(self):
        """Flush pending metrics."""
        self.handle.flush()
        self.writer.flush()

    def close(self):
        """Close both metric outputs."""
        self.writer.close()
        self.handle.close()


def setup_device(config):
    """Configure Torch threads and check the requested device."""
    torch.set_num_threads(config["run"]["cpu_threads"])
    device = torch.device(config["run"]["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; configure device='cpu'")
        probe = torch.ones(4, device=device, requires_grad=True)
        probe.square().sum().backward()
    return device


def seed_run(seed):
    """Reset random generators for one seed and modality."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_directory(output, seed, modality):
    """Return the stage directory for one seed and modality."""
    return Path(output) / f"seed-{seed}" / modality


def check_output(output, seeds):
    """Reject existing run directories before starting either modality."""
    for seed in seeds:
        for modality in MODALITIES:
            directory = run_directory(output, seed, modality)
            if directory.exists():
                raise FileExistsError(
                    f"Run already exists: {directory}; choose a new output"
                )


def prepare_run(directory, config):
    """Create a fresh run directory and save its configuration."""
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def save_snapshot(path, state):
    """Publish a snapshot only after its complete contents have been written."""
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def to_device(data, indices, device, world):
    """Select a batch and transfer only the model's required modalities."""
    return TensorDict(
        {
            key: value
            for key, value in data.items()
            if world.has_image or "image" not in key
        },
        batch_size=[len(data["vector"])],
    )[indices].to(device)


def checked_step(loss, optimizer, parameters, max_norm):
    """Backpropagate finite losses and fail on nonfinite gradients."""
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training loss")
    parameters = list(parameters)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
    optimizer.step()


@torch.no_grad()
def next_step_mae(world, data, config, device):
    """Measure action-conditioned next-step MAE in physical units.

    Args:
        world: World model to evaluate.
        data: Histories, stored actions, and next frames.
        config: Experiment configuration.
        device: Model device.

    Returns:
        One mean absolute error per state variable.
    """
    totals = torch.zeros(3, device=device)
    count = len(data["vector"])
    for indices in torch.arange(count).split(config["pretrain_eval"]["batch_size"]):
        batch = to_device(data, indices, device, world)
        latent = world.encode(batch["vector"], batch.get("image"))
        future = world.predict(latent, batch["action"])
        prediction = world.decode(future)["vector"][:, -1]
        totals += (prediction - batch["next_vector"]).abs().sum(0)
    return dict(zip(TENSORBOARD_STATE_NAMES, (totals / count).tolist()))


@torch.no_grad()
def reconstruction_mae(world, data, config, device):
    """Measure current-state reconstruction MAE in physical units.

    Args:
        world: World model to evaluate.
        data: Observation histories from a dataset or policy rollout.
        config: Experiment configuration.
        device: Model device.

    Returns:
        One mean absolute error per state variable.
    """
    totals = torch.zeros(3, device=device)
    count = len(data["vector"])
    for indices in torch.arange(count).split(config["pretrain_eval"]["batch_size"]):
        batch = to_device(data, indices, device, world)
        latent = world.encode(batch["vector"], batch.get("image"))
        prediction = world.decode(latent)["vector"][:, -1]
        totals += (prediction - batch["vector"][:, -1]).abs().sum(0)
    return dict(zip(TENSORBOARD_STATE_NAMES, (totals / count).tolist()))
