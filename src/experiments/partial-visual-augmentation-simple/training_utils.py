"""Shared logging, model measurements, and run setup for both training stages."""

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter

EXPERIMENT_VERSION = 13
MODALITIES = ("vector", "combined")


class MetricWriter:
    """Write every scalar to CSV and TensorBoard.

    Args:
        directory: Run directory for one seed and modality.
        resume_at: Saved episode and update counts, if resuming an agent run.
    """

    def __init__(self, directory, *, resume_at=None):
        path = directory / "metrics.csv"
        log_directory = directory / "tensorboard"
        if resume_at is not None:
            episodes, updates = resume_at
            temporary = path.with_suffix(".tmp")
            old_events = list(log_directory.glob("events.out.tfevents.*"))

            # Keep only metrics covered by the checkpoint.
            with (
                path.open(newline="") as source,
                temporary.open("w", newline="") as target,
            ):
                reader = csv.reader(source)
                writer = csv.writer(target)
                writer.writerow(next(reader))
                for row in reader:
                    if len(row) != 3:
                        continue
                    metric, step, value = row
                    limit = updates if metric.startswith("train/wm/") else episodes
                    if int(step) <= limit:
                        writer.writerow((metric, step, value))
            temporary.replace(path)

            # TensorBoard uses both episode and update steps in this run.
            self.writer = SummaryWriter(str(log_directory))
            with path.open(newline="") as source:
                for row in csv.DictReader(source):
                    self.writer.add_scalar(
                        row["metric"], float(row["value"]), int(row["step"])
                    )
            self.writer.flush()
            for event in old_events:
                event.unlink()
            self.handle = path.open("a", newline="")
            self.csv = csv.writer(self.handle)
            return

        self.handle = path.open("x", newline="")
        self.csv = csv.writer(self.handle)
        self.csv.writerow(("metric", "step", "value"))
        self.writer = SummaryWriter(str(log_directory))

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
    totals = torch.zeros(len(world.state_names), device=device)
    count = len(data["vector"])
    for indices in torch.arange(count).split(config["pretrain_eval"]["batch_size"]):
        batch = to_device(data, indices, device, world)
        latent = world.encode(batch["vector"], batch.get("image"))
        future = world.predict(latent, batch["action"])
        prediction = world.decode(future)["vector"][:, -1]
        totals += (prediction - batch["next_vector"]).abs().sum(0)
    return dict(zip(world.state_names, (totals / count).tolist()))


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
    totals = torch.zeros(len(world.state_names), device=device)
    count = len(data["vector"])
    for indices in torch.arange(count).split(config["pretrain_eval"]["batch_size"]):
        batch = to_device(data, indices, device, world)
        latent = world.encode(batch["vector"], batch.get("image"))
        prediction = world.decode(latent)["vector"][:, -1]
        totals += (prediction - batch["vector"][:, -1]).abs().sum(0)
    return dict(zip(world.state_names, (totals / count).tolist()))
