"""Pretrain vector and vector-plus-image world models on a fixed dataset."""

import argparse
from pathlib import Path
from time import perf_counter

import torch
from experiment_config import environment_settings, load_config, matches_environment
from training_utils import (
    EXPERIMENT_VERSION,
    MODALITIES,
    MetricWriter,
    check_output,
    checked_step,
    prepare_run,
    reconstruction_mae,
    run_directory,
    save_snapshot,
    seed_run,
    setup_device,
    to_device,
)

from src.common.environments.cartpole.observations import IMAGE_SIZE
from src.common.models.world_model import WorldModel


def load_dataset(config):
    """Load ID/OOD pairs and check settings, sample counts, and image size."""
    dataset = torch.load(
        config["pretrain"]["dataset"], weights_only=False, map_location="cpu"
    )
    if not matches_environment(
        dataset.get("environment"), environment_settings(config)
    ) or not matches_environment(
        dataset.get("evaluation_environment"), environment_settings(config, ood=True)
    ):
        raise ValueError("Dataset environment changed; regenerate the ID/OOD dataset")
    for key, section in (("training", "pretrain"), ("evaluation", "pretrain_eval")):
        if (
            key not in dataset
            or len(dataset[key]["action"]) != config[section]["transitions"]
        ):
            raise ValueError(f"Dataset size does not match {section}.transitions")
        for name in ("image", "next_image"):
            images = dataset[key].get(name)
            if images is None or images.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError(
                    f"Regenerate the dataset with {IMAGE_SIZE}x{IMAGE_SIZE} images"
                )
    return {
        "training": dataset["training"],
        "pretrain_evaluation": dataset["evaluation"],
    }


def run_pretraining(config, seed, modality, payload, directory):
    """Train and evaluate one world model, then publish its final weights.

    Args:
        config: Experiment configuration.
        seed: World-model training seed.
        modality: Vector or combined observations.
        payload: Shared training and OOD evaluation pairs.
        directory: New output directory for this run.
    """
    prepare_run(directory, config)
    seed_run(seed)
    device = torch.device(config["run"]["device"])
    world = WorldModel(modality, config["environment"]).to(device)
    settings = config["pretrain"]
    data = payload["training"]
    optimizer = torch.optim.Adam(world.parameters(), lr=settings["world_lr"])
    updates, training_seconds = 0, 0.0
    writer = MetricWriter(directory)
    try:
        # Train each epoch on the same fixed dataset.
        for epoch in range(1, settings["epochs"] + 1):
            start = perf_counter()
            order = torch.randperm(len(data["action"]))
            totals = {"loss_total": 0.0, "loss_rec": 0.0, "loss_dyn": 0.0}
            for indices in order.split(settings["batch_size"]):
                batch = to_device(data, indices, device, world)
                loss, losses = world.loss(
                    batch, settings["vector_weight"], settings["image_weight"]
                )
                checked_step(
                    loss, optimizer, world.parameters(), settings["max_grad_norm"]
                )
                updates += 1
                for key, value in (
                    ("loss_total", loss),
                    ("loss_rec", losses["reconstruction_loss"]),
                    ("loss_dyn", losses["dynamics_loss"]),
                ):
                    totals[key] += value.item() * len(indices)
            if device.type == "cuda":
                torch.cuda.synchronize()
            training_seconds += perf_counter() - start
            metrics = {key: value / len(order) for key, value in totals.items()}
            metrics.update(wall_time=training_seconds, grad_steps=updates)

            # Evaluate and snapshot at the configured cadence and final epoch.
            evaluating = (
                epoch % settings["eval_every"] == 0 or epoch == settings["epochs"]
            )
            if evaluating:
                for label, observations in (
                    ("train", data),
                    ("test", payload["pretrain_evaluation"]),
                ):
                    metrics.update(
                        {
                            f"mae_{name}_{label}": value
                            for name, value in reconstruction_mae(
                                world, observations, config, device
                            ).items()
                        }
                    )
            for key, value in metrics.items():
                writer.add_scalar(f"pretrain/wm/{key}", value, epoch)
            if evaluating:
                state = {
                    "experiment_version": EXPERIMENT_VERSION,
                    "world": world.state_dict(),
                    "modality": modality,
                    "seed": seed,
                    "epoch": epoch,
                    "world_updates": updates,
                    "world_training_seconds": training_seconds,
                    "environment": environment_settings(config),
                    "evaluation_environment": environment_settings(config, ood=True),
                    "config": config,
                }
                save_snapshot(directory / f"world-epoch-{epoch:03d}.pt", state)
            print(
                f"World model seed={seed} {modality}: epoch {epoch}/{settings['epochs']}, "
                f"loss={metrics['loss_total']:.6f}, updates={updates:,}",
                flush=True,
            )
    finally:
        writer.close()
    save_snapshot(directory / "world-final.pt", state)


def main():
    """Pretrain both modalities for each configured world-model seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.toml")
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output", type=Path, help="Override the world-model output directory"
    )
    args = parser.parse_args()
    config = load_config(args.config, "pretrain", args.smoke)
    setup_device(config)
    output = args.output or Path(config["run"]["output"]) / "world-models"
    check_output(output, config["pretrain"]["seeds"])
    payload = load_dataset(config)
    for seed in config["pretrain"]["seeds"]:
        for modality in MODALITIES:
            run_pretraining(
                config, seed, modality, payload, run_directory(output, seed, modality)
            )


if __name__ == "__main__":
    main()
