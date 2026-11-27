"""Print the architecture and parameter count of each world model."""

from pathlib import Path

from experiment_config import build_world_model, load_config

if __name__ == "__main__":
    config = load_config(Path(__file__).with_name("config.toml"), "pretrain")
    for modality, label in (("vector", "Vector WM"), ("combined", "Vector+image WM")):
        model = build_world_model(modality, config["environment"])
        print(f"\n{label}\n{model}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
