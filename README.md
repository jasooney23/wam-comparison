# WAM Comparison

A machine learning and scientific computing workspace managed with [uv](https://docs.astral.sh/uv/).

## Setup

Install uv, then create the project environment and install the locked dependencies:

```powershell
uv sync
```

The project is pinned to Python 3.14 in `.python-version` for TorchRL wheel support. Run Python commands through uv so they use the project environment:

```powershell
uv run python
```

## Included packages

The environment includes PyTorch, TorchRL, TensorDict, TorchVision, TorchAudio, NumPy, Pandas, SciPy, scikit-learn, Matplotlib, Seaborn, Plotly, tqdm, and ipykernel.

Add further packages with `uv add <package>`; commit both `pyproject.toml` and `uv.lock` to keep environments reproducible.

On Windows, the project uses the CUDA 13.2 PyTorch wheel index. Run `uv sync`
after pulling dependency changes. Other platforms keep their default sources.

See the [visual augmentation experiment](src/experiments/visual-augmentation-simple/README.md)
for configuration, CUDA smoke tests, training modes, compute accounting, and plots.
