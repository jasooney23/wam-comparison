# WAM Comparison

A machine learning and scientific computing workspace managed with [uv](https://docs.astral.sh/uv/).

## Setup

Install uv, then create the project environment and install the locked dependencies:

```powershell
uv sync
```

The project is pinned to Python 3.15 in `.python-version`. Run Python commands through uv so they use the project environment:

```powershell
uv run python
```

## Included packages

The environment includes PyTorch, TorchVision, TorchAudio, NumPy, Pandas, SciPy, scikit-learn, Matplotlib, Seaborn, Plotly, tqdm, and ipykernel.

Add further packages with `uv add <package>`; commit both `pyproject.toml` and `uv.lock` to keep environments reproducible.
