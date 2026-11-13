# Visual augmentation on CartPole

Compare vector-only and vector-plus-image world models under recorded training
compute. The default experiment has three paired seeds and 1,000 PPO episodes
per arm. It does not stop at equal FLOPs or automatically match curves.

PPO uses TorchRL's `ProbabilisticActor`, `ClipPPOLoss`, and `GAE`. Rollouts and
minibatches use TensorDict. A small Gymnasium loop retains exact episode limits
and deterministic episode resets; there is no custom PPO loss or GAE recursion.

## Install and run

Run from the repository root:

```powershell
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"
uv run pytest -q
uv run python src/experiments/visual-augmentation-simple/train.py --smoke --output outputs/smoke-frozen
uv run python src/experiments/visual-augmentation-simple/train.py --smoke --mode alternating --output outputs/smoke-alternating
uv run python src/experiments/visual-augmentation-simple/train.py --smoke --mode joint --output outputs/smoke-joint
```

Windows uses PyTorch's explicit CUDA 13.2 wheel index. Other platforms retain
their default package sources. The config requests CUDA and fails early if it
cannot execute GPU kernels. Set `run.device = "cpu"` explicitly for CPU runs.
The project uses Python 3.14 because TorchRL 0.15 publishes wheels through that
version. `uv sync` installs the pinned interpreter when needed.
Smoke runs use real training, two episodes, and small datasets; they do not
establish model quality.

Edit `config.toml` or pass `--config path/to/config.toml`. Launch a full run by
omitting `--smoke`. Use a new `--output` directory for a new dataset/config.
An existing run is never silently overwritten. Full runs have not been launched
as part of implementation.
Add `--gui` to show the CartPole environment while the agent trains.

Resume one arm with the same command/config, adding its checkpoint:

```powershell
uv run python src/experiments/visual-augmentation-simple/train.py --smoke --output outputs/smoke-frozen --resume outputs/smoke-frozen/seed-0/frozen/vector/episode-000001.pt
```

Resume restores model/optimizer state, normalization, random generators,
episodes, transitions, updates, compute, and training timing. Only the episode
limit may change. Checkpoints are at episode boundaries, so the next reset uses
the same deterministic episode seed. Load only trusted local checkpoints.
Checkpoints from the previous custom PPO backend are rejected. Start a new
output directory for TorchRL runs so the two implementations' curves stay separate.

## Observations and models

Each observation contains four `(position, cart_velocity, pole_angle)` vectors.
Reset repeats the initial observation four times. Visual models also receive
four 64×64 grayscale frames from the unchanged CartPole renderer. Frames use
area resizing; pixels are scaled to `[0, 1]` inside the encoder. Vector means
and standard deviations come only from shared pretraining labels.

Each vector or CNN encoder produces 32 features. Concatenated features project
to 16 latent units. An action-conditioned MLP predicts the next latent; decoders
predict the next single normalized vector and, for visual models, the next
image. Every MLP has one hidden layer of 32 ReLU units. The CNN uses three
stride-two 4×4 convolutions with 16, 32, and 32 channels; the image decoder uses
the corresponding transpose-convolution structure. No autonomous rollout loss
or recurrent memory is added.

The prediction loss is vector MSE plus image MSE, weighted by
`training.vector_weight` and `training.image_weight`. Image loss is absent in
the vector arm. `image` modality is supported, but still uses supervised vector
labels for prediction and comparison; it is not the default comparison arm.

All training modes share random-action pretraining data and warm-start training:

| Mode | Updates during control training |
| --- | --- |
| `frozen` | PPO updates policy and value heads; the world model stays fixed. |
| `alternating` | Update the world model from the latest rollout, then PPO heads with its encoder fixed. |
| `joint` | PPO and weighted prediction loss train the encoder together; prediction also trains dynamics/decoders. |

The policy consumes the current encoded latent, rather than planning through
the dynamics model. Joint mode has one PPO optimizer with separate learning-rate
groups for world-model and policy/value parameters. The pretraining optimizer
is not stepped during joint PPO. Alternating mode owns world-model and policy
updates in separate optimizers. PPO bootstraps time limits, but stops GAE at
every reset. Rollouts finish at episode boundaries and evaluation boundaries;
they can therefore be shorter or longer than the configured step target.

TorchRL owns categorical action sampling, clipped policy loss, squared-error
value loss, optional entropy loss, and GAE normalization over each rollout.
The critic reuses the actor's fresh latent features in each minibatch, so the
encoder runs once for PPO and both policy/value gradients reach it in joint mode.
Alternating and frozen modes detach those features from PPO gradients. GAE reads
separate `done` and `terminated` flags: a time limit bootstraps the final value,
while a true terminal state does not. Behavior log-probabilities and value targets
are fixed before PPO updates. There is no additional value clipping or minibatch
advantage normalization.

## Physics and evaluation

The local copy in `common/environments/cartpole/` preserves upstream licensing.
It adds validated cart/pole masses and viscous hinge torque
`-hinge_damping * angular_velocity` to the coupled equations. It never edits the
installed Gymnasium source. See its `NOTICE.md` for the equations.

Training samples cart mass from `[0.5, 1.5]` kg and hinge damping from
`[0, 0.005]` N·m·s. Pole mass stays at 0.1 kg. Parameters stay constant within
each episode. Defaults preserve Gymnasium's length, gravity, Euler integration,
reward, failure thresholds, and 500-step limit. Rendering exposes no direct
mass/damping cue. The full four-state simulator observation is never fed to a
model.

`evaluation.settings` lists `(cart_mass, hinge_damping)` pairs. The first is the
familiar setting `(1, 0.0025)`. The five others vary mass, damping, or both outside
the training ranges. Evaluation uses deterministic actions, paired reset seeds,
10 episodes per setting, and fixed independent random-action prediction data.
Measure normalized one-step vector MSE for both arms; image MSE is supplementary.
Evaluation cannot update parameters or training counters. No multi-step rollout
evaluation is included.

Images depict position and angle already present in vector history. Temporal
vectors also expose angular velocity. A visual advantage is therefore not
guaranteed. The extra image-prediction loss also changes the objective; this is
not a controlled test of input information alone. Episode-return saturation at
500 can hide differences. Check prediction error and held-out settings too.

## Compute conventions

Counters describe nominal arithmetic, not measured hardware instructions or
energy. They depend on executed batch shapes, not just parameter counts or
episode count. CUDA kernel fusion, padding optimizations, and precision can
change actual hardware work. Model and optimizer arithmetic estimates use these
fixed conventions in every arm:

| Operation | Count |
| --- | --- |
| Linear | Two operations per multiply-add, plus output bias additions. |
| Convolution | Two operations per nominal kernel multiply-add, including padding. |
| Transpose convolution | Input elements × output channels/group × kernel area × two, plus bias. |
| Layer backward | Weight-gradient work and input-gradient work when needed; bias reduction. |
| ReLU | One scalar operation forward and backward per element. |
| Sigmoid | Four scalar operations forward, three backward per element. |
| Vector normalization / MSE | Shape-based scalar-operation allowance in the training loop. |
| Policy distribution / PPO loss / GAE | Fixed per-sample scalar allowances documented in code. |
| Adam | Fifteen scalar operations per parameter with a gradient, in the broader scope. |
| Physics | Forty scalar operations per transition, in the broader scope. |
| Image preprocessing | `8 × source_height × source_width + 4096` operations per frame, in the broader scope. |

Thus the neural total is a consistent arithmetic estimate, including approximated
loss/distribution work; it is not an exhaustive profiler measurement. Tensor
reshaping, indexing, concatenation, storage, random generation, comparisons,
gradient-norm clipping, rendering, and general Python execution have no complete
FLOP accounting here. Activation estimates include comparisons as scalar work.
Do not interpret these counters as exact total-device FLOPs.

Both scopes are selectable through `accounting.scope`:

- `neural`: network arithmetic for pretraining, PPO, and training action/value inference.
- `broader_estimate`: neural totals plus estimated preprocessing, physics, and Adam work.

Both totals are always saved. Evaluation and held-out dataset creation are
excluded. Shared training-data collection costs are logged separately and charged
once to each arm's broader total, representing an independently run arm. Each
arm's pretraining is included at episode zero. Reused datasets retain their
original collection metadata. Additional vector normalization, loss, or image
scaling allowances are recorded at their training call sites.

Training elapsed time excludes evaluation and shared collection, which has its
own metadata. Training totals synchronize CUDA at phase/update boundaries.
Environment, rendering, preprocessing, and batch-transfer timings are recorded
separately. Batch-transfer timing includes host enqueue work and is not a GPU
kernel timer; timing categories are diagnostic and can overlap with training.

## Artifacts

Under `run.output`:

- `seed-N/data.pt`: shared training and held-out data, settings, and collection costs.
- `seed-N/MODE/MODALITY/config.json`, `collection.json`: resolved settings and collection metadata.
- `metrics.csv`: episode counts, scores, errors, compute, transitions, updates, and timing.
- `episode-NNNNNN.pt`: resumable model and optimizer checkpoints, including counters by phase.
- `curves.png`: all-setting scores, vector errors, and both compute totals against episodes.
- `tensorboard/`: scalar event logs, including pretraining losses.
- `comparison-MODE.png`: familiar-setting comparison across arms, showing seed means ± one standard deviation and selected compute totals.

```powershell
uv run tensorboard --logdir outputs/visual-augmentation-simple
```

For equal-compute analysis, inspect scores alongside cumulative compute over the
range both arms cover. No interpolation, extrapolation, or automated matching
is performed. Raw evaluation means and within-setting episode standard deviations
are saved. Checkpoints, logs, plots, and datasets are ignored by version control.
