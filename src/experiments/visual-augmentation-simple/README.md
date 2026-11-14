# Visual augmentation on CartPole

Compare vector-only and vector-plus-image world models under a shared training
compute budget. The default experiment has three paired seeds. Each arm trains
until it reaches `training.compute_budget` (100 trillion nominal operations).
`training.episodes = 0` removes the episode cap; a positive value adds a cap.
Comparison plots and CSVs match cumulative compute over the range all arms cover.

PPO uses TorchRL's `ProbabilisticActor`, `ClipPPOLoss`, and `GAE`. Rollouts and
minibatches use TensorDict. A small Gymnasium loop retains exact episode limits
and deterministic episode resets; there is no custom PPO loss or GAE recursion.

## Install and run

Run from the repository root:

```powershell
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"
uv run pytest -q
uv run python src/experiments/visual-augmentation-simple/train.py --smoke --mode frozen --output outputs/smoke-frozen
uv run python src/experiments/visual-augmentation-simple/train.py --smoke --mode alternating --output outputs/smoke-alternating
uv run python src/experiments/visual-augmentation-simple/train.py --smoke --mode joint --output outputs/smoke-joint
```

Windows uses PyTorch's explicit CUDA 13.2 wheel index. Other platforms retain
their default package sources. The config requests CUDA and fails early if it
cannot execute GPU kernels. Set `run.device = "cpu"` explicitly for CPU runs.
The project uses Python 3.14 because TorchRL 0.15 publishes wheels through that
version. `uv sync` installs the pinned interpreter when needed.
Smoke runs use real training, two episodes, and small datasets; they do not
establish model quality or guarantee a shared compute range for comparison.

Edit `config.toml` or pass `--config path/to/config.toml`. Launch a full run by
omitting `--smoke`. Use a new `--output` directory for a new dataset/config.
An existing run is never silently overwritten. Full runs have not been launched
as part of implementation.
Add `--gui` to show the CartPole environment while the agent trains.

Resume one arm with the same command/config, adding its checkpoint:

```powershell
uv run python src/experiments/visual-augmentation-simple/train.py --smoke --mode frozen --output outputs/smoke-frozen --resume outputs/smoke-frozen/seed-0/frozen/vector/episode-000001.pt
```

Resume restores model/optimizer state, normalization, random generators,
episodes, transitions, updates, compute, and training timing. Only the episode
limit may change. Checkpoints are at episode boundaries, so the next reset uses
the same deterministic episode seed. Load only trusted local checkpoints.
Datasets and checkpoints carry an experiment version. Artifacts from the old
prediction-only objective or PPO backend are rejected. Start a new output
directory for this experiment definition.

## Observations and models

Each observation contains four `(position, cart_velocity, pole_angle)` vectors.
Reset repeats the initial observation four times. Visual models also receive
four 64×64 grayscale frames from the unchanged CartPole renderer. Frames use
area resizing; pixels are scaled to `[0, 1]` inside the encoder. Vector means
and standard deviations come only from shared pretraining labels.

Each vector or CNN encoder produces 16 features. Concatenated features project
to 16 latent units. An action-conditioned MLP predicts the next latent. The same
decoders reconstruct the current four-frame observation from its latent and
predict the next four-frame observation from the predicted latent. Vector
outputs have shape `(4, 3)`; image outputs have shape `(4, 64, 64)`.
Every MLP has one hidden layer of 32 ReLU units. The CNN uses three
stride-two 4×4 convolutions with 16, 32, and 32 channels; the image decoder uses
the corresponding transpose-convolution structure. No autonomous rollout loss
or recurrent memory is added.

For each of reconstruction and dynamics, average squared errors across batch,
history, and features/pixels separately for each active modality. Sum vector
and image MSE using `training.vector_weight` and `training.image_weight` (both
default to one). Sum reconstruction and dynamics losses with equal weights.
The next history consists of the last three input frames followed by the new
frame, before any episode reset. The dataset stores only that new target frame
to avoid duplicating histories. Vector targets use training-only normalization;
image targets use pixels scaled to `[0, 1]`.

Vector-only models have no image decoder or loss. Image-only models have no
vector decoder or loss and require no vector labels. Combined models train
both branches end to end. The default comparison uses vector and combined.

All training modes share random-action pretraining data and warm-start training:

| Mode | Updates during control training |
| --- | --- |
| `frozen` | PPO updates policy and value heads; the world model stays fixed. |
| `alternating` | Update the world model from the latest rollout, then PPO heads with its encoder fixed. |
| `joint` | Update the WAM alongside PPO in each minibatch, using separate losses and optimizers. |

The policy consumes the current encoded latent. Every mode detaches WAM features
from PPO gradients. WAM loss updates only the WAM; PPO loss updates only the policy
and value heads. The WAM optimizer continues from pretraining during joint and
alternating updates. `world_updates` sets WAM steps per PPO minibatch in joint
mode and full passes over a rollout in alternating mode. PPO bootstraps time
limits, but stops GAE at every reset. Rollouts finish at episode and evaluation
boundaries; they can be shorter or longer than the configured step target.

TorchRL owns categorical action sampling, clipped policy loss, squared-error
value loss, optional entropy loss, and GAE normalization over each rollout.
The critic reuses the actor's fresh, detached latent features in each minibatch,
so the encoder runs once for PPO. GAE reads
separate `done` and `terminated` flags: a time limit bootstraps the final value,
while a true terminal state does not. Behavior log-probabilities and value targets
are fixed before PPO updates. There is no additional value clipping or minibatch
advantage normalization.

## Physics and evaluation

The local copy in `common/environments/cartpole/` preserves upstream licensing.
It adds validated cart/pole masses and viscous hinge torque
`-hinge_damping * angular_velocity` to the coupled equations. It never edits the
installed Gymnasium source. See its `NOTICE.md` for the equations.

Training samples cart mass from `[1.0, 1.01]` kg and hinge damping from
`[0, 0.0001]` N·m·s. Pole mass stays at 0.1 kg. Parameters stay constant within
each episode. Defaults preserve Gymnasium's length, gravity, Euler integration,
reward, failure thresholds, and 500-step limit. Rendering exposes no direct
mass/damping cue. The full four-state simulator observation is never fed to a
model.

`evaluation.settings` lists held-out `(cart_mass, hinge_damping)` pairs.
Validation rejects any pair inside the joint training ranges, including their
boundaries. A setting is OOD if mass, damping, or both fall outside those ranges.
Evaluation covers every configured setting with the same episode count, using
deterministic actions and identical reset seeds across modalities and evaluation
intervals. Fixed independent random-action datasets use paired collection seeds.
Defaults use 10 control episodes and 256 prediction transitions per setting.

Metrics include return, reconstruction loss, and dynamics loss for each setting
(`heldout_N`) and their uniform mean (`ood`). Each loss also has separate vector
and image metrics when those modalities are active. Evaluation cannot update
parameters, normalization, or training counters. No multi-step rollout evaluation
is included.

Images depict position and angle already present in vector history. Temporal
vectors also expose angular velocity. A visual advantage is therefore not
guaranteed. The extra image reconstruction and prediction losses change the objective; this is
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
- `curves.png`: all-setting scores, OOD WAM losses, and both compute totals against episodes.
- `tensorboard/`: scalar event logs, including pretraining losses.
- `comparison-MODE.png`: mean OOD return and WAM losses at common compute, with seed means ± one standard deviation.
- `comparison-MODE.csv`: the matched values, operation counts, seed counts, and interpolation method.

```powershell
uv run tensorboard --logdir outputs/visual-augmentation-simple
```

Each seed's evaluations are linearly interpolated onto common operation counts
before calculating means and seed spread. The range starts at the largest
pretraining cost and ends at the smallest final cost or budget across all runs.
There is no extrapolation. These are interpolated estimates, not fresh evaluations
of models trained to exactly identical operation counts. If smoke runs have no
shared range, the comparison CSV is empty and the plot explains why.

Budget checks occur at episode and optimizer-minibatch boundaries. The final
operation count can exceed the budget by one such unit. `metrics.csv` records
`budget_reached` and `budget_overshoot`; comparisons stop at the budget. A positive
episode cap can stop an arm earlier. A budget that cannot cover pretraining fails
with an explicit error. Evaluation is recorded at the configured episode interval
and at the final training boundary. No full-budget runs have been validated.

Raw evaluations and within-setting episode standard deviations are also saved.
Checkpoints, logs, plots, and datasets are ignored by version control.
