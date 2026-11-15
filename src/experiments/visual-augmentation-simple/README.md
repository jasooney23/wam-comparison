# Visual augmentation on CartPole

Compare vector-only and vector-plus-image world models over three paired seeds.
Each arm trains for `training.episodes` episodes. Comparisons align runs by
cumulative environment steps and total training time over the ranges all arms
cover. The specification is [init-plan.md](init-plan.md).

PPO uses TorchRL's `ProbabilisticActor`, `ClipPPOLoss`, and `GAE`. Rollouts and
minibatches use TensorDict. A small Gymnasium loop retains exact episode limits
and deterministic episode resets; there is no custom PPO loss or GAE recursion.

## Install and run

Run from the repository root:

```powershell
uv sync
uv run python src/experiments/visual-augmentation-simple/generate_dataset.py
uv run python src/experiments/visual-augmentation-simple/train.py
```

The default dataset contains 10,000 samples. Each starts with a uniformly
sampled cart position and pole angle within the termination limits, and cart
velocity and pole angular velocity in `[-2, 2]`. The generator samples physics
parameters from the configured training ranges and simulates three random
steps to produce a four-frame history, rejecting histories that terminate early.
It samples a uniform random action and saves the resulting next frame. This
synthetic distribution need not match the agent's experience. Generation is
seeded and the image tensors require approximately 205 MB of storage.

The world model trains for 100 passes over the fixed dataset. Checkpoints are
saved every 20 epochs and at the final epoch. The agent then trains with that
world model frozen. Each world-model checkpoint also gets an OOD prediction
evaluation, saved in `world-metrics.csv` and TensorBoard. For a new agent using a
saved world model, configure one seed and one modality and run:

```powershell
uv run python src/experiments/visual-augmentation-simple/train.py --output outputs/new-agent --world-checkpoint outputs/visual-augmentation-simple/seed-0/vector/world-epoch-100.pt
```

Agent-only runs and resumed runs use the saved normalization buffers and do not
require the pretraining dataset. Held-out data is still collected or loaded for
evaluation. The saved world-model training time and update count remain part of
the new agent's compute accounting.

Set `run.device = "cpu"` for CPU runs. Configure a different dataset path before
generating a smoke dataset; pass `--smoke` to both scripts. Smoke runs use 32
samples, one pretraining epoch and two agent episodes. Add `--gui` to display
the environment during agent training. Use `--resume` with an episode checkpoint
to resume an arm; only its episode limit may change. Load trusted checkpoints only.
Experiment version 9 rejects older agent checkpoints and held-out caches because
horizon-zero evaluation changed. Start a new output directory for this version.
The fixed pretraining dataset and version 8 world-model checkpoints can be reused;
the world-model architecture and training loss are unchanged.

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

Vector-only models have no image decoder or loss. Combined models train
both branches end to end. The default comparison uses vector and combined.

The policy consumes the encoded latent. The world model is frozen throughout
agent training; PPO updates the policy and value heads. PPO bootstraps time
limits but stops GAE at resets. Rollouts finish at episode and evaluation
boundaries and can differ from the configured step target.

TorchRL owns categorical action sampling, clipped policy loss, squared-error
value loss, optional entropy loss, and GAE normalization over each rollout.
The critic reuses the actor's fresh, detached latent features in each minibatch,
so the encoder runs once for PPO. GAE reads
separate `done` and `terminated` flags: a time limit bootstraps the final value,
while a true terminal state does not. Behavior log-probabilities and value targets
are fixed before PPO updates. There is no additional value clipping or minibatch
advantage normalization.

## Physics and evaluation

The local copy in `src/common/environments/cartpole/` preserves upstream licensing.
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
Defaults use 10 control episodes and 256 prediction windows per setting.
`evaluation.horizon = 4` measures direct reconstruction (step 0) and four
successive latent dynamics steps. The same uniformly sampled action sequence is
used for each modality. Windows stop at episode boundaries. At each step the
model decodes its current latent state without encoding a new observation.
Set `evaluation.horizon = 0` to measure only reconstruction of the current state,
with no dynamics step. This is the same calculation as step zero of a longer
evaluation rollout.

Metrics include episodic return and separate state L2 errors for cart
position, cart velocity, and pole angle for each setting (`heldout_N`) and their
uniform mean (`ood`). Each scalar L2 error is the mean absolute difference in
physical units over held-out windows. Metrics have `_h0` through `_h4` suffixes
at the default horizon. Image output is not evaluated.
Evaluation cannot update parameters, normalization, or training counters.

Images depict position and angle already present in vector history. Temporal
vectors also expose angular velocity. A visual advantage is therefore not
guaranteed. The extra image reconstruction and prediction losses change the objective; this is
not a controlled test of input information alone. Episode-return saturation at
500 can hide differences. Check prediction error and held-out settings too.

## Compute measures

Each run records separate wall-clock training seconds and optimizer updates for
the world model and agent. `total_training_seconds` is their sum, including the
original pretraining cost when loading a world-model checkpoint.
`world_training_seconds` covers only pretraining
minibatch loops. `agent_training_seconds` covers policy rollout collection and
PPO update loops. `agent_environment_steps` counts only policy rollout steps;
the world model has no environment-step measure. Dataset collection, evaluation,
logging, and checkpoint writing are excluded from both timers. CUDA work is synchronized
before each training timer stops.
These measures describe different aspects of compute and do not imply equal cost
per step or update across modalities.

## Artifacts

Under `run.output`:

- `seed-N/data.pt`: held-out data and dataset settings.
- `seed-N/MODALITY/config.json`: resolved settings.
- `metrics.csv`: episode counts, scores, errors, separate training times and updates, and agent environment steps.
- `world-metrics.csv`: OOD world-model errors, training time, and updates at each pretraining checkpoint.
- `world-epoch-NNN.pt`: pretrained world-model checkpoints.
- `episode-NNNNNN.pt`: resumable model and optimizer checkpoints, including update counts.
- `curves.png`: all-setting scores, OOD state errors, and training counts against episodes.
- `tensorboard/`: scalar event logs, including pretraining losses.
- `comparison.png`: mean OOD return and state errors at common environment steps, with seed means ± one standard deviation.
- `comparison.csv`: the matched values, environment steps, seed counts, and interpolation method.
- `comparison-time.png` and `comparison-time.csv`: the same metrics matched by total world-model plus agent training seconds.

```powershell
uv run tensorboard --logdir outputs/visual-augmentation-simple
```

Each seed's evaluations are linearly interpolated onto common agent environment
steps before calculating means and seed spread. The range starts at zero and
ends at the smallest final agent step count across all runs.
There is no extrapolation. These are interpolated estimates, not fresh evaluations
of models trained to exactly identical environment steps. If smoke runs have no
shared range, the comparison CSV is empty and the plot explains why.

Time comparisons use the same method over the shared range of total training
seconds. This range starts at the largest initial pretraining cost and ends at
the smallest final total training cost. It may be empty, especially for smoke
runs. These estimates support comparisons at common training costs; the runs
still stop by episode count, not by an enforced wall-clock budget. Compare
timings collected on the same hardware and device settings.

Evaluation is recorded at the configured episode interval and at the final episode.

Raw evaluations and within-setting episode standard deviations are also saved.
Checkpoints, logs, plots, and datasets are ignored by version control.
