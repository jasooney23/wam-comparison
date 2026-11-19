# Visual augmentation on CartPole

Compare vector-only and vector-plus-image world models over paired seeds.
The world model is pretrained on fixed synthetic pairs, then frozen while PPO
trains the policy and value heads. Dataset generation, WM pretraining, and
agent training are separate commands.

## Run

Run from the repository root:

```powershell
uv sync
uv run python src/experiments/visual-augmentation-simple/generate_dataset.py
uv run python src/experiments/visual-augmentation-simple/pretrain_world_model.py
uv run python src/experiments/visual-augmentation-simple/train_agent.py
```

Both training scripts always run vector and combined through the same training
pathway for each seed. Set WM seeds in `pretrain.seeds` and agent seeds in
`train.seeds`. The lists may differ, but agent seed N requires both completed
WMs for seed N. Before training any agent, the agent script checks every
requested pair for matching seed, modality, version, training environment,
completed epoch, and valid model weights. A missing or incompatible save raises
an error; intermediate checkpoints are never used as a fallback.

The default stage outputs are `run.output/world-models` and `run.output/agents`.
Both training scripts accept `--config`, `--smoke`, and `--output`. `--output`
changes only the invoked stage's destination. Agent training accepts
`--world-models` to select a different WM source directory and `--gui` to display
the training environment. For example, reuse a completed pretraining run:

```powershell
uv run python src/experiments/visual-augmentation-simple/train_agent.py --world-models outputs/visual-augmentation-simple/world-models --output outputs/new-agents
```

Agent training needs no dataset. WMs remain frozen, including their saved
normalization statistics. Agent snapshots record the source WM path and seed.
There is no resume support. Existing run directories are never overwritten;
choose a new stage output directory to rerun training. Checkpoints must match
the experiment version.

The generator also refuses to overwrite its dataset. Choose a new
`pretrain.dataset` path when regenerating; its `--output` overrides the dataset
destination. Existing datasets remain usable when their image size, ID/OOD
environment settings, and pair counts match the experiment. Datasets and
checkpoints that predate the observation settings use the default observation
layout.

For a CPU smoke check, set `run.device = "cpu"`, use a separate `run.output`
and `pretrain.dataset`, and pass `--smoke` to all three commands. This uses seed
0 for each stage, 32 ID pairs, 500 OOD pairs, one pretraining epoch, and two agent
episodes per modality. These checks verify behavior, not training quality.

## Configuration

- `run`: shared output root, device, and CPU threads.
- `environment`: pole mass, episode limit, and CartPole observation settings.
- `train`: agent seeds, episode count, policy learning rate, and shared ID ranges.
- `environment.ood`: shared OOD ranges for both evaluation phases.
- `pretrain`: WM seeds, dataset path, ID pair count, epochs, optimizer settings,
  gradient clipping, batch size, loss weights, and `eval_every` (default 5 epochs).
- `pretrain_eval`: OOD pair count (default 500) and MAE evaluation batch size.
- `train_eval`: joint evaluation interval in agent episodes, evaluation episode
  count, and WM prediction horizon.
- `ppo`: rollout size, minibatches, epochs, discount, GAE, and loss settings.

`environment.history_frames` sets the number of frames in each vector and
image history. It defaults to 4 and must be a positive integer. The dataset
generator requires `max_steps` to be at least `history_frames`.
The boolean flags `input_cart_pos`, `input_cart_vel`, `input_pole_angle`, and
`input_pole_vel` select vector elements in that order. Enable at least one.
By default, the first three are enabled. Changing the frame count or enabled
elements requires a matching dataset and world-model checkpoint.

ID physics use the cart mass and hinge damping ranges under `train`.
Each OOD parameter is sampled uniformly from its `environment.ood` range after
removing its training interval, including boundaries. Wider remaining intervals
receive proportionally more samples. For example, `[0.25, 1.5]` minus `[0.5, 1]`
leaves values below `0.5` or above `1`. An empty remainder for either parameter
raises an error. Both evaluation phases use these exclusions; regenerate existing
datasets when their saved evaluation settings do not include the exclusions. Parameters
stay constant during a history or episode. There are no fixed holdout settings.

## Data and evaluation

The generator saves 10,000 ID pairs and 500 OOD pairs in one file. It samples
cart position and pole angle within termination limits, and cart velocity and
pole angular velocity in `[-2, 2]`. It simulates `history_frames - 1` random steps
to fill the history, rejecting episodes that end early, then saves a uniformly
sampled action and the next frame. Both splits use this process with independent
seeds and their respective physics ranges. They need not resemble policy data.
The default 32 by 32 images require about 54 MB across the two splits.
Regenerate 64 by 64 datasets and retrain their combined world models for this resolution.

Pretraining losses are sample-weighted epoch averages. Every `pretrain.eval_every`
epochs and at the final epoch, evaluate MAE in physical units and save a checkpoint.
Horizon 0 compares the reconstructed latest frame with the latest observed frame
on the training and OOD splits. Horizon 1 advances the OOD latent with the stored
action once and compares the decoded latest frame with `next_vector`.

During agent training, periodic joint evaluation generates OOD policy episodes.
Agent return and WM prediction errors use these same episodes, not separate
random-action windows. Physics and reset seeds are paired across modalities and
evaluation intervals; policy actions are deterministic. At each observation,
encode the current history and compare its reconstruction at horizon zero.
Advance prior latents with the actions actually selected by the policy and
compare decoded predictions at each future horizon. Windows stop at episode
boundaries, including the final observation as a future target. Each horizon
averages all available windows and records its count. An unavailable horizon
has NaN MAE and zero windows. `horizon = 0` makes no dynamics calls.

Separately, `train/wm/` reconstruction MAE measures what the training agent sees,
once per collected PPO rollout update. Evaluation never changes model weights,
normalization statistics, or optimizer counters.

## Metrics and artifacts

Each `run.output/STAGE/seed-N/MODALITY/` directory contains `config.json`,
`metrics.csv`, and `tensorboard/`, where STAGE is `world-models` or `agents`.
The CSV columns are `metric,step,value`. Every scalar goes to both destinations
with the same tag and step. Phase prefixes identify the step axis:

| Metric prefix | Step axis | Destination |
| --- | --- | --- |
| `pretrain/wm/` | Pretraining epoch | CSV and TensorBoard |
| `train/ppo/episode_return` | Completed episode | CSV and TensorBoard |
| `train/wm/` | Cumulative PPO optimizer updates | CSV and TensorBoard |
| `eval/` | Completed training episodes | CSV and TensorBoard |
| `compute/` | Completed training episodes | CSV and TensorBoard |

Pretraining tags are `loss_total`, `loss_rec`, `loss_dyn`, `wall_time`,
`grad_steps`, and `mae_{variable}_train`, `mae_{variable}_0_test`, and
`mae_{variable}_1_test` for each selected state variable.
Losses and compute are logged each epoch; MAE follows `pretrain.eval_every`.
Agent-observation tags are `train/wm/mae_{cartPos/cartVel/poleAngle/poleVel}`
for the selected state elements.
Every TensorBoard value is also written to the CSV with the same tag and step.
Joint evaluation records return mean/spread, per-variable `mae_*_hN_ood`, and
`windows_hN`. Compute records world/agent training seconds, optimizer updates,
agent environment steps, and total training seconds.

Training timers exclude dataset generation, evaluation, logging, and checkpoint
writing. CUDA work is synchronized before stopping timers. Agent time includes
policy rollout collection and PPO updates. Loading a pretrained WM retains its
original training cost. These counters do not imply equal cost per update.

Pretraining writes `world-epoch-NNN.pt` at each evaluation and publishes
`world-final.pt` only after successful completion. These contain weights,
normalization buffers, configuration, environment metadata, and training counters.
Agent training writes `episode-NNNNNN.pt` snapshots at every evaluation, including
episode zero. They contain model weights, configuration, counters, and source WM
provenance, without optimizer or random-generator state. There is no automatic
chart generation or comparison export.

The split preserves training methods and metrics, but resetting each independent
stage's random generators can change trajectories compared with the old combined
script.

```powershell
uv run tensorboard --logdir outputs/visual-augmentation-simple
```

## Models and physics

Each vector frame contains the selected CartPole state elements. Reset repeats
the first frame to fill the history. Combined models also receive a 32 by 32
grayscale image for each frame. Rendering exposes no direct mass or damping cue.

Vector MLP and image CNN features project to a shared latent of size 16.
Action-conditioned latent dynamics predict the next latent. Decoders reconstruct
histories of the configured length. MLPs used by the experiment have one hidden
layer of 32.
Images are scaled to `[0, 1]`; vectors use training-only means and standard
deviations. Reconstruction and dynamics losses each sum weighted modality MSE,
then the two losses are summed. Dynamics targets shift the history and append
the saved next frame. Vector-only models have no image decoder or image loss.

TorchRL supplies categorical policies, ClipPPOLoss, and GAE. PPO trains only
policy/value heads. It bootstraps time limits and stops GAE at episode resets.
Rollouts end at episode and evaluation boundaries and can exceed their target
step count. There is no separate custom PPO loss or value clipping.

The local CartPole preserves upstream licensing and adds cart/pole masses and
viscous hinge torque. See [NOTICE.md](../../common/environments/cartpole/NOTICE.md).
The combined model has an extra image loss; this experiment does not isolate
input information from changes to the training objective. Small smoke runs
verify behavior, not training quality.
