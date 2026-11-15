# Visual augmentation on CartPole

Compare vector-only and vector-plus-image world models over paired seeds.
The world model is pretrained on fixed synthetic pairs, then frozen while PPO
trains the policy and value heads. The specification is [init-plan.md](init-plan.md).

## Run

Run from the repository root:

```powershell
uv sync
uv run python src/experiments/visual-augmentation-simple/generate_dataset.py
uv run python src/experiments/visual-augmentation-simple/train.py
```

The generator refuses to overwrite an existing dataset. Choose a new
`pretrain.dataset` path when regenerating. Version 10 requires a dataset with
ID and OOD range metadata; older datasets and checkpoints require a new run.

For a CPU smoke check, set `run.device = "cpu"` and choose a separate dataset
path. Pass `--smoke` to both commands. This uses 32 ID pairs, 500 OOD pairs,
one pretraining epoch, and two agent episodes. `--output` changes the generator's
dataset destination or the trainer's run output directory respectively.

For agent-only training, configure one seed and modality, then use:

```powershell
uv run python src/experiments/visual-augmentation-simple/train.py --output outputs/new-agent --world-checkpoint outputs/visual-augmentation-simple/seed-0/vector/world-epoch-100.pt
```

Agent-only and resumed runs do not need the dataset. Use `--resume` with a
trusted episode checkpoint to continue its run. Only `train.episodes` may change.
Resume restores the model, optimizers, random states, counters, and CSV position;
TensorBoard events are rebuilt from the durable CSV rows. Add `--gui` to display
the agent's training environment.

## Configuration

- `run`: output directory, device, seeds, modalities, and CPU threads.
- `environment`: pole mass and episode limit.
- `train`: agent episode count, policy learning rate, and shared ID ranges.
- `environment.ood`: shared OOD ranges for both evaluation phases.
- `pretrain`: dataset path, ID pair count, epochs, optimizer settings, batch size,
  loss weights, and `eval_every` (default 5 epochs).
- `pretrain_eval`: OOD pair count (default 500) and reconstruction evaluation batch size.
- `train_eval`: joint evaluation interval in agent episodes, evaluation episode
  count, and WM prediction horizon.
- `ppo`: rollout size, minibatches, epochs, discount, GAE, and loss settings.

ID physics use cart mass `[1.0, 1.01]` and hinge damping `[0, 0.0001]`.
OOD physics use mass `[0.25, 2.0]` and damping `[0.0025, 0.01]`.
At least one dimension must be disjoint from its ID range, including boundaries,
so every sampled OOD combination is outside the training distribution.
Both phases sample uniformly from the same corresponding ranges. Parameters
stay constant during a history or episode. There are no fixed holdout settings.

## Data and evaluation

The generator saves 10,000 ID pairs and 500 OOD pairs in one file. It samples
cart position and pole angle within termination limits, and cart velocity and
pole angular velocity in `[-2, 2]`. It simulates three random steps to make a
four-frame history, rejecting histories that end early, then saves a uniformly
sampled action and the next frame. Both splits use this process with independent
seeds and their respective physics ranges. They need not resemble policy data.
The default images require about 215 MB across the two splits.

Pretraining losses are sample-weighted epoch averages. Every `pretrain.eval_every`
epochs and at the final epoch, evaluate current-state reconstruction MAE over
all ID training pairs and all saved OOD pairs, then save a world-model checkpoint.
MAE uses the newest frame and original physical units. OOD pairs never train the
model or set normalization statistics.

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

Each `run.output/seed-N/MODALITY/` directory contains one `metrics.csv` with
columns `metric,step,value`. Phase prefixes identify the step axis:

| Metric prefix | Step axis | Destination |
| --- | --- | --- |
| `pretrain/wm/` | Pretraining epoch | CSV and TensorBoard |
| `train/ppo/episode_return` | Completed episode | CSV and TensorBoard |
| `train/wm/` | Cumulative PPO optimizer updates | CSV and TensorBoard |
| `eval/` | Completed training episodes | CSV |
| `compute/` | Completed training episodes | CSV |

Pretraining tags are `loss_total`, `loss_rec`, `loss_dyn`, `wall_time`,
`grad_steps`, and `mae_{cartPos/cartVel/poleAngle}_{train/test}`.
Losses and compute are logged each epoch; MAE follows `pretrain.eval_every`.
Agent-observation tags are `train/wm/mae_{cartPos/cartVel/poleAngle}`.
Every TensorBoard value is also written to the CSV with the same tag and step.
Joint evaluation records return mean/spread, per-variable `mae_*_hN_ood`, and
`windows_hN`. Compute records world/agent training seconds, optimizer updates,
agent environment steps, and total training seconds.

Training timers exclude dataset generation, evaluation, logging, and checkpoint
writing. CUDA work is synchronized before stopping timers. Agent time includes
policy rollout collection and PPO updates. Loading a pretrained WM retains its
original training cost. These counters do not imply equal cost per update.

Other artifacts are `config.json`, `tensorboard/`, pretrained
`world-epoch-NNN.pt`, and resumable `episode-NNNNNN.pt` checkpoints.
There is no automatic chart generation or comparison export.

```powershell
uv run tensorboard --logdir outputs/visual-augmentation-simple
```

## Models and physics

Each input contains four `(cart_position, cart_velocity, pole_angle)` vectors;
reset repeats the first frame. Combined models also receive four 64 by 64
grayscale images. The full four-state simulator observation and physics
parameters are never fed to the model. Rendering exposes no direct mass or
damping cue. Angular velocity can still be inferred from temporal observations,
so an advantage from images is not guaranteed.

Vector MLP and image CNN features project to a shared latent of size 16.
Action-conditioned latent dynamics predict the next latent. Decoders reconstruct
four-frame histories. MLPs used by the experiment have one hidden layer of 32.
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
