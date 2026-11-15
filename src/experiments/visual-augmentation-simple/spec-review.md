# Implementation review against init-plan.md

Reviewed on 2026-11-15. `init-plan.md` is the specification for this review.
The user approved the correction plan, excluding image-only support. The
approved corrections are implemented.

## Matches

- Gymnasium environment, standard CartPole actions and rewards, variable cart
  mass, fixed pole mass, and viscous hinge damping.
- Four-frame histories of cart position, cart velocity, and pole angle. Angular
  velocity and physics parameters are absent from model inputs. Rendering has
  no direct mass or damping cues.
- Vector MLP and image CNN encoders with fixed 16-feature outputs, concatenation
  and projection to a 16-unit latent, action-conditioned latent dynamics, and
  vector MLP / image transpose-CNN decoders.
- Vector-only and vector-plus-image input, matching the specification as edited
  during this review. Image-only support is no longer required.
- Every MLP instantiated by the experiment has one 32-unit hidden layer.
- End-to-end world-model training with reconstruction and dynamics losses.
  Each modality's squared errors are averaged separately; the default weights
  are equal.
- A fixed, seeded dataset made from sampled simulator states and random actions.
  Pretraining checkpoints are saved every 20 epochs and at the final epoch.
- PPO policy and value training with the world model frozen, including correct
  handling of time limits and episode boundaries.
- OOD-only periodic agent evaluation with paired seeds, episodic returns,
  per-variable errors, and open-loop latent prediction through horizon four.
- Separate training time and optimizer-update counters for the world model and
  agent, agent environment-step counts, and TensorBoard logging.

## Corrections

1. **Horizon zero now measures reconstruction.** It uses the same calculation
   as step zero of a positive-horizon evaluation. Regression tests reject any
   call to dynamics in this case.
2. **Checkpoint-only runs no longer require the pretraining dataset.** Both
   `--world-checkpoint` and `--resume` load only held-out data and use saved
   normalization buffers. Fresh pretraining still validates the dataset and
   its cache signature. Version 8 world-model checkpoints remain compatible;
   older agent checkpoints and evaluation caches require a new run because
   evaluation semantics and recorded metrics changed.
3. **Compute comparisons now include total training time.** The sum of world
   and agent training seconds is recorded and used for `comparison-time.csv`
   and `comparison-time.png`, alongside the existing step comparisons. Original
   pretraining costs are retained for loaded world models. World models are
   evaluated at every pretraining checkpoint, with results in `world-metrics.csv`
   and TensorBoard. Comparisons use shared ranges without extrapolation. They
   remain interpolated estimates, not runs stopped at identical wall-clock
   budgets; short runs may have no shared time range.
4. **The reusable MLP defaults to a 32-unit hidden layer.** An explicit empty
   list still creates a linear model. The experiment already used explicit
   32-unit layers, so its architecture is unchanged.
5. **Documentation reflects the specification and config.** The README now
   describes 10,000 samples and about 205 MB of image tensors, reconstruction
   at horizon zero, checkpoint-only runs, and the new metrics. Earlier GRU
   research recommendations are marked as historical advice.

The current metric averages the scalar L2 norm for each state variable,
equivalent to mean absolute error. The plan does not specify how to aggregate
L2 errors across samples. Keep this documented convention unless a different
definition, such as RMSE, is requested.

## Verification

Before changes, `.venv/Scripts/python.exe -m pytest -q` passed all 31 tests.
After the corrections, all 36 tests passed. There were two existing Gymnasium
warnings about unbounded observation-space limits. Regression coverage includes
checkpoint-only training, resume without training data, periodic world-model
evaluation, reconstruction-only evaluation, MLP defaults, and comparison ranges.

The two checkpoint-only tests also passed after adding version 8 world-model
compatibility. Ruff lint and formatting checks passed for all changed Python
files, and `git diff --check` found no whitespace errors.

CPU command-line smoke checks passed for synthetic dataset generation, both
modalities, a new agent using a saved combined world model, and resuming that
agent. Both checkpoint paths used a nonexistent dataset path. The checks also
verified world-model evaluation CSVs, total training-time accounting, and both
comparison artifact pairs. Artifacts are under `outputs/spec-review-dypj8ld1/`.
These small runs verify behavior, not training quality or a visual advantage.
