# Simple CartPole world model: architecture notes

Research date: 2026-11-12. These are design recommendations, not tested results.

## Recommendation

Start with a small observation encoder, a deterministic GRU memory, and an action-conditioned MLP prediction head. Use the same memory and prediction target for vector-only and vector-plus-image models. This is a simpler first experiment than a stochastic recurrent state-space model (RSSM). The World Models paper supports separating visual encoding, recurrent memory, and control; its actual model uses a VAE and a probabilistic recurrent predictor, so this proposal is a deliberate simplification. [World Models](https://worldmodels.github.io/)

Suggested starting sizes, chosen for simplicity rather than taken from a paper:

- Vector encoder: MLP, 3 inputs, 32 hidden units, 32 outputs.
- Image encoder: small CNN taking a fixed grayscale frame, producing 32 features. Check that resizing preserves small pole-angle changes.
- Combined encoder: concatenate the vector and image features, then project to 32 features.
- Memory: GRU with 64 hidden units, consuming the encoded observation and previous action.
- Predictor: MLP taking memory and the current action, predicting the next observed vector `(x, x_velocity, angle)`.
- Policy and value: small MLPs taking the memory state. Freeze the world model during initial PPO experiments so the feature distribution stays fixed.

At time t: `h_t = GRU([encode(o_t), a_(t-1)], h_(t-1))`; predict `o_(t+1)` from `[h_t, a_t]`. Reset memory at episode boundaries and give the initial previous action a defined zero encoding. Train on sequences, not shuffled independent transitions. This defines a predictive recurrent representation; it does not yet provide autonomous latent imagination. Add a separate latent transition only if open-loop model rollouts become necessary.

## Why memory matters

Gymnasium's full CartPole observation contains position, cart velocity, pole angle, and angular velocity. Removing angular velocity makes a single vector observation incomplete. [Gymnasium CartPole documentation](https://gymnasium.farama.org/v0.29.0/environments/classic_control/cart_pole/)

The standard renderer draws the cart and pole from position and angle; it does not draw a velocity indicator. Therefore a single added image does not directly reveal the omitted angular velocity. Temporal changes can reveal it, but temporal vector angles already provide that clue. This is an inference from the simulator and renderer, and means visual improvement is not guaranteed. [Gymnasium source](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py)

## Alternatives

| Model | Benefit | Limitation | Role |
| --- | --- | --- | --- |
| Two-to-four observations plus action history, encoded then fed to an MLP | Small and easy to debug; successive angles support velocity estimation | Fixed history; weaker fit for inferring unknown episode parameters over time | Useful cheap baseline |
| Encoder plus deterministic GRU | Keeps history without selecting a fixed window; little machinery | Point predictions do not express uncertainty; recurrence needs sequence training | Recommended main model |
| RSSM | Combines deterministic memory with stochastic latent state; supports uncertain futures and latent rollouts | Adds prior/posterior networks, sampling, and a variational objective | Later extension if needed |

PlaNet introduced an RSSM with both deterministic and stochastic transitions and reported that both mattered for its planning tasks. That is evidence for RSSMs on those tasks, not evidence that they are necessary for this CartPole comparison. [PlaNet](https://planetrl.github.io/)

## Keep the comparison interpretable

Use the shared next observed vector as the first prediction target, with normalized mean squared error. Both arms then learn the same task. Providing the full simulator state, including hidden angular velocity, as a training label is privileged supervision and changes the experiment; reserve it for evaluation or a clearly named separate condition. A held-out probe can measure angular-velocity recovery without feeding its gradient back into the world model.

Do not add image reconstruction only to the visual arm in the first comparison: that changes both input and training objective. An image-only arm can use the same vector labels, but must be described as supervised prediction of vector state from images rather than learning only from images.

Measure prediction error at several horizons on the same held-out trajectories, plus PPO episode return on unseen parameter settings. If using the proposed observation predictor for longer predictions, explicitly feed predicted vectors back through the encoder; combined-input rollout requires a defined treatment of unavailable future images. Teacher-forced multi-step evaluation is not open-loop rollout accuracy.

Equal parameter counts or equal training steps are not equal compute. Include visual preprocessing, encoder forward/backward work, and policy training in a declared budget; report wall time on the same hardware alongside update counts and environment transitions. Treat architecture sizes and sequence length as settings to document, not tuned advantages for one arm.
