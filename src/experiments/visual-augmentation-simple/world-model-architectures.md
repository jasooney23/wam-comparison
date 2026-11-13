# Simple action-conditioned world model architectures

Research date: 2026-11-12. Architecture advice below is a design judgment, not an experimental result.

Here, a dynamics model predicts the next state or latent state given the current state and an action. A world model may also include observation, reward, and episode-ending prediction heads. PlaNet explicitly separates transition, observation, and reward models. [PlaNet](https://planetrl.github.io/)

| Architecture | Action-conditioned transition | When it fits | Main cost or limitation |
| --- | --- | --- | --- |
| Deterministic MLP | `next_state = MLP([state, action])`; optionally predict a change and add it to the state | Small state vectors that contain enough information to predict the future | No memory or explicit uncertainty |
| Probabilistic MLP ensemble | Each MLP predicts a Gaussian mean and variance from `[state, action]` | State vectors; uncertainty matters for planning | Several models and sampled rollouts |
| Encoder plus latent MLP | `z = encoder(observation)`; `next_z = MLP([z, action])` | Images or mixed observations where a compact state is sufficient | Must train a useful representation; one frame may omit velocity |
| Encoder plus recurrent dynamics | Update a GRU or LSTM from `[z, action]`; predict the next latent from its memory | Partial observations where history matters | Sequence training and memory management |
| Recurrent state-space model (RSSM) | Recurrent memory uses the previous latent and action; a prior predicts the next stochastic latent | Partial observations, uncertain futures, latent imagination | Prior/posterior networks, sampling, and variational training |

The first two rows follow the deterministic and probabilistic neural dynamics families compared in PETS. Its probabilistic ensemble uses bootstrapped neural networks with Gaussian outputs; disagreement between models represents uncertainty from limited data, while predicted variance represents outcome variability. A single deterministic MLP is a simpler baseline, not the complete PETS algorithm. The optional state-change parameterization above is a design choice. [PETS, sections 4–5](https://arxiv.org/html/1805.12114v1)

The encoder-plus-MLP row describes a simple design pattern rather than one named algorithm. TD-MPC is a representative task-oriented latent dynamics approach: it learns compact dynamics for short-horizon planning together with reward/value signals, without requiring image reconstruction. A useful latent representation needs a training signal beyond a freely learned next-latent match alone. [TD-MPC](https://www.nicklashansen.com/td-mpc/)

For recurrent dynamics, the classic **World Models** architecture uses a convolutional VAE to encode images and an LSTM with a mixture-density head to predict `p(next_z | z, action, memory)`. A deterministic GRU with an MLP head is a simpler adaptation, not the paper's exact architecture. [World Models, model and appendix](https://worldmodels.github.io/)

**PlaNet** introduced the RSSM with both deterministic recurrent memory and stochastic latent state. During observation, a posterior uses the current observation to update the latent belief; during imagination, the prior advances it without future observations. **Dreamer** uses a learned latent world model to train behavior through imagined trajectories. These are more involved starting points than a deterministic recurrent model. [PlaNet](https://planetrl.github.io/), [Dreamer](https://danijar.com/project/dreamer/)

For this CartPole experiment, my starting recommendation is a small MLP for full state observations, or a small encoder plus deterministic recurrent dynamics when angular velocity is hidden. Add probabilistic outputs or an RSSM only when uncertain futures or latent planning are part of the experiment. An observation predictor that requires real future observations to advance its memory is not yet an autonomous latent simulator.
