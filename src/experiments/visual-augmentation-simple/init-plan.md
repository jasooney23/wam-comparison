# Experiment: visual-augmentation-simple
## Background
- Guiding question: Does adding visual input to an action-conditioned world model (with partial observability of the environment state) improve its understanding of the world (environment) under equal compute budgets?
- Motivation: 
    - Immediate: 
        - Achieving better WMs under equal compute constraints may be possible by spreading out training compute cost over multiple data domains
        - If performance does improve, it points towards supporting the following long-horizon motivation
    - Long-horizon: The Platonic Representation Hypothesis suggests that all models converge on the same representation of the world. If so, a world model could learn "better" representations of the world if it learns from more domains of data (visual, text, lived experience, etc.) than just a single mode, because more domains provide a more complete scope of the world, enabling better learned representations.

## Approach
- Compare two simple RL agents built with world-action-models (WAMs). 
    - Agents run in simple environments
        - To start: Cartpole env
    - Evaluate based on how the WAMs/agents perform in unfamiliar parameterizations of the environment
        - To start: The cartpole is parameterized by the mass of the cart relative to the pole and the friction of the pin securing the pole to the cart
        - Agent evaluated by episode score
        - WAM evaluated by {tbd}
- Environment
    - Actions and rewards same as Gymnasium cartpole
    - Input
        - Vector input (both agents): cart position, cart velocity, pole angle
        - Image rendering of the env
        - Input is a stack of the past G = 4 frames
    - Importantly:
        - The vector state should not give full information of the state, otherwise image input to the WAM may offer no benefit
        - The visual and vector inputs should be independent from environment parameters to avoid shortcuts for the model when evaluating in out of distribution parameterizations
- agents
    - Use a simple generic PPO policy, learning algorithm, hyperparameters etc.
        - Use generic, simple hyperparameters - the goal is not necessarily performance, but comparison of these two agents
    - WAM spec: 
        - Start with a simple model that supports vector input, image input, and both at the same time
        - Use encode -> dynamic predictor -> decode architecture
            - Encoder
                - MLP for vector input
                - CNN for image input
                - Concat outputs, for any combination of inputs linear project to fixed latent size 16
                    - Encoder heads for MLP, CNN do not change output size
            - Dynamics net
                - MLP with agent action as input, predicts to/from latent space
            - Decoder
                - MLP for vector
                - transpose CNN for image
            - All MLPs have one hidden layer of 32 units 
        - Train WAM end to end

- Measuring compute
    - Measure compute in the following ways:
        - Wall clock time
        - Environment steps
        - Training updates (WM and agent separately)

- Evaluation
    - Use the following as evaluation metrics:
    - WM: L2 of each state variable separately. i.e one loss for cart position, one for cart velocity, and one for pole angle
        - Evaluate up to the horizon EVAL_HORIZON = 4. e.g. evaluate the error directly reconstructing the state, error after one step on a uniformly sampled action with the dynamics latent, then again, all the way up to 4 dynamics steps. EVAL_HORIZON = 0 means no horizon evaluation, and only evaluation on the immediate step.
    - Agent: Episodic return

- Training
    - Pre-train the WM on a fixed dataset
        - Checkpoint at periodic intervals for number of training updates/passes of dataset (e.g. one checkpoint every 20 epochs)
    - Then train the agent
        - The WM is frozen during agent training
    - Separate the training phases: training script can be set to traing WM > Agent, or use existing WM and train a new agent only

- WM Pre-training dataset
    - Generate by artificially constructing possible states for the env
        - e.g. one training pair is state = [cart_pos, cart_vel, pole_angle] = [-1, -0.5, 0.3, ...], and the image uniformly sampled action, and the env's next state + image
        - This is NOT representative of the distribution the WM will actually experience as a result of the agent nor does it really guarantee valid states
            - BUT it's OK for a simple env and toy experiment

## Guidelines
- Envs should be built under the Gymnasium structure
- Build models modularly as they may be reused
    - Specifically, build an MLP class which can serve as backbone for many WAM/policy/value/etc. networks for WAMs/agents
- Use pytorch
    - Support monitoring with tensorboard


### End state
- WAM which is trained via reconstruction and dynamics loss
    - Reconstruction: Enc to z > Dec to x, L2 dec x against actual
        - Do this for vector and image separately; average vector and image separately
    - Dynamics: Enc to zt > Predict zt+1 with at > Dec to xt+1, L2 dec xt+1 against actual
        - Average
    - Losses summed at equal weights for now
- Agent trained with the simplest PPO possible
- Vector state does NOT contain full information of the environment
- Evaluation at periodic intervals
    - Evaluate on some number of episodes
    - env parameters draw from the non-training parameters so WAM and agents only see out of distribution episodes
    - evaluate on same seed for env start and drawing parameters