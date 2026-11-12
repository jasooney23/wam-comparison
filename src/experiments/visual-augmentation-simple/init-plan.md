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
            

## Guidelines
- Envs should be built under the Gymnasium structure
- Build models modularly as they may be reused
    - Specifically, build an MLP class which can serve as backbone for many WAM/policy/value/etc. networks for WAMs/agents
- Use pytorch
    - Support monitoring with tensorboard
