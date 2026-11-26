## Nov 25, 2027
### Guiding question(s)
- PRH says all models converge on the same representation of the world, even with different modes and applications
- If you combine modes, does it make the model more intelligent in some sense?
    - e.g. does a text+image -> text+image model learn a better representation than a pure text -> text model?

### Cases of Modes & Information
- There are a few different ways of asking the question: "Does adding more modes to a world model make it learn a better representation of the world"?
- Are the different modes providing the same information, or different? Do they show the world perfectly?
    - Case 1: Modes all have perfect information of the world
        - Then, each mode of the WM is simply showing the world from a different perspective, but in theory one mode is enough to capture/reconstruct the entire world
        - No new information is added by different modes. Only different perspectives
        - Idealized case. In any real/applied model, will the model actually have perfect information of the world?
        - I expect that it is probably true that even in this case, more modes can increase intelligence; e.g. if one mode is very obtuse/difficult to extract information out of, then under equal compute adding a more "information-accessible" mode will make the WM better
            - For example, two modes, the world is the current posture of a person walking: Mode 1 is just a 3D surface of the person (maybe a mesh), Mode 2 is that mesh turned into a string and hashed. Of course it will be easier to learn from Mode 1 than Mode 2. 
            - Similar argument: If both 1 and 2 are okay-ish, then combining may produce better results than 1 or 2 alone.
    - Case 2: Modes all have partial information of the world, but they are all realizations of the same underlying information
        - Again, no new information introduced by new modes. Intelligence of the model simply depends on: Do different perspectives of the same world benefit it?
        - Example: Basic cartpole
            - Vector input mode: cart & pole pos with frame stack
            - Image input mode: the whole scene with frame stack
            - From the image, you can only deduce the position of cart and pole at each frame. So, both modes are actually the same information but with different realizations
    - Case 3: Modes all have partial information of the world, and they are realizations of different partial sets of information
        - This case has two moving parts: 
            1. Introducing new modes
            2. Introducing new information
        - Possibly the most practical/relevant to real-world?
            - e.g. under the multimodal model paradigm, image and text are both realizing different subsets of the world. and so a multimodal model is both gaining more information and new perspectives on the world.
        - Example: WeatherNext3 improves by using weather station data in its input and output for training. Station data and reanalysis data are two different realizations of the same world.
            - potentially dubious argument: both are slightly inaccurate realizations of the world, say with some noise (e.g. sensor error, reanalysis bias, etc.).