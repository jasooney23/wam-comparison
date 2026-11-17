TASK: Change the structure of the training in this experiment visual-augmentation-simple.

Currently:
- Training is combined in one script for WM and agent
- WM gets trained first during `wm_pretraining`, then agent gets trained using the frozen WM during `agent_training`
- The WM is pretrained on a synthetic pre-generated dataset drawn from the environment. The physics parameters (mass and damping) are drawn uniformly from a training parameter distribution. 
- During pretraining the WM is evaluated on environments with parameterizations drawn from an out-of-distribution set
- The agent is evaluated using the total return over full play-throughs of environments with parameterizations drawn from an out-of-distribution set

The change:
- Separate WM pretraining and agent training into separate scripts
- The scripts are independent, but work together in the following workflow:
    1. Pre-train the world model.
        - Pre-training outputs the same evaluation metrics as today for the WM
        - The evaluation metrics are saved to both a CSV and tensorboard
        - The script uses one training pathway twice: Once to train a vector input only WM and once to train a vector+image WM
    2. Train the agent on the pre-trained world model
        - Only train if both vector and vector+image pre-trained WMs exist
        - The evaluation metrics are saved to both a CSV and tensorboard
        - The script uses one training pathway twice

