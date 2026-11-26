# Partial visual augmentation

Does adding an image improve an action-conditioned world model when both
modalities observe the same part of the state?

- Compare vector input with combined vector and image input.
- The vector contains the current horizontal cart position and pole angle.
- The image is the current Gymnasium CartPole scene, converted to grayscale
  and resized to 32 by 32 pixels.
- Use default Gymnasium CartPole-v1 physics and episode limits throughout.
- Train the world models with L2 reconstruction and next-step prediction losses.
- Evaluate the world models on separately sampled transitions and the PPO
  agents on separately seeded episodes in the same environment.
