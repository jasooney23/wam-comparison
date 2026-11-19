"""CartPole observation settings shared by the environment and models."""

IMAGE_SIZE = 32

STATE_NAMES = {
    "input_cart_pos": "cartPos",
    "input_cart_vel": "cartVel",
    "input_pole_angle": "poleAngle",
    "input_pole_vel": "poleVel",
}
OBSERVATION_DEFAULTS = {
    "history_frames": 4,
    "input_cart_pos": True,
    "input_cart_vel": True,
    "input_pole_angle": True,
    "input_pole_vel": False,
}


def observation_layout(settings):
    """Read the history length and selected state indices in CartPole order.

    Args:
        settings: Environment settings with optional observation overrides.

    Returns:
        History length and a list of selected state indices.

    Raises:
        ValueError: A setting is invalid or no state elements are selected.
    """
    options = OBSERVATION_DEFAULTS | settings
    frames = options["history_frames"]
    if type(frames) is not int or frames < 1:
        raise ValueError("environment.history_frames must be a positive integer")
    indices = []
    for index, key in enumerate(STATE_NAMES):
        if type(options[key]) is not bool:
            raise ValueError(f"environment.{key} must be a boolean")
        if options[key]:
            indices.append(index)
    if not indices:
        raise ValueError("Enable at least one environment input")
    return frames, indices
