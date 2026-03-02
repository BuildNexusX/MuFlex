"""Example reward script discoverable by MuFlex.

Rename/copy this file to create your own reward mode.
Mode name shown in GUI/environment equals file stem (e.g. ``example_reward``).
"""


def compute_reward(self, scaled_observation):
    """Simple placeholder reward.

    Replace with your own logic. Keep return type as float.
    """
    return 0.0