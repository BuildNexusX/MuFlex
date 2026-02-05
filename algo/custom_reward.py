"""Template for user-defined reward in MuFlex environments.

Edit this file to create your own reward calculation.  The environment
will import :func:`compute_reward` when ``reward_mode='custom'`` is used
in :class:`src.env.MuFlex`.

Function signature
------------------
``compute_reward(self, scaled_observation)``

``self``
    Instance of :class:`src.env.MuFlex`.  You can access any of the
    environment attributes (e.g. ``self.fmu_configs`` or
    ``self.time_steps``) to help with the calculation.
``scaled_observation``
    Observation array normalised to the range ``[-1, 1]``.  To obtain the
    physical values you can use::

        raw_observation = (
            scaled_observation * (self.obs_high - self.obs_low) + self.obs_low
        )

Return value
------------
The function must return a single ``float`` representing the reward for
the current step (higher is better).

For a complete example refer to ``algo/default_reward.py``.
"""

from __future__ import annotations


def compute_reward(self, scaled_observation):
    """Example custom reward function.

    Replace the body of this function with your own logic.  The default
    implementation simply returns ``0.0`` for every step.
    """

    # TODO: implement custom reward logic here.
    return 0.0
