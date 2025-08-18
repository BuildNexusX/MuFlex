"""Replay buffer utilities for off‑policy RL.

This module provides two small helpers frequently used in RL:

``process_state`` — strips unused features from raw environment states.
``ReplayBuffer`` — a fixed‑capacity FIFO memory with (optional) persistence
  and a tiny hook for injecting *prior* transitions (e.g. expert demos).


Nothing here is framework‑specific; it works with both NumPy and PyTorch
networks out of the box.
"""

import numpy as np
import random
import pickle
from collections import deque

# -----------------------------------------------------------------------------
# Feature‑selection helper -----------------------------------------------------
# -----------------------------------------------------------------------------

# Indices to *discard* from the raw state vector.  Update this list if the environment's observation layout changes.
DROP_INDICES = [
    1,
]


def process_state(raw_state):
    """Return a cleaned, 1‑D NumPy array suitable for the agent.

    Parameters
    ----------
    raw_state : array‑like
        Observation as produced by the environment.

    Steps
    -----
    1. Cast input to ``np.float32`` – most networks expect that dtype.
    2. Flatten if a higher‑rank tensor comes in (common with image obs).
    3. Drop pre‑selected indices defined in :pydata:`DROP_INDICES`.

    Returns
    -------
    np.ndarray
        Filtered 1‑D array ready to feed into a policy network.
    """

    # Ensure ``raw_state`` is a NumPy array of float32.
    if not isinstance(raw_state, np.ndarray):
        raw_state = np.array(raw_state, dtype=np.float32)

    # Flatten multi‑dim tensors – keeps the helper generic.
    if raw_state.ndim > 1:
        raw_state = raw_state.flatten()

    # Boolean mask for the indices we want to **keep**.
    valid_indices = [i for i in range(len(raw_state)) if i not in DROP_INDICES]
    return raw_state[valid_indices]

# -----------------------------------------------------------------------------
# Experience replay implementation -------------------------------------------
# -----------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed‑size circular buffer storing ``(s, a, r, s', done)`` tuples.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions kept in memory.  Once full, the oldest
        samples are discarded *first‑in‑first‑out*.
    """

    def __init__(self, capacity):
        self.capacity = int(capacity)
        # ``deque`` with ``maxlen`` handles FIFO eviction automatically.
        self.buffer = deque(maxlen=self.capacity)

    # ------------------------------------------------------------------
    # Core API ----------------------------------------------------------
    # ------------------------------------------------------------------

    def push(self, state, action, reward, next_state, done):
        """Insert one transition tuple into the buffer.

        Older samples are dropped automatically when ``capacity`` is exceeded.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Uniformly sample a *mini‑batch* of transitions.

        Returns a generator of stacked NumPy arrays in the canonical DQN order
        ``(states, actions, rewards, next_states, dones)``.
        """
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        """Number of stored transitions."""
        return len(self.buffer)

    # ------------------------------------------------------------------
    # Extras ------------------------------------------------------------
    # ------------------------------------------------------------------

    def add_prior_knowledge(self, state, action, reward, next_state, done, num_copies=1):
        """Duplicate a transition *num_copies* times.

        Useful for seeding the buffer with expert demonstrations so that they
        appear more frequently during early training.
        """
        for _ in range(num_copies):
            self.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    # Persistence helpers ----------------------------------------------
    # ------------------------------------------------------------------

    def save_buffer(self, filepath):
        """Pickle the current buffer to *filepath* (binary mode)."""
        with open(filepath, "wb") as f:
            pickle.dump(list(self.buffer), f)

    def load_buffer(self, filepath):
        """Load transitions from a pickled file, respecting ``capacity``."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        # ``deque`` with ``maxlen`` will truncate if the file is larger than
        # the configured capacity.
        self.buffer = deque(data, maxlen=self.capacity)

    def clear(self):
        """Remove *all* stored transitions without changing ``capacity``."""
        self.buffer.clear()
