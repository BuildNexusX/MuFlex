"""Reward-function discovery and loading utilities.

All reward modes are discovered from ``algo/reward/*.py`` and each script must
expose ``compute_reward(self, scaled_observation)``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _reward_dir() -> Path:
    return _repo_root() / "algo" / "reward"


def _iter_reward_scripts() -> list[Path]:
    reward_dir = _reward_dir()
    if not reward_dir.exists():
        return []

    scripts: list[Path] = []
    for path in sorted(reward_dir.glob("*.py")):
        if path.stem.startswith("_") or path.stem == "__init__":
            continue
        scripts.append(path)
    return scripts


def list_available_reward_modes() -> list[str]:
    """Return all selectable reward modes discovered from ``algo/reward``."""
    scripts = _iter_reward_scripts()
    mode_set = {p.stem for p in scripts}

    # Keep common modes in front if present, append the rest alphabetically.
    preferred = [mode for mode in ("default", "custom") if mode in mode_set]
    others = sorted(mode for mode in mode_set if mode not in {"default", "custom"})
    return preferred + others


def reward_script_path(mode: str) -> Path | None:
    """Return source file path for a reward mode, when resolvable."""
    path = _reward_dir() / f"{mode}.py"
    return path.resolve() if path.exists() else None


def resolve_reward_function(mode: str) -> Callable:
    """Resolve and return ``compute_reward(self, scaled_observation)`` for mode."""
    path = _reward_dir() / f"{mode}.py"
    if not path.exists():
        available = ", ".join(list_available_reward_modes())
        raise ValueError(
            f"Unknown reward_mode='{mode}'. Reward scripts must be in algo/reward. "
            f"Available: [{available}]"
        )

    module_name = f"algo.reward.{mode}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load reward module from: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reward_fn = getattr(module, "compute_reward", None)
    if not callable(reward_fn):
        raise AttributeError(f"Reward script '{path}' must define callable compute_reward(self, scaled_observation)")
    return reward_fn
