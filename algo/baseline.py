"""Reference controller used for quick simulations (used in paper).

The functions below provide a baseline policy that sends fixed temperature set-points to each FMU.
"""

from typing import Optional
from pathlib import Path

import numpy as np

from src.env import MuFlex


def get_physical_action(io_type: str, mins: list[float], maxs: list[float], category: str = "building") -> list[float]:
    """Return a nominal physical action vector for one FMU.

    The returned list contains *physical* set-points in the order expected by
    the environment and is clipped to the FMU bounds.
    """
    if io_type == "OfficeS":
        candidate = [25, 25, 25, 25, 25, 15]
    elif io_type == "OfficeM":
        candidate = [25, 25, 25, 15, 15, 15]
    elif io_type == "Energym_House":
        candidate = [0.5]
    elif category.lower() == "pv" or io_type.lower() == "pv":
        candidate = [1.0] * len(mins)
    else:
        candidate = [(mn + mx) / 2.0 for mn, mx in zip(mins, maxs)]

    if len(candidate) != len(mins):
        candidate = [(mn + mx) / 2.0 for mn, mx in zip(mins, maxs)]

    return [min(max(v, mn), mx) for v, mn, mx in zip(candidate, mins, maxs)]


def convert_to_continuous_scale(vals, mins, maxs):
    """Map physical values to the normalized action space ([-1, 1]).

    Parameters
    ----------
    vals, mins, maxs : list[float]
        Parallel lists describing the physical values and their bounds.
    """
    return [(v - mn) / (mx - mn) * 2 - 1 for v, mn, mx in zip(vals, mins, maxs)]


def run_baseline(
    fmu_configs,
    sim_days: int = 1,
    start_date: int = 201,
    step_size: int = 900,
    reward_mode: str = "example_reward",
    save_results: bool = False,
    max_steps: Optional[int] = None,
    action_type: str = "continuous",
    include_hour: bool = True,
    include_day_of_year: bool = True,
    include_episode_progress: bool = True,
    normalize_observation: bool = True,
    rl_control_window_only: bool = True,
    office_hour_start: str = "08:00",
    office_hour_end: str = "18:00",
    step_info_print_interval: int = 10,
    physical_actions: Optional[list[list[float]]] = None,
):
    """Execute the baseline policy for a configurable number of steps.

    Parameters
    ----------
    fmu_configs : list[dict]
        Mapping of FMU paths and their ``io_type`` identifiers.
    sim_days, start_date, step_size : int
        Simulation horizon parameters forwarded to :class:`~src.env.MuFlex`.
    physical_actions : list[list[float]], optional
        Explicit physical actions per FMU; defaults to nominal set-points.
    """
    env = MuFlex(
        fmu_configs=fmu_configs,
        sim_days=sim_days,
        start_date=start_date,
        step_size=step_size,
        log_level=7,
        action_type=action_type,
        reward_mode=reward_mode,
        save_results=save_results,
        include_hour=include_hour,
        include_day_of_year=include_day_of_year,
        include_episode_progress=include_episode_progress,
        normalize_observation=normalize_observation,
        rl_control_window_only=rl_control_window_only,
        office_hour_start=office_hour_start,
        office_hour_end=office_hour_end,
        step_info_print_interval=step_info_print_interval,
    )

    _, _ = env.reset()
    io_types = [cfg["io_type"] for cfg in fmu_configs]

    # ------------------------------------------------------------------
    # Determine the physical action vector for each FMU
    # ------------------------------------------------------------------
    if physical_actions is None:
        physical_actions = [
            get_physical_action(
                io_type=t,
                mins=env.base_mins_list[idx],
                maxs=env.base_maxs_list[idx],
                category=env.fmu_categories[idx],
            )
            for idx, t in enumerate(io_types)
        ]

    if len(physical_actions) != len(io_types):
        raise ValueError("physical_actions length does not match number of FMUs")

    converted_actions: list[float] = []
    for idx, phys in enumerate(physical_actions):
        mins = env.base_mins_list[idx]
        maxs = env.base_maxs_list[idx]
        if len(phys) != len(mins):
            raise ValueError(f"Incorrect action length for FMU #{idx + 1}")
        for v, mn, mx in zip(phys, mins, maxs):
            if v < mn or v > mx:
                raise ValueError(
                    f"Value {v} out of bounds [{mn}, {mx}] for FMU #{idx + 1}"
                )
        if action_type == "continuous":
            converted_actions.extend(convert_to_continuous_scale(phys, mins, maxs))
        else:
            intervals = env.intervals_list[idx]
            disc = [
                int(round((p - mn) / inter))
                for p, mn, inter in zip(phys, mins, intervals)
            ]
            converted_actions.extend(disc)

    dtype = np.float32 if action_type == "continuous" else np.int64
    actions_np = np.array(converted_actions, dtype=dtype)

    # ------------------------------------------------------------------
    # Execute environment steps using constant baseline actions
    # ------------------------------------------------------------------
    done = False
    total_reward = 0.0
    step_count = 0
    while not done:
        next_state, reward, terminated, truncated, _ = env.step(actions_np)
        done = terminated or truncated
        total_reward += reward
        step_count += 1
        if max_steps is not None and step_count >= max_steps:
            break

    env.close()
    print(f"Done. Steps: {step_count}, Total Reward: {total_reward}")


def run_continuous():
    """Quick smoke test using four bundled FMUs."""
    project_root = Path(__file__).resolve().parent.parent
    small_office_dir = project_root / "models" / "small_office"
    medium_office_dir = project_root / "models" / "medium_office"
    fmu_configs = [
        {"path": str(small_office_dir / "small_baseline_v1.fmu"), "io_type": "OfficeS"},
        {"path": str(small_office_dir / "small_baseline_v2.fmu"), "io_type": "OfficeS"},
        {"path": str(medium_office_dir / "medium_baseline_v1.fmu"), "io_type": "OfficeM"},
        {"path": str(medium_office_dir / "medium_baseline_v2.fmu"), "io_type": "OfficeM"},
    ]

    run_baseline(
        fmu_configs=fmu_configs,
        sim_days=1,
        start_date=189,
        step_size=900,
        save_results=True,
    )


if __name__ == "__main__":
    run_continuous()