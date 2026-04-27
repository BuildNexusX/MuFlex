from __future__ import annotations

import argparse
import ast
import json
import time
import tracemalloc
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.env import MuFlex
from algo.baseline import convert_to_continuous_scale, get_physical_action


@dataclass
class EnvDefinition:
    name: str
    code: str


def load_env_definitions(env_file: Path) -> list[EnvDefinition]:
    """Parse env_list.txt into a list of environment definitions."""
    if not env_file.exists():
        raise FileNotFoundError(f"env_list.txt not found at {env_file}")

    envs: list[EnvDefinition] = []
    current_name: str | None = None
    code_lines: list[str] = []

    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            parts = line[2:].split(maxsplit=1)
            if parts and parts[0].isdigit():
                if current_name and code_lines:
                    envs.append(EnvDefinition(current_name, "\n".join(code_lines).strip()))
                current_name = parts[1].strip() if len(parts) > 1 else f"Env {parts[0]}"
                code_lines = []
            continue

        if line.strip() == "---":
            if current_name and code_lines:
                envs.append(EnvDefinition(current_name, "\n".join(code_lines).strip()))
            current_name = None
            code_lines = []
            continue

        if current_name is not None:
            code_lines.append(line)

    if current_name and code_lines:
        envs.append(EnvDefinition(current_name, "\n".join(code_lines).strip()))

    return envs


def _safe_kw_value(node: ast.AST) -> Any:
    """
    Parse keyword values from envs.
    """
    try:
        return ast.literal_eval(node)
    except Exception:
        if isinstance(node, ast.Name):
            return node.id
        return None


def parse_env_code(code: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract fmu_configs and MuFlex from saved env code."""
    fmu_configs: list[dict[str, Any]] | None = None
    env_kwargs: dict[str, Any] = {}

    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]

            if isinstance(target, ast.Name) and target.id == "fmu_configs":
                try:
                    fmu_configs = ast.literal_eval(node.value)
                except Exception as e:
                    raise ValueError(f"Failed to parse fmu_configs as literal: {e}") from e

            if isinstance(target, ast.Name) and target.id == "env":
                if isinstance(node.value, ast.Call):
                    for kw in node.value.keywords:
                        if kw.arg is None:
                            continue
                        val = _safe_kw_value(kw.value)
                        if val is not None:
                            env_kwargs[kw.arg] = val

    if fmu_configs is None:
        raise ValueError("No fmu_configs found in env definition")

    return fmu_configs, env_kwargs


def normalize_fmu_paths(fmu_configs: list[dict[str, Any]], repo_root: Path) -> list[dict[str, Any]]:
    """Normalize FMU paths"""
    normalized: list[dict[str, Any]] = []
    for cfg in fmu_configs:
        cfg = dict(cfg)
        path_value = str(cfg.get("path", ""))
        if "models_15min/" in path_value.replace("\\", "/"):
            suffix = path_value.replace("\\", "/").split("models_15min/", 1)[1]
            cfg["path"] = str(repo_root / "models_15min" / suffix)
        normalized.append(cfg)
    return normalized


def build_actions(env: MuFlex, fmu_configs: list[dict[str, Any]], action_type: str) -> np.ndarray:
    """Build the normalized action vector based on io_type defaults."""
    io_types = [cfg["io_type"] for cfg in fmu_configs]
    physical_actions = [
        get_physical_action(
            io_type=io_type,
            mins=env.base_mins_list[idx],
            maxs=env.base_maxs_list[idx],
            category=env.fmu_categories[idx],
        )
        for idx, io_type in enumerate(io_types)
    ]

    converted_actions: list[float] = []
    for idx, phys in enumerate(physical_actions):
        mins = env.base_mins_list[idx]
        maxs = env.base_maxs_list[idx]
        if len(phys) != len(mins):
            raise ValueError(f"Incorrect action length for FMU #{idx + 1}")

        for v, mn, mx in zip(phys, mins, maxs):
            if v < mn or v > mx:
                raise ValueError(f"Value {v} out of bounds [{mn}, {mx}] for FMU #{idx + 1}")

        if action_type == "continuous":
            converted_actions.extend(convert_to_continuous_scale(phys, mins, maxs))
        else:
            intervals = env.intervals_list[idx]
            disc = [int(round((p - mn) / inter)) for p, mn, inter in zip(phys, mins, intervals)]
            converted_actions.extend(disc)

    dtype = np.float32 if action_type == "continuous" else np.int64
    return np.array(converted_actions, dtype=dtype)


def get_memory_usage_mb() -> float:
    """Return current RSS memory usage in MB"""
    try:
        import psutil  # type: ignore

        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        pass

    try:
        import resource  # type: ignore

        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024
    except Exception:
        return float("nan")


def run_scalability_test(
    env_name: str,
    env_file: Path,
    max_steps: int | None = None,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parent.parent

    envs = load_env_definitions(env_file)
    selected = next((env for env in envs if env.name == env_name), None)
    if selected is None:
        available = ", ".join(env.name for env in envs)
        raise ValueError(f"Environment '{env_name}' not found. Available: {available}")

    fmu_configs, env_kwargs = parse_env_code(selected.code)
    fmu_configs = normalize_fmu_paths(fmu_configs, repo_root)

    sim_days = env_kwargs.get("sim_days", 1)
    start_date = env_kwargs.get("start_date", 201)
    step_size = env_kwargs.get("step_size", 900)
    action_type = env_kwargs.get("action_type", "continuous")
    include_hour = env_kwargs.get("include_hour", True)
    reward_mode = env_kwargs.get("reward_mode", "example_reward")

    env = MuFlex(
        fmu_configs=fmu_configs,
        sim_days=sim_days,
        start_date=start_date,
        step_size=step_size,
        log_level=7,
        action_type=action_type,
        include_hour=include_hour,
        reward_mode=reward_mode,
        save_results=False,
    )

    env.reset()
    actions = build_actions(env, fmu_configs, action_type)

    done = False
    total_reward = 0.0
    step_count = 0

    while not done:
        _, reward, terminated, truncated, _ = env.step(actions)
        total_reward += float(reward)
        step_count += 1
        done = bool(terminated or truncated)

        if max_steps is not None and step_count >= max_steps:
            break

    env.close()

    return {
        "env_name": env_name,
        "num_fmus": len(fmu_configs),
        "sim_days": sim_days,
        "start_date": start_date,
        "step_size": step_size,
        "action_type": action_type,
        "include_hour": include_hour,
        "reward_mode": reward_mode,
        "max_steps": max_steps,
        "steps_run": step_count,
        "total_reward": total_reward,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MuFlex scalability test.")

    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "src" / "env_list.txt",
        help="Path to env_list.txt",
    )

    parser.add_argument(
        "--env-name",
        default=None,
        help="Environment name in env_list.txt. Use 'all' to run all envs.",
    )

    parser.add_argument(
        "--list-envs",
        action="store_true",
        help="List all environment names found in env_list.txt and exit.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional max steps to run",
    )

    return parser.parse_args()


def main(
    manual_env_name: str | None = None,
    manual_env_file: Path | None = None,
    manual_max_steps: int | None = None,
) -> None:
    args = parse_args()

    if args.list_envs:
        envs = load_env_definitions(args.env_file)
        print("Available environments:")
        for e in envs:
            print(f"  - {e.name}")
        return

    env_file = args.env_file if args.env_file is not None else manual_env_file
    env_name = args.env_name if args.env_name is not None else manual_env_name
    max_steps = args.max_steps if args.max_steps is not None else manual_max_steps

    if env_file is None:
        raise SystemExit("Error: env_file is not set.")
    if env_name is None:
        raise SystemExit("Error: --env-name is required (or set manual_env_name at the bottom).")

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "simulation_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    envs = load_env_definitions(env_file)

    # decide which env to run
    if str(env_name).lower() == "all":
        targets = [e.name for e in envs]
    else:
        targets = [env_name]

    for name in targets:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"scalability_{name}_{ts}.json"

        tracemalloc.start()
        start_time = time.perf_counter()
        start_mem = get_memory_usage_mb()

        run_info = run_scalability_test(
            env_name=name,
            env_file=env_file,
            max_steps=max_steps,
        )

        end_time = time.perf_counter()
        end_mem = get_memory_usage_mb()
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        elapsed = end_time - start_time

        metrics: dict[str, Any] = {
            "timestamp_local": ts,
            "elapsed_seconds": elapsed,
            "memory_rss_start_mb": start_mem,
            "memory_rss_end_mb": end_mem,
            "peak_tracemalloc_mb": peak_mem / (1024 * 1024),
            **run_info,
        }

        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        print(f"[{name}] Done. Steps: {metrics['steps_run']}, Total Reward: {metrics['total_reward']}")
        print(f"[{name}] Elapsed time: {elapsed:.2f}s")
        print(f"[{name}] Memory usage (start): {start_mem:.2f} MB")
        print(f"[{name}] Memory usage (end): {end_mem:.2f} MB")
        print(f"[{name}] Peak traced memory: {metrics['peak_tracemalloc_mb']:.2f} MB")
        print(f"[{name}] Saved results to: {out_path}")

if __name__ == "__main__":
    MANUAL_ENV_NAME = "scal4"
    MANUAL_ENV_FILE = Path(__file__).resolve().parent.parent / "src" / "env_list.txt"
    MANUAL_MAX_STEPS = None
    main(
        manual_env_name=MANUAL_ENV_NAME,
        manual_env_file=MANUAL_ENV_FILE,
        manual_max_steps=MANUAL_MAX_STEPS,
    )
