import gc
from typing import Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.env import MuFlex as MuFlexCore, IO_DEFINITIONS, log_rl


class MuFlex(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        fmu_configs,
        sim_days: int = 1,
        start_date: int = 1,
        step_size: int = 900,
        log_level: int = 7,
        action_type: str = "continuous",
        reward_mode: str = "default",
        save_results: bool = False,
        include_hour: bool = True,
        include_day_of_year: bool = True,
        include_episode_progress: bool = True,
        normalize_observation: bool = True,
        double_reset: bool = False,
        rl_control_window_only: bool = True,
        office_hour_start: str = "08:00",
        office_hour_end: str = "18:00",
        step_info_print_interval: int = 10,
    ):
        super().__init__()

        self.fmu_configs = fmu_configs
        self.num_fmus = len(fmu_configs)

        self.sim_days = sim_days
        self.start_date = start_date
        self.step_size = step_size
        self.log_level = log_level
        self.action_type = action_type.lower()
        self.reward_mode = reward_mode

        self.include_hour = include_hour
        self.include_day_of_year = include_day_of_year
        self.include_episode_progress = include_episode_progress
        self.normalize_observation = normalize_observation
        self.double_reset = bool(double_reset)

        self.rl_control_window_only = bool(rl_control_window_only)
        self.office_hour_start = str(office_hour_start)
        self.office_hour_end = str(office_hour_end)
        self.step_info_print_interval = max(0, int(step_info_print_interval))

        self._save_results = bool(save_results)
        self._episode_counter = 0
        self._env: Optional[MuFlexCore] = None

        self._input_dims_list = []
        self._output_names_list = []

        for cfg in self.fmu_configs:
            io_type = cfg["io_type"]
            io_def = IO_DEFINITIONS[io_type]
            self._input_dims_list.append(list(io_def["dims"]))
            self._output_names_list.append(list(io_def["OUTPUTS"]))

        self._build_action_space()
        self._build_observation_space()

    @property
    def save_results(self):
        if self._env is not None:
            return self._env.save_results
        return self._save_results

    @save_results.setter
    def save_results(self, value: bool):
        self._save_results = bool(value)
        if self._env is not None:
            self._env.save_results = bool(value)

    def _build_action_space(self):
        if self.action_type == "continuous":
            total_dims = sum(len(dims) for dims in self._input_dims_list)
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(total_dims,),
                dtype=np.float32,
            )
        elif self.action_type == "discrete":
            discrete_dims = []
            for dims in self._input_dims_list:
                discrete_dims.extend(dims)
            self.action_space = spaces.MultiDiscrete(discrete_dims)
        else:
            assert False, "Unsupported action_type"

    def _time_feature_dims(self) -> int:
        dims = 0
        if self.include_hour:
            dims += 2
        if self.include_day_of_year:
            dims += 2
        if self.include_episode_progress:
            dims += 1
        return dims

    def _build_observation_space(self):
        total_output_dims = sum(len(outs) for outs in self._output_names_list)
        time_feature_dims = self._time_feature_dims()
        observation_dim = total_output_dims + time_feature_dims

        obs_low = np.zeros(observation_dim, dtype=np.float32)
        obs_high = np.ones(observation_dim, dtype=np.float32)

        index_offset = 0
        if self.include_hour:
            obs_low[0:2] = -1.0
            obs_high[0:2] = 1.0
            index_offset = 2
        if self.include_day_of_year:
            obs_low[index_offset:index_offset + 2] = -1.0
            obs_high[index_offset:index_offset + 2] = 1.0
            index_offset += 2
        if self.include_episode_progress:
            obs_low[index_offset] = 0.0
            obs_high[index_offset] = 1.0
            index_offset += 1

        for cfg in self.fmu_configs:
            io_type = cfg["io_type"]
            io_def = IO_DEFINITIONS[io_type]
            local_low = io_def["ob_base_low"]
            local_high = io_def["ob_base_high"]
            length = len(local_low)
            obs_low[index_offset:index_offset + length] = local_low
            obs_high[index_offset:index_offset + length] = local_high
            index_offset += length

        if self.normalize_observation:
            box_low = np.zeros(observation_dim, dtype=np.float32)
            box_high = np.ones(observation_dim, dtype=np.float32)
        else:
            box_low = obs_low.copy()
            box_high = obs_high.copy()

        self.observation_space = spaces.Box(
            low=box_low,
            high=box_high,
            shape=(observation_dim,),
            dtype=np.float32,
        )

    def _make_inner_env(self):
        return MuFlexCore(
            fmu_configs=self.fmu_configs,
            sim_days=self.sim_days,
            start_date=self.start_date,
            step_size=self.step_size,
            log_level=self.log_level,
            action_type=self.action_type,
            reward_mode=self.reward_mode,
            save_results=self._save_results,
            include_hour=self.include_hour,
            include_day_of_year=self.include_day_of_year,
            include_episode_progress=self.include_episode_progress,
            normalize_observation=self.normalize_observation,
            rl_control_window_only=self.rl_control_window_only,
            office_hour_start=self.office_hour_start,
            office_hour_end=self.office_hour_end,
            step_info_print_interval=self.step_info_print_interval,
        )

    def reset(self, seed=None, options=None):
        if self._env is not None:
            try:
                self._env.close()
            except Exception as e:
                log_rl(f"Previous inner env close failed: {e}", tag="Wrapper Reset")
            finally:
                self._env = None

        gc.collect()

        self._env = self._make_inner_env()

        obs, info = self._env.reset(seed=seed, options=options)

        if self.double_reset:
            obs, info = self._env.reset(seed=None, options=options)

        self._episode_counter += 1
        return obs, info

    def step(self, action):
        if self._env is None:
            raise RuntimeError("MuFlex wrapper: step() called before reset().")
        return self._env.step(action)

    def close(self):
        if self._env is not None:
            try:
                self._env.close()
            except Exception as e:
                log_rl(f"Inner env close failed: {e}", tag="Wrapper Close")
            finally:
                self._env = None

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if self._env is not None and hasattr(self._env, name):
            return getattr(self._env, name)
        raise AttributeError(name)