"""Gymnasium environment wrapping multiple building FMUs (Functional Mock-up Units)"""

import numpy as np
import pandas as pd
import os
import datetime
import re
from pyfmi import load_fmu
import gymnasium as gym
from gymnasium import spaces
from src.config import load_fmu_config
from src.reward_registry import resolve_reward_function

# -----------------------------------------------------------------------------
# Colour-helper print functions -------------------------------------------------
# -----------------------------------------------------------------------------

def blue_print(*args, **kwargs):
    print("\033[31m", *args, "\033[0m", **kwargs)

def green_print(*args, **kwargs):
    print("\033[32m", *args, "\033[0m", **kwargs)

def yellow_print(*args, **kwargs):
    print("\033[33m", *args, "\033[0m", **kwargs)

def magenta_print(*args, **kwargs):
    print("\033[35m", *args, "\033[0m", **kwargs)

_RESET = "\033[0m"
_RL_COLOR = "\033[35m"
_FMU_COLOR = "\033[31m"

def _format_prefix(category, tag=None):
    parts = [category]
    if tag:
        parts.append(tag)
    return "[" + "][".join(parts) + "]"

def _log(message_parts, *, category, tag=None, color=None):
    prefix = _format_prefix(category, tag)
    message = " ".join(str(part) for part in message_parts)
    print(f"{color}{prefix} {message}{_RESET}")

def log_rl(*messages, tag=None):
    _log(messages, category="RL", tag=tag, color=_RL_COLOR)

def log_fmu(*messages, tag=None):
    _log(messages, category="FMU", tag=tag, color=_FMU_COLOR)

def _load_io_definitions():
    """Load I/O definitions for each FMU archetype from config."""
    raw = load_fmu_config()
    io_defs = {}
    for name, cfg in raw.items():
        io_defs[name] = {
            "INPUTS": cfg["INPUTS"],
            "OUTPUTS": cfg["OUTPUTS"],
            "ob_base_low": np.array(cfg["ob_base_low"], dtype=np.float32),
            "ob_base_high": np.array(cfg["ob_base_high"], dtype=np.float32),
            "dims": cfg["dims"],
            "intervals": cfg["intervals"],
            "base_mins": cfg["base_mins"],
            "base_maxs": cfg["base_maxs"],
            "category": cfg.get("category", "building"),
        }
    return io_defs

IO_DEFINITIONS = _load_io_definitions()

def _format_category_label(category: str) -> str:
    """Return a label for a given FMU category."""
    return "PV" if category.lower() == "pv" else "Building"


# -----------------------------------------------------------------------------
# Gymnasium environment --------------------------------------------------------
# -----------------------------------------------------------------------------
class MuFlex(gym.Env):
    """MuFlex - A Physics-based Platform for Multi-Building Flexibility Analysis and Coordination

    The environment instantiates one FMU per building and steps them in lock-step
    every `step_size` seconds.  An agent receives observations and returns either
    continuous or discrete actions that are mapped to physical set-points.

    Parameters
    ----------
    fmu_configs : List[dict]
        Each dict must contain keys ``{'path', 'io_type'}`` where *io_type* matches
        an FMU type defined in *Add_FMU.py*.
    sim_days : int, default=1
        Number of simulated days per episode (simulation always starts at
        *start_date* 00:00).
    start_date : int, default=1
        Start day of the EnergyPlus weather file (1–365).  Only affects FMU
        internal calendars; *does not* influence reward logic directly.
    step_size : int, default=900
        Simulation step [s] (900s = 15min).
    log_level : int, default=7
        PyFMI log level (1=Fatal … 7=Debug).
    action_type : {'continuous', 'discrete'}, default='continuous'
        * continuous → actions in ``[-1, 1]`` mapped linearly to physical range.
        * discrete   → integer bins defined in ``dims_*`` tables.
    reward_mode : str, default='example_reward'
        Reward mode name. Uses scripts from ``algo/reward/<mode>.py``.
    save_results : bool, default=False
        Persist per-step I/O and reward traces to ``./simulation_data_<timestamp>/``.
    """

    def __init__(
        self,
        fmu_configs,
        sim_days: int = 1,
        start_date: int = 1,
        step_size: int = 900,
        log_level: int = 7,
        action_type: str = "continuous",
        reward_mode: str = "example_reward",
        save_results: bool = False,
        include_hour: bool = True,
        include_day_of_year: bool = True,
        include_episode_progress: bool = True,
        normalize_observation: bool = True,
        rl_control_window_only: bool = True,
        office_hour_start: str = "08:00",
        office_hour_end: str = "18:00",
        step_info_print_interval: int = 10,
    ):
        # Set up FMUs, simulation horizon and reward parameters -------------------------------------------------
        self.fmu_configs = fmu_configs
        self.num_fmus = len(fmu_configs)
        self.sim_days = sim_days
        self.start_date = start_date
        self.step_size = step_size
        self.log_level = log_level
        self.action_type = action_type.lower()
        self.save_results = save_results
        self.include_hour = include_hour
        self.include_day_of_year = include_day_of_year
        self.include_episode_progress = include_episode_progress
        self.normalize_observation = normalize_observation
        self.rl_control_window_only = bool(rl_control_window_only)
        self.office_hour_start = str(office_hour_start)
        self.office_hour_end = str(office_hour_end)
        self.office_hour_start_seconds = self._parse_hour_string_to_seconds(self.office_hour_start)
        self.office_hour_end_seconds = self._parse_hour_string_to_seconds(self.office_hour_end)
        self.step_info_print_interval = max(0, int(step_info_print_interval))
        self.reward_list = []
        self.reward_mode = reward_mode
        import types
        reward_function = resolve_reward_function(reward_mode)
        self.compute_reward = types.MethodType(reward_function, self)

        # Folder for optional result export --------------------------------------
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_folder = f"simulation_data_{now}"
        if self.save_results:
            log_rl("Saving results is enabled.", tag="Config")

        # ------------------------------------------------------------------
        # Time axis construction
        # ------------------------------------------------------------------
        self.start_time_seconds = 86_400 * (self.start_date - 1)
        self.stop_time_seconds = self.start_time_seconds + 86_400 * self.sim_days
        # One simulation day is always 86,400 seconds, independent of step size.
        self.time_steps = np.arange(
            self.start_time_seconds,
            self.stop_time_seconds + self.step_size,
            self.step_size,
        )
        self.num_steps = len(self.time_steps)

        # Runtime state -----------------------------------------------------
        self.current_step = 0
        self.done = False
        self.truncated = False

        # ------------------------------------------------------------------
        # FMU initialisation loop
        # ------------------------------------------------------------------
        self.fmus = []                # list[pyfmi.FMUModelCS2]
        self.input_names = []         # list[list[str]]
        self.output_names = []        # list[list[str]]
        self.record_dataframes = []   # mirrors per-FMU I/O for later export
        self.input_dims_list = []
        self.intervals_list = []
        self.base_mins_list = []
        self.base_maxs_list = []
        self.fmu_categories = []      # list[str]
        self.fmu_labels = []          # list[str]
        self._category_counts = {}    # category -> count

        for fmu_index, one_config in enumerate(self.fmu_configs):
            fmu_path = one_config["path"]
            io_type_name = one_config["io_type"]
            category_name = IO_DEFINITIONS[io_type_name].get("category", "building")
            self.fmu_categories.append(category_name)
            category_count = self._category_counts.get(category_name, 0) + 1
            self._category_counts[category_name] = category_count
            label_prefix = _format_category_label(category_name)
            fmu_label = f"{label_prefix} FMU #{category_count}"
            self.fmu_labels.append(fmu_label)

            # --- Load and initialise the FMU --------------------------------
            log_fmu(
                f"Loading {self.fmu_labels[fmu_index]}: "
                f"io_type = {io_type_name}, path = {fmu_path}",
                tag="Init",
            )
            fmu_object = load_fmu(fmu_path, kind="cs", log_level=self.log_level)
            log_fmu(
                f"Initializing {self.fmu_labels[fmu_index]}, time range = "
                f"({self.start_time_seconds}, {self.stop_time_seconds})",
                tag="Init",
            )
            fmu_object.initialize(
                start_time=self.start_time_seconds,
                stop_time=self.stop_time_seconds,
            )
            self.fmus.append(fmu_object)

            # --- Cache I/O metadata for fast access ------------------------
            input_list = IO_DEFINITIONS[io_type_name]["INPUTS"]
            output_list = IO_DEFINITIONS[io_type_name]["OUTPUTS"]
            dims_list = IO_DEFINITIONS[io_type_name]["dims"]
            interval_list = IO_DEFINITIONS[io_type_name]["intervals"]
            base_min_list = IO_DEFINITIONS[io_type_name]["base_mins"]
            base_max_list = IO_DEFINITIONS[io_type_name]["base_maxs"]

            self.input_names.append(input_list)
            self.output_names.append(output_list)
            self.input_dims_list.append(dims_list)
            self.intervals_list.append(interval_list)
            self.base_mins_list.append(base_min_list)
            self.base_maxs_list.append(base_max_list)

            record_dataframe = pd.DataFrame(columns=input_list + output_list)
            self.record_dataframes.append(record_dataframe)

        # Gymnasium spaces ---------------------------------------------------
        self.build_action_space()
        self.build_observation_space()

        # Reward coefficients ----------------------------------------------
        self.penalty = 20.0  # legacy coefficient (unused in current formula)
        # Comfort band per building type (°C)
        self.temp_interval = [(23, 25)] * self.num_fmus

    # ---------------------------------------------------------------------
    # Gym Space builders -------------------------------------------------------
    # ---------------------------------------------------------------------
    def build_action_space(self):
        """Construct the Gymnasium action space."""
        if self.action_type == "continuous":
            # Flatten all FMU inputs → single Box of shape (total_dims,).
            total_dims = sum(len(dims) for dims in self.input_dims_list)
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(total_dims,), dtype=np.float32
            )
        elif self.action_type == "discrete":
            # Each element is an integer index into the respective bin.
            discrete_dims = [dim for dims in self.input_dims_list for dim in dims]
            self.action_space = spaces.MultiDiscrete(discrete_dims)
        else:
            raise ValueError("Unsupported action_type")

    def build_observation_space(self):
        """Construct the Gymnasium observation space."""
        total_output_dimensions = sum(len(outputs) for outputs in self.output_names)
        time_feature_dims = self._time_feature_dims()
        observation_dimension = total_output_dimensions + time_feature_dims

        # Raw bounds used internally for normalization.
        self.obs_low = np.zeros(observation_dimension, dtype=np.float32)
        self.obs_high = np.ones(observation_dimension, dtype=np.float32)

        index_offset = 0
        if self.include_hour:
            self.obs_low[0:2] = -1.0
            self.obs_high[0:2] = 1.0
            index_offset = 2
        if self.include_day_of_year:
            self.obs_low[index_offset:index_offset + 2] = -1.0
            self.obs_high[index_offset:index_offset + 2] = 1.0
            index_offset += 2
        if self.include_episode_progress:
            self.obs_low[index_offset] = 0.0
            self.obs_high[index_offset] = 1.0
            index_offset += 1

        for one_config in self.fmu_configs:
            io_type_name = one_config["io_type"]
            local_low = IO_DEFINITIONS[io_type_name]["ob_base_low"]
            local_high = IO_DEFINITIONS[io_type_name]["ob_base_high"]
            length = len(local_low)
            self.obs_low[index_offset : index_offset + length] = local_low
            self.obs_high[index_offset : index_offset + length] = local_high
            index_offset += length

        # Public observation space matches returned observations.
        if self.normalize_observation:
            box_low = np.zeros(observation_dimension, dtype=np.float32)
            box_high = np.ones(observation_dimension, dtype=np.float32)
        else:
            box_low = self.obs_low.copy()
            box_high = self.obs_high.copy()

        self.observation_space = spaces.Box(
            low=box_low, high=box_high, shape=(observation_dimension,), dtype=np.float32
        )

    # ---------------------------------------------------------------------
    # Standard Gym methods -------------------------------------------------
    # ---------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """Reset environment and move to the first RL-control step if needed."""
        super().reset(seed=seed)
        self.current_step = 0
        self.done = False
        self.truncated = False

        # Clear in-memory DataFrames and transition records.
        for record_dataframe in self.record_dataframes:
            record_dataframe.drop(record_dataframe.index, inplace=True)
        self.reward_list.clear()

        # Same office-hour handling as the reference code:
        # outside the RL control window, advance the FMUs with default actions
        # but do not collect these steps as RL transitions/rewards.
        if self.rl_control_window_only:
            default_action = self._default_action()
            while (not self.done) and (not self._is_rl_control_step(self.current_step)):
                self._advance_one_step(default_action, collect_transition_data=False)

        if self.done:
            raw_observation = self.get_observation(self.time_steps[-1])
        else:
            raw_observation = self.get_observation(self.time_steps[self.current_step])
        obs = self.format_observation(raw_observation)
        return obs, {}

    def step(self, action):
        """Execute one RL transition during the office-hour control window."""
        if self.done:
            last_observation = self.get_observation(self.time_steps[-1])
            return (
                self.format_observation(last_observation),
                0.0,
                True,
                True,
                {"TimeLimit.truncated": True},
            )

        if self.rl_control_window_only and not self._is_rl_control_step(self.current_step):
            raise RuntimeError(
                f"step() called outside RL control window at step {self.current_step}. "
                "Call reset() first and only step using returned transitions."
            )

        next_obs, reward_value, done, truncated = self._advance_one_step(
            action,
            collect_transition_data=True,
        )

        if self.rl_control_window_only:
            default_action = self._default_action()
            while (not self.done) and (not self._is_rl_control_step(self.current_step)):
                next_obs, _, done, truncated = self._advance_one_step(
                    default_action,
                    collect_transition_data=False,
                )

        return next_obs, reward_value, done, truncated, {"TimeLimit.truncated": truncated}

    # ---------------------------------------------------------------------
    # Helper / utility methods -------------------------------------------
    # ---------------------------------------------------------------------
    def _default_action(self):
        """Return the default action used outside office hours."""
        default_values = []
        for fmu_index in range(self.num_fmus):
            default_values.extend([1.0] * len(self.input_dims_list[fmu_index]))

        if self.action_type == "continuous":
            return np.array(default_values, dtype=np.float32)

        discrete_action = []
        cursor = 0
        for fmu_index in range(self.num_fmus):
            dims = self.input_dims_list[fmu_index]
            mins = self.base_mins_list[fmu_index]
            intervals = self.intervals_list[fmu_index]
            for dim_index, dim_size in enumerate(dims):
                target_val = default_values[cursor]
                base_min = float(mins[dim_index])
                interval = float(intervals[dim_index])
                if interval <= 0:
                    idx = 0
                else:
                    idx = int(round((target_val - base_min) / interval))
                idx = max(0, min(dim_size - 1, idx))
                discrete_action.append(idx)
                cursor += 1

        return np.array(discrete_action, dtype=np.int64)

    def _is_rl_control_step(self, step_index: int) -> bool:
        """Return True when the current step is inside the office-hour window."""
        if not self.rl_control_window_only:
            return True
        if step_index >= self.num_steps:
            return False

        current_time = int(self.time_steps[step_index])
        seconds_in_day = (current_time - self.start_time_seconds) % 86_400
        return self.office_hour_start_seconds <= seconds_in_day < self.office_hour_end_seconds

    @staticmethod
    def _parse_hour_string_to_seconds(hour_string: str) -> int:
        """Parse office-hour string in HH:MM format into seconds."""
        match = re.fullmatch(r"([01]\d|2[0-3]):([0-5]\d)", hour_string)
        if match is None:
            raise ValueError(f"Invalid office hour format: {hour_string}. Expected HH:MM.")
        hour = int(match.group(1))
        minute = int(match.group(2))
        return hour * 3600 + minute * 60

    def _advance_one_step(self, action, collect_transition_data: bool):
        """Advance one physical simulation step.

        collect_transition_data=True means this step is an RL transition.
        collect_transition_data=False means this is an outside-office-hour default step.
        """
        self._validate_action(action)
        unscaled_actions = self.unscale_action(action)

        current_step = self.current_step
        current_time = self.time_steps[current_step]

        for fmu_index, fmu_object in enumerate(self.fmus):
            input_var_names = self.input_names[fmu_index]
            input_values = unscaled_actions[fmu_index]
            for name_item, value_item in zip(input_var_names, input_values):
                fmu_object.set(name_item, value_item)
            self.record_dataframes[fmu_index].loc[current_step, input_var_names] = input_values

        for fmu_object in self.fmus:
            fmu_object.do_step(current_t=current_time, step_size=self.step_size, new_step=True)

        self.record_outputs(current_step)
        raw_next_observation = self.get_observation(current_time + self.step_size)
        next_obs = self.format_observation(raw_next_observation)
        reward_value = self.compute_reward(next_obs)

        if collect_transition_data:
            self.reward_list.append({"Step": current_step, "TotalReward": reward_value})

        if self._should_print_step_info(current_step):
            control_mode = "RL" if collect_transition_data else "DEFAULT"
            self._print_step_info(
                action,
                unscaled_actions,
                raw_next_observation,
                next_obs,
                reward_value,
                control_mode,
            )

        self.current_step += 1
        if self.current_step >= self.num_steps:
            self.done = True
            self.truncated = True
        else:
            self.truncated = False

        return next_obs, reward_value, self.done, self.truncated

    def _validate_action(self, action) -> None:
        """Validate the flattened action before applying it to FMUs."""
        if self.action_type == "discrete":
            start = 0
            for fmu_idx in range(self.num_fmus):
                dims = self.input_dims_list[fmu_idx]
                end = start + len(dims)
                sub_action = action[start:end]
                for dim_index, val in enumerate(sub_action):
                    if not (0 <= val < dims[dim_index]):
                        raise ValueError(
                            f"Discrete action out-of-bounds for {self.fmu_labels[fmu_idx]}, "
                            f"dimension {dim_index}: {val}"
                        )
                start = end
            return

        if np.any(action < -1.0) or np.any(action > 1.0):
            raise ValueError(f"Continuous action out-of-bounds, action = {action}")

    def _should_print_step_info(self, step_index: int) -> bool:
        """Control whether verbose step information is printed."""
        return self.step_info_print_interval > 0 and (step_index % self.step_info_print_interval == 0)

    def _print_step_info(self, action, unscaled_actions, raw_obs_vals, next_obs, reward, control_mode):
        log_rl("--------------------------------------------------", tag="Step Info")
        log_rl(f"Step: {self.current_step}", tag="Step Info")
        log_rl(f"Control Mode: {control_mode}", tag="Step Info")
        log_rl(f"Global Action (raw input): {action}", tag="Step Info")
        log_rl(f"Global Action (physical values): {unscaled_actions}", tag="Step Info")
        offset = self._time_feature_dims()
        for i in range(self.num_fmus):
            out_len = len(self.output_names[i])
            fmux_raw_output = raw_obs_vals[offset : offset + out_len]
            offset += out_len
            log_fmu(
                f"{self.fmu_labels[i]} observation outputs ({self.output_names[i]}): {fmux_raw_output}",
                tag="Step Info",
            )
            log_fmu(
                f"{self.fmu_labels[i]} physical action: {unscaled_actions[i]}",
                tag="Step Info",
            )
        obs_type = "normalized" if self.normalize_observation else "raw"
        log_rl(f"Next state ({obs_type}): {next_obs}", tag="Step Info")
        log_rl(f"Reward: {reward}", tag="Step Info")

    # ------------------------------------------------------------------
    # Action scaling ----------------------------------------------------
    # ------------------------------------------------------------------
    def unscale_action(self, action):
        """Convert agent output → *physical* set-points.

        Continuous case performs linear interpolation + rounding to nearest
        `interval`.  Discrete case simply looks up via bin index.
        """
        results_list = []
        if self.action_type == "discrete":
            start = 0
            for fmu_index in range(self.num_fmus):
                length = len(self.input_dims_list[fmu_index])
                sub_action = action[start : start + length]
                intervals_array = self.intervals_list[fmu_index]
                base_minimum_array = self.base_mins_list[fmu_index]
                temp_actions = []
                for action_index, discrete_index in enumerate(sub_action):
                    offset_value = base_minimum_array[action_index] + intervals_array[action_index] * discrete_index
                    temp_actions.append(offset_value)
                results_list.append(temp_actions)
                start += length
        else: # continuous
            start = 0
            for fmu_index in range(self.num_fmus):
                length = len(self.input_dims_list[fmu_index])
                sub_action = action[start : start + length]
                intervals_array = self.intervals_list[fmu_index]
                base_minimum_array = self.base_mins_list[fmu_index]
                base_maximum_array = self.base_maxs_list[fmu_index]
                temp_actions = []
                for dim_index, raw_value in enumerate(sub_action):
                    scaled_01 = (raw_value + 1) / 2
                    minimum_val = base_minimum_array[dim_index]
                    maximum_val = base_maximum_array[dim_index]
                    range_val = maximum_val - minimum_val
                    real_val = minimum_val + range_val * scaled_01
                    act_size = intervals_array[dim_index]
                    # Snap on a grid anchored at the minimum bound, then clamp
                    # back into [min, max] to avoid floating-point overshoot.
                    snapped_val = minimum_val + round((real_val - minimum_val) / act_size) * act_size
                    rounded_val = min(maximum_val, max(minimum_val, snapped_val))
                    temp_actions.append(rounded_val)
                results_list.append(temp_actions)
                start += length
        return results_list

    # ------------------------------------------------------------------
    # Observation helpers ----------------------------------------------
    # ------------------------------------------------------------------
    def record_outputs(self, step_index):
        """Log FMU outputs at the current step."""
        for fmu_index, fmu_object in enumerate(self.fmus):
            output_variable_names = self.output_names[fmu_index]
            values_got = fmu_object.get(output_variable_names)
            for column_name, column_value in zip(output_variable_names, values_got):
                self.record_dataframes[fmu_index].loc[step_index, column_name] = column_value

    def _time_feature_dims(self) -> int:
        """Return the number of environment-provided time features."""
        dims = 0
        if self.include_hour:
            dims += 2
        if self.include_day_of_year:
            dims += 2
        if self.include_episode_progress:
            dims += 1
        return dims

    def get_observation(self, target_time):
        """Assemble *unnormalised* observation array."""
        total_observation_length = self._time_feature_dims()
        for outputs in self.output_names:
            total_observation_length += len(outputs)
        observation_array = np.zeros(total_observation_length, dtype=np.float32)

        index_offset = 0
        if self.include_hour:
            hour_in_day = int((target_time - self.start_time_seconds) // 3600) % 24
            angle = 2.0 * np.pi * (hour_in_day / 24.0)
            observation_array[index_offset] = np.sin(angle)
            observation_array[index_offset + 1] = np.cos(angle)
            index_offset += 2
        if self.include_day_of_year:
            day_of_year = (int(target_time // 86_400) % 365) + 1
            day_angle = 2.0 * np.pi * ((day_of_year - 1) / 365.0)
            observation_array[index_offset] = np.sin(day_angle)
            observation_array[index_offset + 1] = np.cos(day_angle)
            index_offset += 2
        if self.include_episode_progress:
            elapsed = float(target_time - self.start_time_seconds)
            duration = float(max(self.stop_time_seconds - self.start_time_seconds, 1))
            episode_progress = min(max(elapsed / duration, 0.0), 1.0)
            observation_array[index_offset] = episode_progress
            index_offset += 1

        # FMU outputs ----------------------------------------------
        for fmu_index, record_dataframe in enumerate(self.record_dataframes):
            output_count = len(self.output_names[fmu_index])
            if (not record_dataframe.empty) and (self.current_step in record_dataframe.index):
                observation_array[
                index_offset: index_offset + output_count
                ] = record_dataframe.loc[self.current_step, self.output_names[fmu_index]].values
            else:
                observation_array[index_offset: index_offset + output_count] = 0.0
            index_offset += output_count
        return observation_array

    def scale_observation(self, observation_array):
        """Normalise observations to the [0, 1] interval."""
        scaled_result = (observation_array - self.obs_low) / (self.obs_high - self.obs_low)
        return np.clip(scaled_result, 0.0, 1.0)

    def format_observation(self, observation_array):
        """Return observations according to the normalization switch."""
        if self.normalize_observation:
            return self.scale_observation(observation_array)
        return observation_array.astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence -------------------------------------------------------
    # ------------------------------------------------------------------
    def save_fmu_data(self, output_folder=None):
        """Write recorded FMU data to Excel files (one file per FMU)."""
        if output_folder is None:
            output_folder = self.output_folder
        os.makedirs(output_folder, exist_ok=True)
        for i, df in enumerate(self.record_dataframes):
            file_path = os.path.join(output_folder, f"fmu_{i + 1}_data.xlsx")
            df.to_excel(file_path, index_label="Step")
            log_fmu(f"Saved {self.fmu_labels[i]} data to: {file_path}", tag="Save")

    def save_reward_data(self, output_folder=None):
        """Write stepwise reward history to an Excel file."""
        if output_folder is None:
            output_folder = self.output_folder
        os.makedirs(output_folder, exist_ok=True)
        df_rewards = pd.DataFrame(self.reward_list)
        file_path = os.path.join(output_folder, "rewards.xlsx")
        df_rewards.to_excel(file_path, index=False)
        log_rl(f"Saved reward data to: {file_path}", tag="Save")

    # ------------------------------------------------------------------
    # Cleanup -----------------------------------------------------------
    # ------------------------------------------------------------------
    def close(self):
        """Terminate FMUs and optionally persist simulation data."""
        log_rl("Terminating all FMUs...", tag="Close")
        for fmu_index, fmu_object in enumerate(self.fmus):
            try:
                fmu_object.terminate()
                log_fmu(f"{self.fmu_labels[fmu_index]} terminated successfully.", tag="Close")
            except Exception as e:
                log_rl(f"{self.fmu_labels[fmu_index]} termination failed: {e}", tag="Close")
        log_rl("All FMUs have been processed.", tag="Close")
        if self.save_results:
            self.save_fmu_data()
            self.save_reward_data()
