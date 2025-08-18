"""Gymnasium environment wrapping multiple building FMUs (Functional Mock‑up Units)"""

import numpy as np
import pandas as pd
import os
import datetime
from pyfmi import load_fmu
import gymnasium as gym
from gymnasium import spaces
from src.config import load_fmu_config
from algo.default_reward import compute_reward as default_compute_reward
from algo.custom_reward import compute_reward as custom_compute_reward

# -----------------------------------------------------------------------------
# Colour‑helper print functions -------------------------------------------------
# -----------------------------------------------------------------------------

def blue_print(*args, **kwargs):
    print("\033[31m", *args, "\033[0m", **kwargs)

def green_print(*args, **kwargs):
    print("\033[32m", *args, "\033[0m", **kwargs)

def yellow_print(*args, **kwargs):
    print("\033[33m", *args, "\033[0m", **kwargs)

def magenta_print(*args, **kwargs):
    print("\033[35m", *args, "\033[0m", **kwargs)

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
        }
    return io_defs

IO_DEFINITIONS = _load_io_definitions()

# -----------------------------------------------------------------------------
# Gymnasium environment --------------------------------------------------------
# -----------------------------------------------------------------------------
class MuFlex(gym.Env):
    """MuFlex - A Physics-based Platform for Multi-Building Flexibility Analysis and Coordination

    The environment instantiates one FMU per building and steps them in lock‑step
    every `step_size` seconds.  An agent receives *normalised observations* and
    returns either continuous or discrete actions that are mapped to physical
    set‑points.

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
    max_total_hvac_power : float, default=140_000.0
        Cluster‑level power cap [W].  Exceeding it incurs a quadratic penalty.
    reward_mode : {'default', 'custom'}, default='default'
        Select the built-in reward or the user-defined function in
        ``algo/custom_reward.py``.
    save_results : bool, default=False
        Persist per‑step I/O and reward traces to ``./simulation_data_<timestamp>/``.
    """

    # ---------------------------------------------------------------------
    # Construction ---------------------------------------------------------
    # ---------------------------------------------------------------------

    def __init__(
        self,
        fmu_configs,
        sim_days: int = 1,
        start_date: int = 1,
        step_size: int = 900,
        log_level: int = 7,
        action_type: str = "continuous",
        max_total_hvac_power: float = 140_000.0,
        hvac_weight: float = 0.5,
        temp_weight: float = 0.1,
        max_power_weight: float = 2.0,
        reward_mode: str = "default",
        save_results: bool = False,
        include_hour: bool = True,
    ):
        # Set up FMUs, simulation horizon and reward parameters -------------------------------------------------
        self.fmu_configs = fmu_configs
        self.num_fmus = len(fmu_configs)
        self.sim_days = sim_days
        self.start_date = start_date
        self.step_size = step_size
        self.log_level = log_level
        self.action_type = action_type.lower()
        self.max_total_hvac_power = max_total_hvac_power
        self.save_results = save_results
        self.include_hour = include_hour
        self.reward_list = []
        self.hvac_weight = hvac_weight
        self.temp_weight = temp_weight
        self.max_power_weight = max_power_weight
        self.reward_mode = reward_mode
        import types
        if reward_mode == "custom":
            reward_function = custom_compute_reward
        else:
            reward_function = default_compute_reward
        self.compute_reward = types.MethodType(reward_function, self)

        # Folder for optional result export --------------------------------------
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_folder = f"simulation_data_{now}"
        if self.save_results:
            blue_print("Saving results is enabled.")

        # ------------------------------------------------------------------
        # Time axis construction
        # ------------------------------------------------------------------
        self.start_time_seconds = 86_400 * (self.start_date - 1)
        self.stop_time_seconds = self.start_time_seconds + self.step_size * 4 * 24 * self.sim_days
        # Current formula assumes 900 s (15 min) steps via (4 * 24 * sim_days).
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
        self.record_dataframes = []   # mirrors per‑FMU I/O for later export
        self.input_dims_list = []
        self.intervals_list = []
        self.base_mins_list = []
        self.base_maxs_list = []

        for fmu_index, one_config in enumerate(self.fmu_configs):
            fmu_path = one_config["path"]
            io_type_name = one_config["io_type"]

            # --- Load and initialise the FMU --------------------------------
            blue_print(f"[Init] Loading FMU #{fmu_index + 1}: io_type = {io_type_name}, path = {fmu_path}")
            fmu_object = load_fmu(fmu_path, kind="cs", log_level=self.log_level)
            blue_print(
                f"[Init] Initializing FMU #{fmu_index + 1}, time range = "
                f"({self.start_time_seconds}, {self.stop_time_seconds})"
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
        # Last‑step component penalties stored for logging.
        self.last_hvac_penalty = 0.0
        self.last_temp_penalty = 0.0
        self.last_maxpower_penalty = 0.0

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
        """Construct normalised observation space [0,1]."""
        total_output_dimensions = sum(len(outputs) for outputs in self.output_names)

        # Observation layout:
        #   if include_hour: index 0 -> hour of day in [0, 23]
        #   following indices -> concatenated raw FMU outputs (per FMU order)
        #
        # Note: There is no binary tail element; all features are either hour (integer in [0,23])
        # or continuous outputs. Bounds come from IO_DEFINITIONS.

        observation_dimension = total_output_dimensions + (1 if self.include_hour else 0)
        self.obs_low = np.zeros(observation_dimension, dtype=np.float32)
        self.obs_high = np.ones(observation_dimension, dtype=np.float32)
        self.obs_high[0] = 23 # hour ∈ [0, 23]

        index_offset = 0
        if self.include_hour:
            self.obs_high[0] = 23  # hour ∈ [0, 23]
            index_offset = 1
        for one_config in self.fmu_configs:
            io_type_name = one_config["io_type"]
            local_low = IO_DEFINITIONS[io_type_name]["ob_base_low"]
            local_high = IO_DEFINITIONS[io_type_name]["ob_base_high"]
            length = len(local_low)
            self.obs_low[index_offset : index_offset + length] = local_low
            self.obs_high[index_offset : index_offset + length] = local_high
            index_offset += length

        # Final element is binary; bounds already [0,1].
        self.observation_space = spaces.Box(
            low=self.obs_low, high=self.obs_high, shape=(observation_dimension,), dtype=np.float32
        )

    # ---------------------------------------------------------------------
    # Standard Gym methods -------------------------------------------------
    # ---------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """ Reset environment – Internal logs are cleared but FMUs retain their state across
        episodes. TODO: Extend this method if FMU re-initialisation is required."""
        super().reset(seed=seed)
        self.current_step = 0
        self.done = False
        self.truncated = False

        # Clear in‑memory DataFrames.
        for record_dataframe in self.record_dataframes:
            record_dataframe.drop(record_dataframe.index, inplace=True)

        raw_observation = self.get_observation(self.time_steps[self.current_step])
        obs = self.scale_observation(raw_observation)
        return obs, {}

    def step(self, action):
        """Execute one simulation step.

        Returns a 5‑tuple `(obs, reward, done, truncated, info)` consistent with
        Gymnasium v0.29.
        """
        # ------------------------------------------------------------------
        # 0. Handle *after‑terminal* call (as per Gym contract) ------------
        # ------------------------------------------------------------------
        if self.done:
            last_observation = self.get_observation(self.time_steps[-1])
            return (
                self.scale_observation(last_observation),
                0.0,
                True,
                True,
                {"TimeLimit.truncated": True},
            )
        # ------------------------------------------------------------------
        # 1. Action validation ---------------------------------------------
        # ------------------------------------------------------------------
        if self.action_type == "discrete":
            start = 0
            for fmu_idx in range(self.num_fmus):
                dims = self.input_dims_list[fmu_idx]
                end = start + len(dims)
                sub_action = action[start:end]
                for dim_index, val in enumerate(sub_action):
                    if not (0 <= val < dims[dim_index]):
                        raise ValueError(
                            f"Discrete action out-of-bounds for FMU {fmu_idx}, "
                            f"dimension {dim_index}: {val}"
                        )
                start = end
        else:
            if np.any(action < -1.0) or np.any(action > 1.0):
                raise ValueError(f"Continuous action out-of-bounds, action = {action}")

        # ------------------------------------------------------------------
        # 2. Map actions → physical set‑points ------------------------------
        # ------------------------------------------------------------------
        unscaled_actions = self.unscale_action(action)

        # ------------------------------------------------------------------
        # 3. Apply to FMUs --------------------------------------------------
        # ------------------------------------------------------------------
        current_time = self.time_steps[self.current_step]
        for fmu_index, fmu_object in enumerate(self.fmus):
            input_var_names = self.input_names[fmu_index]
            input_values = unscaled_actions[fmu_index]
            for name_item, value_item in zip(input_var_names, input_values):
                fmu_object.set(name_item, value_item)
            # Cache for debugging/export.
            self.record_dataframes[fmu_index].loc[self.current_step, input_var_names] = input_values
        # Perform co‑simulation step (Co‑simulation FMUs prefer *do_step*).
        for fmu_object in self.fmus:
            fmu_object.do_step(current_t=current_time, step_size=self.step_size, new_step=True)

        # ------------------------------------------------------------------
        # 4. Collect outputs & compute reward ------------------------------
        # ------------------------------------------------------------------
        self.record_outputs(self.current_step)
        raw_next_observation = self.get_observation(current_time + self.step_size)
        next_obs = self.scale_observation(raw_next_observation)
        reward_value = self.compute_reward(next_obs)

        # Track diagnostics for potential offline inspection.
        self.reward_list.append(
            {
                "Step": self.current_step,
                "TotalReward": reward_value,
                "HVAC_penalty": self.last_hvac_penalty,
                "Temperature_penalty": self.last_temp_penalty,
                "MaxPower_penalty": self.last_maxpower_penalty,
            }
        )

        # Optional verbose printout at selected steps.
        if self.current_step in [1,2,3,4,5]:
            self._print_step_info(action, unscaled_actions, raw_next_observation, next_obs, reward_value)

        # ------------------------------------------------------------------
        # 5. Advance global index & flag termination -----------------------
        # ------------------------------------------------------------------
        self.current_step += 1
        if self.current_step >= self.num_steps:
            self.done = True
            self.truncated = True
        else:
            self.truncated = False
        return next_obs, reward_value, self.done, self.truncated, {"TimeLimit.truncated": self.truncated}

    # ---------------------------------------------------------------------
    # Helper / utility methods -------------------------------------------
    # ---------------------------------------------------------------------
    def _print_step_info(self, action, unscaled_actions, raw_obs_vals, next_obs, reward):
        """Pretty console dump for debugging at *select* steps."""
        green_print("--------------------------------------------------")
        green_print(f"[Step Info] Step: {self.current_step}")
        green_print(f"  Global Action (raw input): {action}")
        green_print(f"  Global Action (physical values): {unscaled_actions}")
        offset = 1
        for i in range(self.num_fmus):
            out_len = len(self.output_names[i])
            fmux_raw_output = raw_obs_vals[offset : offset + out_len]
            offset += out_len
            green_print(f"  FMU {i + 1} raw output: {fmux_raw_output}")
            green_print(f"  FMU {i + 1} physical action: {unscaled_actions[i]}")
        green_print(f"  Next state (normalized): {next_obs}")
        green_print(f"  Reward: {reward}")

    # ------------------------------------------------------------------
    # Action scaling ----------------------------------------------------
    # ------------------------------------------------------------------
    def unscale_action(self, action):
        """Convert agent output → *physical* set‑points.

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
                    rounded_val = round(real_val / act_size) * act_size
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

    def get_observation(self, target_time):
        """Assemble *unnormalised* observation array for a given hour-of-the-day."""
        total_observation_length = (1 if self.include_hour else 0)
        for outputs in self.output_names:
            total_observation_length += len(outputs)
        observation_array = np.zeros(total_observation_length, dtype=np.float32)

        index_offset = 0
        if self.include_hour:
            hour_in_day = int((target_time - self.start_time_seconds) // 3600) % 24
            observation_array[0] = hour_in_day
            index_offset = 1

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
            blue_print(f"Saved FMU {i + 1} data to: {file_path}")

    def save_reward_data(self, output_folder=None):
        """Write stepwise reward history to an Excel file."""
        if output_folder is None:
            output_folder = self.output_folder
        os.makedirs(output_folder, exist_ok=True)
        df_rewards = pd.DataFrame(self.reward_list)
        file_path = os.path.join(output_folder, "rewards.xlsx")
        df_rewards.to_excel(file_path, index=False)
        blue_print(f"Saved reward data to: {file_path}")

    # ------------------------------------------------------------------
    # Cleanup -----------------------------------------------------------
    # ------------------------------------------------------------------
    def close(self):
        """Terminate FMUs and optionally persist simulation data."""
        blue_print("[Close] Terminating all FMUs...")
        for fmu_index, fmu_object in enumerate(self.fmus):
            try:
                fmu_object.terminate()
                blue_print(f"[Close] FMU #{fmu_index + 1} terminated successfully.")
            except Exception as e:
                blue_print(f"[Close] FMU #{fmu_index + 1} termination failed: {e}")
        blue_print("[Close] All FMUs have been processed.")
        if self.save_results:
            self.save_fmu_data()
            self.save_reward_data()

