"""Reward calculation for MuFlex environments."""

def green_print(*args, **kwargs):
    """Print helper that highlights debug lines in green."""
    print("\033[32m", *args, "\033[0m", **kwargs)


def compute_reward(self, scaled_observation):
    """Default reward from *normalised* observation.

    Components
    ----------
    hvac_penalty_value : float
        Linear w.r.t. total HVAC power (normalised to power cap).
    temperature_penalty_value : float
        Sum of squared temperature deviations outside comfort band during
        occupied hours.
    max_power_penalty_value : float
        Quadratic penalty if cap exceeded; small bonus (‑0.5) for staying
        below cap during DR window.

    Notes
    -----
    The individual penalties are weighted using coefficients stored on the
    environment instance (``self.hvac_weight`` etc.).
    """
    hvac_penalty_value = 0.0
    temperature_penalty_value = 0.0
    max_power_penalty_value = 0.0
    raw_observation = scaled_observation * (self.obs_high - self.obs_low) + self.obs_low

    # Derive current wall‑clock hour/minute.
    current_time_val = (
        self.time_steps[self.current_step + 1] if self.current_step + 1 < self.num_steps else self.time_steps[-1]
    )
    time_of_day = (current_time_val - self.start_time_seconds) % (24 * 3600)
    hour_now = time_of_day // 3600
    minute_now = (time_of_day % 3600) // 60

    # Occupied & DR windows -------------------------------------------
    is_work_time = (hour_now > 8 or (hour_now == 8 and minute_now >= 15)) and (
        hour_now < 18 or (hour_now == 18 and minute_now == 0)
    )
    grid_service_time = (hour_now > 7 or (hour_now == 7 and minute_now >= 15)) and (
        hour_now < 18 or (hour_now == 18 and minute_now == 0)
    )

    # ------------------------------------------------------------------
    # 1. Aggregate HVAC power -----------------------------------------
    # ------------------------------------------------------------------
    index_offset = 1
    total_hvac_power_actual = 0.0
    for fmu_index, cfg in enumerate(self.fmu_configs):
        outputs = self.output_names[fmu_index]
        out_len = len(outputs)
        fmu_raw_vals = raw_observation[index_offset : index_offset + out_len]
        index_offset += out_len
        io_type = cfg["io_type"]
        if io_type == "OfficeS":
            coil_idx = outputs.index("coilPower")
            fan_idx = outputs.index("fanPower")
            total_hvac_power_actual += fmu_raw_vals[coil_idx] + fmu_raw_vals[fan_idx]
        else:
            coil_bot_idx = outputs.index("coilPower_bot")
            coil_mid_idx = outputs.index("coilPower_mid")
            coil_top_idx = outputs.index("coilPower_top")
            fan_bot_idx = outputs.index("fanPower_bot")
            fan_mid_idx = outputs.index("fanPower_mid")
            fan_top_idx = outputs.index("fanPower_top")
            total_hvac_power_actual += (
                fmu_raw_vals[coil_bot_idx]
                + fmu_raw_vals[coil_mid_idx]
                + fmu_raw_vals[coil_top_idx]
                + fmu_raw_vals[fan_bot_idx]
                + fmu_raw_vals[fan_mid_idx]
                + fmu_raw_vals[fan_top_idx]
            )
    hvac_penalty_value = total_hvac_power_actual / self.max_total_hvac_power

    # ------------------------------------------------------------------
    # 2. Thermal comfort ----------------------------------------------
    # ------------------------------------------------------------------
    if is_work_time:
        index_offset_for_temp = 1
        for fmu_index, cfg in enumerate(self.fmu_configs):
            outputs = self.output_names[fmu_index]
            out_len = len(outputs)
            fmu_raw_vals = raw_observation[index_offset_for_temp : index_offset_for_temp + out_len]
            index_offset_for_temp += out_len
            temp_low, temp_high = self.temp_interval[fmu_index]
            dev_sum = 0.0
            zone_list = (
                ["z5Temp", "z1Temp", "z3Temp", "z2Temp", "z4Temp"]
                if cfg["io_type"] == "OfficeS"
                else ["BotMeanTemp", "MidMeanTemp", "TopMeanTemp"]
            )
            for z_name in zone_list:
                if z_name in outputs:
                    z_idx = outputs.index(z_name)
                    real_temp = fmu_raw_vals[z_idx]
                    dev = 0.0
                    if real_temp < temp_low:
                        dev = temp_low - real_temp
                    elif real_temp > temp_high:
                        dev = real_temp - temp_high
                    if dev > 1.0:
                        dev = dev * dev  # quadratic beyond ±1°C
                    dev_sum += dev
            # Scale by factor to balance terms.
            temperature_penalty_value += (dev_sum * 100) / 10.0

    # ------------------------------------------------------------------
    # 3. Power limit penalty/bonus ---------------------------------------
    # ------------------------------------------------------------------
    if total_hvac_power_actual > self.max_total_hvac_power:
        max_power_penalty_value = (1 * hvac_penalty_value) ** 2
    else:
        max_power_penalty_value = -0.5 if grid_service_time else 0.0

    # Expose components for logging.
    self.last_hvac_penalty = hvac_penalty_value
    self.last_temp_penalty = temperature_penalty_value
    self.last_maxpower_penalty = max_power_penalty_value

    # Weighted sum → *negative reward* (penalties).
    total_penalty = (
        self.hvac_weight * hvac_penalty_value
        + self.temp_weight * temperature_penalty_value
        + self.max_power_weight * max_power_penalty_value
    )
    final_reward = -total_penalty
    if self.current_step in [32]:
        green_print(f"[Reward] HVAC penalty: {hvac_penalty_value}")
        green_print(f"[Reward] Temperature penalty: {temperature_penalty_value}")
        green_print(f"[Reward] Max power penalty: {max_power_penalty_value}")
        green_print(f"[Reward] Total penalty: {total_penalty}")
        green_print(f"[Reward] Final Reward: {final_reward}")
    return final_reward