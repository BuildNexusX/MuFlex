"""Graphical User Interface for configuring and running MuFlex simulations."""
from __future__ import annotations

import importlib.util
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.ttk as ttk
from PIL import Image, ImageTk

import ast
import pprint

from algo.baseline import run_baseline
from src.config import load_fmu_config
from src.env import IO_DEFINITIONS

def parse_env_code(code: str) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Extract ``fmu_configs`` and MuFlex parameters from an env snippet.

    The snippet should define ``fmu_configs`` and create an ``env`` via
    ``MuFlex(...)``.  Only literal assignments are understood; the helper is
    aimed at quickly re‑loading configurations saved from the GUI.
    """

    tree = ast.parse(code)
    fmu_configs: list[dict[str, str]] = []
    params: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
            if target == "fmu_configs":
                fmu_configs = ast.literal_eval(node.value)
            elif target == "env" and isinstance(node.value, ast.Call):
                for kw in node.value.keywords:
                    if kw.arg == "fmu_configs":
                        continue
                    params[kw.arg] = ast.literal_eval(kw.value)
    return fmu_configs, params

class MuFlexGUI(tk.Tk):
    """Main window used to configure FMUs and launch simulations."""
    def __init__(self) -> None:
        """Build all widgets and default parameters."""
        super().__init__()
        self.title("MuFlex Simulator")
        self.geometry("720x1000")
        self.resizable(False, False)
        bg_path = Path(__file__).parent / "figs" / "GUI.jpg"
        self._bg_image = None
        if bg_path.exists():
            self._bg_image = ImageTk.PhotoImage(Image.open(bg_path))
            bg_label = tk.Label(self, image=self._bg_image)
            bg_label.place(relwidth=1, relheight=1)
            bg_label.lower()
        self.configure(bg="white")

        # Use a notebook with three tabs: create, run, and manage environments
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill="both")

        self.create_tab = tk.Frame(self.notebook, bg="white")
        self.run_tab = tk.Frame(self.notebook, bg="white")
        self.manage_tab = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.create_tab, text="Create Env")
        self.notebook.add(self.run_tab, text="Run Baseline")
        self.notebook.add(self.manage_tab, text="Manage Envs")

        if self._bg_image is not None:
            for tab in (self.create_tab, self.run_tab, self.manage_tab):
                tab_bg = tk.Label(tab, image=self._bg_image)
                tab_bg.place(relwidth=1, relheight=1)
                tab_bg.lower()

        content = tk.Frame(self.create_tab, bg="white")
        content.place(relx=0.5, rely=0.5, anchor="center")

        # ====== FMU Configuration ======
        self.fmu_defs = load_fmu_config() # definitions loaded from JSON file
        self.fmu_types = list(self.fmu_defs.keys())
        fmu_frame = tk.LabelFrame(content, text="FMU Configuration", bg="white", labelanchor="n")
        fmu_frame.grid(row=0, column=0, padx=5, pady=5)
        self.type_count_vars: dict[str, tk.IntVar] = {}
        self.type_frames: dict[str, tk.Frame] = {}
        self.type_path_vars: dict[str, list[tk.StringVar]] = {}
        self.type_spinboxes: dict[str, tk.Spinbox] = {}
        self.confirmed_fmu_configs: list[dict[str, str]] = []
        self.input_names: list[str] = []
        self.output_names: list[str] = []
        self.discrete_dims: list[int] = []
        self.fmu_confirmed = False
        row = 0
        for t in self.fmu_types:
            tk.Label(fmu_frame, text=t, bg="white").grid(row=row, column=0, sticky="w")
            var = tk.IntVar(value=0)
            self.type_count_vars[t] = var
            spin = tk.Spinbox(
                fmu_frame,
                from_=0,
                to=10,
                width=5,
                textvariable=var,
                command=self.update_fmu_entries,
            )
            spin.grid(row=row, column=1, sticky="w")
            self.type_spinboxes[t] = spin
            frame = tk.Frame(fmu_frame, bg="white")
            frame.grid(row=row + 1, column=0, columnspan=3, sticky="w")
            self.type_frames[t] = frame
            self.type_path_vars[t] = []
            row += 2

        fmu_frame.grid_columnconfigure(0, weight=1)
        fmu_frame.grid_columnconfigure(1, weight=1)
        self.fmu_confirm_btn = tk.Button(
            fmu_frame, text="Confirm", command=self.confirm_fmus, bg="white", width=10
        )
        self.fmu_confirm_btn.grid(row=row, column=0, pady=2, sticky="e")
        self.fmu_reset_btn = tk.Button(
            fmu_frame,
            text="Reset",
            command=self.reset_fmus,
            bg="white",
            state="disabled",
            width=10,
        )
        self.fmu_reset_btn.grid(row=row, column=1, pady=2, sticky="w")

        # ====== Simulation Parameters ======
        sim_frame = tk.LabelFrame(content, text="Simulation Parameters", bg="white", labelanchor="n")
        sim_frame.grid(row=1, column=0, padx=5, pady=5)

        self.sim_days_var = tk.IntVar(value=1)
        self.start_date_var = tk.IntVar(value=201)
        self.step_size_var = tk.IntVar(value=900)
        self.sim_confirmed = False

        tk.Label(sim_frame, text="sim_days", bg="white").grid(row=0, column=0, sticky="w")
        tk.Entry(sim_frame, textvariable=self.sim_days_var, width=10).grid(row=0, column=1)
        tk.Label(sim_frame, text="start_date", bg="white").grid(row=1, column=0, sticky="w")
        tk.Entry(sim_frame, textvariable=self.start_date_var, width=10).grid(row=1, column=1)
        tk.Label(sim_frame, text="step_size", bg="white").grid(row=2, column=0, sticky="w")
        tk.Entry(sim_frame, textvariable=self.step_size_var, width=10).grid(row=2, column=1)
        sim_frame.grid_columnconfigure(0, weight=1)
        sim_frame.grid_columnconfigure(1, weight=1)
        self.sim_confirm_btn = tk.Button(sim_frame, text="Confirm", command=self.confirm_sim, bg="white", width=10)
        self.sim_confirm_btn.grid(row=3, column=0, pady=2, sticky="e")
        self.sim_reset_btn = tk.Button(
            sim_frame, text="Reset", command=self.reset_sim, bg="white", state="disabled", width=10
        )
        self.sim_reset_btn.grid(row=3, column=1, pady=2, sticky="w")

        # ====== Spaces ======
        self.space_frame = tk.LabelFrame(content, text="Action/Observation Space Preview", bg="white", labelanchor="n")
        self.space_frame.grid(row=2, column=0, padx=5, pady=5)
        self.action_type_var = tk.StringVar(value="continuous")
        tk.Label(self.space_frame, text="Action type", bg="white").grid(row=0, column=0, sticky="w")
        self.action_type_menu = tk.OptionMenu(self.space_frame, self.action_type_var, "continuous", "discrete")
        self.action_type_menu.grid(row=0, column=1, sticky="w")
        self.include_hour_var = tk.BooleanVar(value=True)
        self.include_hour_cb = tk.Checkbutton(
            self.space_frame,
            text="Include hour of day (observation)",
            variable=self.include_hour_var,
            bg="white",
        )
        self.include_hour_cb.grid(row=1, column=0, columnspan=2, sticky="w")
        self.space_confirmed = False
        self.action_dim_label = tk.Label(self.space_frame, text="", bg="white", justify="left")
        self.action_dim_label.grid(row=2, column=0, columnspan=2, sticky="w")
        self.obs_dim_label = tk.Label(self.space_frame, text="", bg="white", justify="left")
        self.obs_dim_label.grid(row=3, column=0, columnspan=2, sticky="w")
        self.inputs_combo = None
        self.outputs_combo = None
        self.space_frame.grid_columnconfigure(0, weight=1)
        self.space_frame.grid_columnconfigure(1, weight=1)
        self.space_confirm_btn = tk.Button(
            self.space_frame, text="Confirm", command=self.confirm_spaces, bg="white", width=10
        )
        self.space_confirm_btn.grid(row=6, column=0, pady=2, sticky="e")
        self.space_reset_btn = tk.Button(
            self.space_frame, text="Reset", command=self.reset_spaces, bg="white", state="disabled", width=10
        )
        self.space_reset_btn.grid(row=6, column=1, pady=2, sticky="w")

        # ====== Reward Function ======
        reward_frame = tk.LabelFrame(content, text="Reward Function", bg="white", labelanchor="n")
        reward_frame.grid(row=3, column=0, padx=5, pady=5)
        self.reward_confirmed = False

        self.reward_mode = tk.StringVar(value="default")
        tk.Radiobutton(
            reward_frame, text="Default", variable=self.reward_mode,
            value="default", command=self.update_reward_mode, bg="white",
        ).grid(row=0, column=0)
        tk.Radiobutton(
            reward_frame, text="Custom", variable=self.reward_mode,
            value="custom", command=self.update_reward_mode, bg="white",
        ).grid(row=0, column=1)

        self.default_reward_frame = tk.Frame(reward_frame, bg="white")
        self.default_reward_frame.grid(row=1, column=0, columnspan=2, pady=2)
        self.max_power_var = tk.DoubleVar(value=103_500.0)
        self.hvac_w_var = tk.DoubleVar(value=0.5)
        self.temp_w_var = tk.DoubleVar(value=0.1)
        self.maxpow_w_var = tk.DoubleVar(value=2.0)
        tk.Label(self.default_reward_frame, text="max_total_hvac_power", bg="white").grid(row=0, column=0)
        tk.Entry(self.default_reward_frame, textvariable=self.max_power_var, width=12).grid(row=0, column=1)
        tk.Label(self.default_reward_frame, text="hvac_weight", bg="white").grid(row=1, column=0)
        tk.Entry(self.default_reward_frame, textvariable=self.hvac_w_var, width=12).grid(row=1, column=1)
        tk.Label(self.default_reward_frame, text="temp_weight", bg="white").grid(row=2, column=0)
        tk.Entry(self.default_reward_frame, textvariable=self.temp_w_var, width=12).grid(row=2, column=1)
        tk.Label(self.default_reward_frame, text="max_power_weight", bg="white").grid(row=3, column=0)
        tk.Entry(self.default_reward_frame, textvariable=self.maxpow_w_var, width=12).grid(row=3, column=1)

        self.custom_reward_frame = tk.Frame(reward_frame, bg="white")
        tk.Label(
            self.custom_reward_frame,
            text="Edit algo/custom_reward.py to customize the reward",
            bg="white",
        ).grid(row=0, column=0)
        self.custom_reward_frame.grid_remove()
        reward_frame.grid_columnconfigure(0, weight=1)
        reward_frame.grid_columnconfigure(1, weight=1)
        self.reward_confirm_btn = tk.Button(
            reward_frame, text="Confirm", command=self.confirm_reward, bg="white", width=10
        )
        self.reward_confirm_btn.grid(row=3, column=0, pady=2, sticky="e")
        self.reward_reset_btn = tk.Button(
            reward_frame, text="Reset", command=self.reset_reward, bg="white", state="disabled", width=10
        )
        self.reward_reset_btn.grid(row=3, column=1, pady=2, sticky="w")

        # Environment name and create button ---------------------------------
        env_name_frame = tk.Frame(content, bg="white")
        env_name_frame.grid(row=4, column=0, pady=5)
        tk.Label(env_name_frame, text="Env name", bg="white").grid(row=0, column=0, padx=(0, 5))
        self.env_name_var = tk.StringVar()
        tk.Entry(env_name_frame, textvariable=self.env_name_var, width=10).grid(row=0, column=1, padx=(0, 5))
        self.create_env_btn = tk.Button(
            env_name_frame, text="Create Env", command=self.create_env, bg="white", state="disabled"
        )
        self.create_env_btn.grid(row=0, column=2)

        # ========================= Run Baseline Tab ==========================
        run_frame = tk.Frame(self.run_tab, bg="white")
        run_frame.place(relx=0.5, rely=0.5, anchor="center")

        self.env_options = self.load_env_options()
        option_labels = [lbl for lbl, _ in self.env_options]
        values = option_labels or ["No environments available"]
        self.selected_env_var = tk.StringVar(value=values[0])
        tk.Label(run_frame, text="Environment", bg="white").grid(row=0, column=0, sticky="w")
        self.env_menu = tk.OptionMenu(run_frame, self.selected_env_var, *values, command=self.on_env_select)
        if not option_labels:
            self.env_menu.config(state="disabled")
        self.env_menu.grid(row=0, column=1, sticky="w")

        self.save_results_var = tk.BooleanVar(value=False)
        self.print_steps_var = tk.BooleanVar(value=False)
        tk.Checkbutton(run_frame, text="Save results", variable=self.save_results_var, bg="white").grid(row=1, column=0, sticky="w")
        tk.Checkbutton(run_frame, text="Print steps", variable=self.print_steps_var, bg="white").grid(row=1, column=1, sticky="w")

        # Baseline action configuration ---------------------------------
        self.action_frame = tk.LabelFrame(run_frame, text="Baseline Actions", bg="white", labelanchor="n")
        self.action_frame.grid(row=2, column=0, columnspan=2, pady=5)
        tk.Label(self.action_frame, text="FMU", bg="white").grid(row=0, column=0, sticky="w")
        self.building_var = tk.StringVar()
        self.building_menu = tk.OptionMenu(self.action_frame, self.building_var, "")
        self.building_menu.config(state="disabled")
        self.building_menu.grid(row=0, column=1, sticky="w")
        self.action_entry_frame = tk.Frame(self.action_frame, bg="white")
        self.action_entry_frame.grid(row=1, column=0, columnspan=2)
        self.actions_confirm_btn = tk.Button(
            self.action_frame, text="Confirm", command=self.confirm_actions, bg="white"
        )
        self.actions_confirm_btn.grid(row=2, column=0, columnspan=2, pady=2)

        tk.Button(run_frame, text="Run", command=self.run_selected_env, bg="white").grid(row=3, column=0, columnspan=2, pady=10)

        # storage for baseline actions
        self.current_fmu_configs: list[dict[str, str]] = []
        self.current_env_params: dict[str, object] = {}
        self.base_mins_list: list[list[float]] = []
        self.base_maxs_list: list[list[float]] = []
        self.action_vars: list[list[tk.StringVar]] = []
        self.building_labels: list[str] = []
        self.physical_actions: list[list[float]] = []
        self.actions_confirmed = False

        if option_labels:
            self.on_env_select(values[0])

        # ========================= Manage Envs Tab ==========================
        manage_frame = tk.Frame(self.manage_tab, bg="white")
        manage_frame.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(manage_frame, text="Saved Environments", bg="white").pack(anchor="w")
        self.env_listbox = tk.Listbox(manage_frame, bg="white", height=8)
        self.env_listbox.pack(fill="both", expand=True)
        tk.Button(manage_frame, text="Delete Selected", command=self.delete_selected_env, bg="white").pack(pady=5)

        self.refresh_env_listbox()

    def browse_file(self, desc: str, pattern: str) -> str:
        """Generic file dialog helper."""
        return filedialog.askopenfilename(filetypes=[(desc, pattern), ("All files", "*.*")])

    def browse_fmu(self, var: tk.StringVar) -> None:
        """Prompt for an FMU file and store the path in *var*."""
        path = self.browse_file("FMU", "*.fmu")
        if path:
            var.set(path)

    def update_fmu_entries(self) -> None:
        """Render FMU path widgets based on user‑selected counts."""
        for t in self.fmu_types:
            frame = self.type_frames[t]
            for widget in frame.winfo_children():
                widget.destroy()
            self.type_path_vars[t].clear()
            for i in range(self.type_count_vars[t].get()):
                var = tk.StringVar()
                self.type_path_vars[t].append(var)
                tk.Label(frame, text=f"{t} {i + 1}", bg="white").grid(row=i, column=0, sticky="w")
                tk.Entry(frame, textvariable=var, width=40).grid(row=i, column=1)
                tk.Button(frame, text="Browse", command=lambda v=var: self.browse_fmu(v), bg="white").grid(row=i, column=2)
    def update_reward_mode(self) -> None:
        """Switch between built‑in and custom reward configurations."""
        if self.reward_mode.get() == "default":
            self.custom_reward_frame.grid_remove()
            self.default_reward_frame.grid()
        else:
            self.default_reward_frame.grid_remove()
            self.custom_reward_frame.grid(row=1, column=0, columnspan=2, pady=2)

    def confirm_fmus(self) -> None:
        """Lock FMU configuration and display space information."""
        try:
            fmu_configs = []
            input_names: list[str] = []
            output_names: list[str] = []
            discrete_dims: list[int] = []
            for t in self.fmu_types:
                self.type_spinboxes[t].config(state="disabled")
                frame = self.type_frames[t]
                for widget in frame.winfo_children():
                    widget.config(state="disabled")
                for var in self.type_path_vars[t]:
                    path = var.get()
                    if not path:
                        raise ValueError(f"Missing FMU path for {t}")
                    fmu_configs.append({"path": path, "io_type": t})
                    input_names.extend(IO_DEFINITIONS[t]["INPUTS"])
                    output_names.extend(IO_DEFINITIONS[t]["OUTPUTS"])
                    discrete_dims.extend(IO_DEFINITIONS[t]["dims"])
            if not fmu_configs:
                self.fmu_reset_btn.config(state="normal")
                raise ValueError("No FMUs configured")
            self.confirmed_fmu_configs = fmu_configs
            self.input_names = input_names
            self.output_names = output_names
            self.discrete_dims = discrete_dims
            self.fmu_confirmed = True
            self.fmu_confirm_btn.config(state="disabled")
            self.fmu_reset_btn.config(state="normal")
            self.update_create_env_state()
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def reset_fmus(self) -> None:
        """Re‑enable FMU configuration widgets."""
        self.confirmed_fmu_configs = []
        self.input_names = []
        self.output_names = []
        self.discrete_dims = []
        for t in self.fmu_types:
            self.type_spinboxes[t].config(state="normal")
            frame = self.type_frames[t]
            for widget in frame.winfo_children():
                widget.config(state="normal")
        self.fmu_confirmed = False
        self.fmu_confirm_btn.config(state="normal")
        self.fmu_reset_btn.config(state="disabled")
        self.action_dim_label.config(text="")
        self.obs_dim_label.config(text="")
        if self.inputs_combo:
            self.inputs_combo.destroy()
            self.inputs_combo = None
        if self.outputs_combo:
            self.outputs_combo.destroy()
            self.outputs_combo = None
        self.space_confirmed = False
        self.action_type_menu.config(state="normal")
        self.include_hour_cb.config(state="normal")
        self.space_confirm_btn.config(state="normal")
        self.space_reset_btn.config(state="disabled")
        self.update_create_env_state()

    def confirm_sim(self) -> None:
        """Lock simulation parameter inputs."""
        self.sim_confirmed = True
        self.sim_confirm_btn.config(state="disabled")
        self.sim_reset_btn.config(state="normal")
        for widget in self.sim_confirm_btn.master.grid_slaves():
            if isinstance(widget, tk.Entry):
                widget.config(state="disabled")
        self.update_create_env_state()

    def reset_sim(self) -> None:
        """Unlock simulation parameter inputs."""
        self.sim_confirmed = False
        self.sim_confirm_btn.config(state="normal")
        self.sim_reset_btn.config(state="disabled")
        for widget in self.sim_confirm_btn.master.grid_slaves():
            if isinstance(widget, tk.Entry):
                widget.config(state="normal")
        self.update_create_env_state()

    def confirm_spaces(self) -> None:
        """Lock space settings and show preview."""
        try:
            if not self.fmu_confirmed:
                raise ValueError("FMUs not confirmed")
            self.action_type_menu.config(state="disabled")
            self.include_hour_cb.config(state="disabled")
            if self.action_type_var.get() == "continuous":
                action_dim = len(self.input_names)
                self.action_dim_label.config(text=f"Action dim: {action_dim}")
            else:
                action_dim = len(self.discrete_dims)
                self.action_dim_label.config(
                    text=f"Action dim: {action_dim} ({self.discrete_dims})"
                )
            obs_dim = len(self.output_names) + (
                1 if self.include_hour_var.get() else 0
            )
            self.obs_dim_label.config(text=f"Observation dim: {obs_dim}")
            if self.inputs_combo:
                self.inputs_combo.destroy()
            if self.outputs_combo:
                self.outputs_combo.destroy()
            self.inputs_var = tk.StringVar(value="Inputs")
            self.inputs_combo = ttk.Combobox(
                self.space_frame, textvariable=self.inputs_var, values=self.input_names, state="readonly"
            )
            self.inputs_combo.grid(row=4, column=0, columnspan=2, sticky="we")
            outputs = self.output_names[:]
            if self.include_hour_var.get():
                outputs.append("hour_of_day")
            self.outputs_var = tk.StringVar(value="Outputs")
            self.outputs_combo = ttk.Combobox(
                self.space_frame, textvariable=self.outputs_var, values=outputs, state="readonly"
            )
            self.outputs_combo.grid(row=5, column=0, columnspan=2, sticky="we")
            self.space_confirmed = True
            self.space_confirm_btn.config(state="disabled")
            self.space_reset_btn.config(state="normal")
            self.update_create_env_state()
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def reset_spaces(self) -> None:
        """Unlock space settings."""
        self.space_confirmed = False
        self.action_type_menu.config(state="normal")
        self.include_hour_cb.config(state="normal")
        self.action_dim_label.config(text="")
        self.obs_dim_label.config(text="")
        if self.inputs_combo:
            self.inputs_combo.destroy()
            self.inputs_combo = None
        if self.outputs_combo:
            self.outputs_combo.destroy()
            self.outputs_combo = None
        self.space_confirm_btn.config(state="normal")
        self.space_reset_btn.config(state="disabled")
        self.update_create_env_state()

    def confirm_reward(self) -> None:
        """Lock reward configuration."""
        self.reward_confirm_btn.config(state="disabled")
        self.reward_reset_btn.config(state="normal")
        for widget in self.default_reward_frame.winfo_children():
            widget.config(state="disabled")
        for widget in self.custom_reward_frame.winfo_children():
            widget.config(state="disabled")
        for widget in self.reward_confirm_btn.master.grid_slaves(row=0):
            widget.config(state="disabled")
        self.reward_confirmed = True
        self.update_create_env_state()

    def reset_reward(self) -> None:
        """Unlock reward configuration."""
        self.reward_confirmed = False
        self.reward_confirm_btn.config(state="normal")
        self.reward_reset_btn.config(state="disabled")
        for widget in self.default_reward_frame.winfo_children():
            widget.config(state="normal")
        for widget in self.custom_reward_frame.winfo_children():
            widget.config(state="normal")
        for widget in self.reward_confirm_btn.master.grid_slaves(row=0):
            widget.config(state="normal")
        self.update_reward_mode()
        self.update_create_env_state()

    def update_create_env_state(self) -> None:
        """Enable create-env button when all sections are confirmed."""
        if all(
            [
                self.fmu_confirmed,
                self.sim_confirmed,
                self.space_confirmed,
                self.reward_confirmed,
            ]
        ):
            self.create_env_btn.config(state="normal")
        else:
            self.create_env_btn.config(state="disabled")

    def create_env(self) -> None:
        """Save environment configuration to file with a custom name."""
        try:
            if not all(
                [
                    self.fmu_confirmed,
                    self.sim_confirmed,
                    self.space_confirmed,
                    self.reward_confirmed,
                ]
            ):
                raise ValueError("All sections must be confirmed")

            env_name = self.env_name_var.get().strip()
            if not (1 <= len(env_name) <= 5 and env_name.isalpha()):
                raise ValueError("Env name must be 1-5 English letters")

            sim_days = self.sim_days_var.get()
            start_date = self.start_date_var.get()
            step_size = self.step_size_var.get()
            action_type = self.action_type_var.get()
            include_hour = self.include_hour_var.get()
            reward_mode = self.reward_mode.get()
            code = (
                "from src.env import MuFlex\n"
                f"fmu_configs = {pprint.pformat(self.confirmed_fmu_configs)}\n"
                f"env = MuFlex(fmu_configs=fmu_configs, sim_days={sim_days}, start_date={start_date}, step_size={step_size}, action_type='{action_type}', include_hour={include_hour}, reward_mode='{reward_mode}')"
            )

            env_file = Path(__file__).parent / "src" / "env_list.txt"
            idx = len(self.env_options) + 1
            with open(env_file, "a", encoding="utf-8") as fh:
                fh.write(f"# {idx} {env_name}\n{code}\n---\n")

            messagebox.showinfo("Env Saved", f"Environment '{env_name}' saved to src/env_list.txt")
            self.env_name_var.set("")
            self.refresh_env_menu()
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def load_env_options(self) -> list[tuple[str, str]]:
        """Parse env_list.txt into a list of (name, code) tuples."""
        env_file = Path(__file__).parent / "src" / "env_list.txt"
        if not env_file.exists():
            return []
        envs: list[tuple[str, str]] = []
        current_name: str | None = None
        code_lines: list[str] = []
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                parts = line[2:].split(maxsplit=1)
                if parts and parts[0].isdigit():
                    if current_name and code_lines:
                        envs.append((current_name, "\n".join(code_lines).strip()))
                    current_name = parts[1].strip() if len(parts) > 1 else f"Env {parts[0]}"
                    code_lines = []
                continue
            if line.strip() == "---":
                if current_name and code_lines:
                    envs.append((current_name, "\n".join(code_lines).strip()))
                current_name = None
                code_lines = []
                continue
            if current_name is not None:
                code_lines.append(line)
        if current_name and code_lines:
            envs.append((current_name, "\n".join(code_lines).strip()))
        return envs

    def refresh_env_menu(self) -> None:
        """Reload environments from file and update the dropdown."""
        self.env_options = self.load_env_options()
        option_labels = [lbl for lbl, _ in self.env_options]
        menu = self.env_menu["menu"]
        menu.delete(0, "end")

        if option_labels:
            for label in option_labels:
                menu.add_command(
                    label=label,
                    command=tk._setit(self.selected_env_var, label, self.on_env_select),
                )
            self.selected_env_var.set(option_labels[0])
            self.env_menu.config(state="normal")
            self.on_env_select(option_labels[0])
            self.selected_env_var.set(option_labels[0])
            self.env_menu.config(state="normal")
        else:
            placeholder = "No environments available"
            menu.add_command(label=placeholder, command=lambda: None)
            self.selected_env_var.set(placeholder)
            self.env_menu.config(state="disabled")
        if hasattr(self, "env_listbox"):
            self.refresh_env_listbox()

    def refresh_env_listbox(self) -> None:
        """Update the manage tab listbox with current environments."""
        self.env_listbox.delete(0, tk.END)
        for label, _ in self.env_options:
            self.env_listbox.insert(tk.END, label)

    def delete_selected_env(self) -> None:
        """Remove the selected environment from env_list.txt."""
        selection = self.env_listbox.curselection()
        if not selection:
            return
        name = self.env_listbox.get(selection[0])
        if not messagebox.askyesno("Confirm Delete", f"Delete environment '{name}'?"):
            return
        env_file = Path(__file__).parent / "src" / "env_list.txt"
        if not env_file.exists():
            return
        lines = env_file.read_text(encoding="utf-8").splitlines()
        header: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("# ") and line[2:].split(maxsplit=1)[0].isdigit():
                break
            header.append(line)
            i += 1
        entries: list[tuple[str, list[str]]] = []
        current_name: str | None = None
        code_lines: list[str] = []
        for line in lines[i:]:
            if line.startswith("# "):
                parts = line[2:].split(maxsplit=1)
                if parts and parts[0].isdigit():
                    if current_name is not None:
                        entries.append((current_name, code_lines))
                    current_name = parts[1].strip() if len(parts) > 1 else f"Env {parts[0]}"
                    code_lines = []
                continue
            if line.strip() == "---":
                if current_name is not None:
                    entries.append((current_name, code_lines))
                current_name = None
                code_lines = []
                continue
            if current_name is not None:
                code_lines.append(line)
        entries = [e for e in entries if e[0] != name]
        new_lines = header
        for idx, (n, codes) in enumerate(entries, start=1):
            new_lines.append(f"# {idx} {n}")
            new_lines.extend(codes)
            new_lines.append("---")
        env_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        self.refresh_env_menu()

    # ------------------ Baseline action helpers ------------------
    def clear_action_entries(self) -> None:
        for w in self.action_entry_frame.winfo_children():
            w.destroy()

    def on_building_select(self, label: str) -> None:
        if label not in self.building_labels:
            return
        idx = self.building_labels.index(label)
        self.building_var.set(label)
        self.clear_action_entries()
        io_type = self.current_fmu_configs[idx]["io_type"]
        inputs = IO_DEFINITIONS[io_type]["INPUTS"]
        mins = self.base_mins_list[idx]
        maxs = self.base_maxs_list[idx]
        vars = self.action_vars[idx]
        for j, (name, var, mn, mx) in enumerate(zip(inputs, vars, mins, maxs)):
            tk.Label(self.action_entry_frame, text=name, bg="white").grid(row=j, column=0, sticky="w")
            tk.Entry(self.action_entry_frame, textvariable=var, width=10).grid(row=j, column=1)
            tk.Label(self.action_entry_frame, text=f"[{mn}, {mx}]", bg="white").grid(row=j, column=2, sticky="w")

    def on_env_select(self, _label: str) -> None:
        try:
            if not self.env_options:
                return
            label = self.selected_env_var.get()
            idx = [lbl for lbl, _ in self.env_options].index(label)
            code = self.env_options[idx][1]

            fmu_configs, params = parse_env_code(code)
            if not fmu_configs:

                raise ValueError("Invalid environment code")
            self.current_fmu_configs = fmu_configs
            self.current_env_params = {
                "sim_days": params.get("sim_days", 1),
                "start_date": params.get("start_date", 1),
                "step_size": params.get("step_size", 900),
                "reward_mode": params.get("reward_mode", "default"),
                "action_type": params.get("action_type", "continuous"),
                "include_hour": params.get("include_hour", True),
            }


            self.base_mins_list = []
            self.base_maxs_list = []
            self.action_vars = []
            self.building_labels = []
            counts: dict[str, int] = {}
            for cfg in fmu_configs:
                io_type = cfg["io_type"]
                counts[io_type] = counts.get(io_type, 0) + 1
                self.building_labels.append(f"{io_type} {counts[io_type]}")
                self.base_mins_list.append(IO_DEFINITIONS[io_type]["base_mins"])
                self.base_maxs_list.append(IO_DEFINITIONS[io_type]["base_maxs"])
                self.action_vars.append([
                    tk.StringVar() for _ in IO_DEFINITIONS[io_type]["INPUTS"]
                ])
            menu = self.building_menu["menu"]
            menu.delete(0, "end")
            if self.building_labels:
                for lbl in self.building_labels:
                    menu.add_command(label=lbl, command=lambda v=lbl: self.on_building_select(v))
                self.building_var.set(self.building_labels[0])
                self.on_building_select(self.building_labels[0])
                self.building_menu.config(state="normal")
            else:
                menu.add_command(label="", command=lambda: None)
                self.building_var.set("")
                self.building_menu.config(state="disabled")
                self.clear_action_entries()
            self.physical_actions = []
            self.actions_confirmed = False
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def confirm_actions(self) -> None:
        try:
            self.physical_actions = []
            for idx, vars in enumerate(self.action_vars):
                mins = self.base_mins_list[idx]
                maxs = self.base_maxs_list[idx]
                vals: list[float] = []
                for name, var, mn, mx in zip(
                    IO_DEFINITIONS[self.current_fmu_configs[idx]["io_type"]]["INPUTS"],
                    vars,
                    mins,
                    maxs,
                ):
                    v_str = var.get()
                    if not v_str:
                        raise ValueError(f"Missing value for {self.building_labels[idx]} {name}")
                    v = float(v_str)
                    if v < mn or v > mx:
                        raise ValueError(
                            f"Value {v} out of bounds [{mn}, {mx}] for {self.building_labels[idx]} {name}"
                        )
                    vals.append(v)
                self.physical_actions.append(vals)
            if not self.physical_actions:
                raise ValueError("No actions defined")
            self.actions_confirmed = True
            messagebox.showinfo("Confirmed", "Baseline actions set")
        except Exception as exc:
            self.actions_confirmed = False
            messagebox.showerror("Error", str(exc))

    def run_selected_env(self) -> None:
        """Run baseline controller for the selected environment."""
        try:
            if not self.env_options:
                raise ValueError("No environments available")
            if not self.actions_confirmed:
                raise ValueError("Baseline actions not confirmed")

            params = dict(
                fmu_configs=self.current_fmu_configs,
                sim_days=self.current_env_params.get("sim_days", 1),
                start_date=self.current_env_params.get("start_date", 1),
                step_size=self.current_env_params.get("step_size", 900),
                max_total_hvac_power=self.max_power_var.get(),
                reward_weights=(
                    self.hvac_w_var.get(),
                    self.temp_w_var.get(),
                    self.maxpow_w_var.get(),
                ),
                reward_mode=self.current_env_params.get("reward_mode", "default"),
                save_results=self.save_results_var.get(),
                print_steps=self.print_steps_var.get(),
                action_type=self.current_env_params.get("action_type", "continuous"),
                include_hour=self.current_env_params.get("include_hour", True),
                physical_actions=self.physical_actions,
            )
            thread = threading.Thread(target=run_baseline, kwargs=params, daemon=True)
            thread.start()
        except Exception as exc:
            messagebox.showerror("Error", str(exc))


if __name__ == "__main__":
    app = MuFlexGUI()
    # app.minsize(720, 600)
    app.mainloop()
