"""
Graphical User Interface for configuring and running MuFlex simulations.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from functools import partial
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.ttk as ttk
from PIL import Image, ImageTk

import ast
import pprint
import re

from algo.baseline import run_baseline
from src.config import load_fmu_config
from src.env import IO_DEFINITIONS
from src.reward_registry import list_available_reward_modes, reward_script_path


def parse_env_code(code: str) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Extract ``fmu_configs`` and MuFlex parameters from an env snippet.

    The snippet should define ``fmu_configs`` and create an ``env`` via
    ``MuFlex(...)``. Only literal assignments are understood; the helper is
    aimed at quickly re-loading configurations saved from the GUI.
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


class ScrollableFrame(ttk.Frame):
    """A vertically scrollable frame implemented with Canvas + Scrollbar."""

    def __init__(self, parent: tk.Widget, *, bg: str = "white", bg_image: ImageTk.PhotoImage | None = None):
        super().__init__(parent)

        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, bg=bg)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self._bg_image = bg_image
        self._bg_id = None
        if self._bg_image is not None:
            self._bg_id = self.canvas.create_image(0, 0, image=self._bg_image, anchor="nw")

        # The interior frame that holds content.
        self.interior = tk.Frame(self.canvas, bg=bg)
        self._window_id = self.canvas.create_window((0, 0), window=self.interior, anchor="n")

        # Ensure background stays behind interior content
        if self._bg_id is not None:
            self.canvas.tag_lower(self._bg_id)

        # Update scroll region when interior changes
        self.interior.bind("<Configure>", self._on_interior_configure)
        # Make interior width follow canvas width
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Mouse wheel scrolling: bind only while cursor is over this widget
        self.interior.bind("<Enter>", self._bind_mousewheel)
        self.interior.bind("<Leave>", self._unbind_mousewheel)

    def _on_interior_configure(self, _event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        side_margin = 120
        inner_w = max(200, event.width - 2 * side_margin)

        self.canvas.itemconfig(self._window_id, width=inner_w)
        self.canvas.coords(self._window_id, event.width // 2, 0)

        if self._bg_id is not None:
            img_w = self._bg_image.width()
            img_h = self._bg_image.height()
            x = (event.width - img_w) // 2
            y = (event.height - img_h) // 2
            self.canvas.coords(self._bg_id, x, y)
            self.canvas.tag_lower(self._bg_id)

    def _bind_mousewheel(self, _event):
        # Windows / macOS
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        # Linux
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _unbind_mousewheel(self, _event):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        # event.delta: Windows usually ±120 per notch; macOS different scale
        step = int(-1 * (event.delta / 120))
        self.canvas.yview_scroll(step, "units")

    def _on_mousewheel_linux(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")


class MuFlexGUI(tk.Tk):
    """Main window used to configure FMUs and launch simulations."""

    def __init__(self) -> None:
        """Build all widgets and default parameters."""
        super().__init__()
        self.title("MuFlex Simulator")
        self.center_window(800, 900)
        self.resizable(False, False)

        # Load background image once (optional)
        bg_path = Path(__file__).parent / "figs" / "GUI.jpg"
        self._bg_image = None
        if bg_path.exists():
            try:
                self._bg_image = ImageTk.PhotoImage(Image.open(bg_path))
            except Exception:
                self._bg_image = None

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

        # Make each tab scrollable.
        self.create_scroll = ScrollableFrame(self.create_tab, bg="white", bg_image=self._bg_image)
        self.create_scroll.pack(fill="both", expand=True)
        self.create_body = self.create_scroll.interior

        self.run_scroll = ScrollableFrame(self.run_tab, bg="white", bg_image=self._bg_image)
        self.run_scroll.pack(fill="both", expand=True)
        self.run_body = self.run_scroll.interior

        self.manage_scroll = ScrollableFrame(self.manage_tab, bg="white", bg_image=self._bg_image)
        self.manage_scroll.pack(fill="both", expand=True)
        self.manage_body = self.manage_scroll.interior
        # ----------------------------------------------------------------------
        def _scroll_active_tab_to_top(_evt=None):
            tab_id = self.notebook.select()  # returns widget name string
            if tab_id == str(self.create_tab):
                self.create_scroll.canvas.yview_moveto(0)
            elif tab_id == str(self.run_tab):
                self.run_scroll.canvas.yview_moveto(0)
            elif tab_id == str(self.manage_tab):
                self.manage_scroll.canvas.yview_moveto(0)

        self.notebook.bind("<<NotebookTabChanged>>", _scroll_active_tab_to_top)
        self.after_idle(_scroll_active_tab_to_top)

        # ========================= Create Env Tab =========================
        content = tk.Frame(self.create_body, bg="white")
        content.pack(padx=10, pady=10, anchor="n")
        content.grid_columnconfigure(0, weight=1)

        # ====== FMU Configuration ======
        self.fmu_defs = load_fmu_config()  # definitions loaded from JSON file
        self.fmu_types = list(self.fmu_defs.keys())
        self.building_types = [t for t in self.fmu_types if self.fmu_defs[t].get("category", "building") != "pv"]
        self.pv_types = [t for t in self.fmu_types if self.fmu_defs[t].get("category", "building") == "pv"]
        self.type_count_vars: dict[str, tk.IntVar] = {}
        self.type_frames: dict[str, tk.Frame] = {}
        self.type_path_vars: dict[str, list[tk.StringVar]] = {}
        self.type_spinboxes: dict[str, tk.Spinbox] = {}
        self.confirmed_fmu_configs: list[dict[str, str]] = []
        self.input_names: list[str] = []
        self.output_names: list[str] = []
        self.discrete_dims: list[int] = []
        self.fmu_confirmed = False

        self.fmu_section, self.fmu_content, self.fmu_collapsed_label = self._build_section(
            content,
            "Model Configuration",
            "Configuration set. Click Reset to edit.",
            row=0,
        )

        building_frame = tk.LabelFrame(self.fmu_content, text="Building Configuration", bg="white", labelanchor="n")
        building_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        pv_frame = tk.LabelFrame(self.fmu_content, text="PV Configuration", bg="white", labelanchor="n")
        pv_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        self._build_type_section(building_frame, self.building_types)
        self._build_type_section(pv_frame, self.pv_types)

        building_frame.grid_columnconfigure(0, weight=1)
        building_frame.grid_columnconfigure(1, weight=1)
        pv_frame.grid_columnconfigure(0, weight=1)
        pv_frame.grid_columnconfigure(1, weight=1)

        self.fmu_controls, self.fmu_confirm_btn, self.fmu_reset_btn = self._build_controls(
            self.fmu_section, self.confirm_fmus, self.reset_fmus, row=1
        )

        # ====== Simulation Parameters ======
        self.sim_section, self.sim_content, self.sim_collapsed_label = self._build_section(
            content,
            "Simulation Parameters",
            "Simulation parameters set. Click Reset to edit.",
            row=1,
        )

        self.sim_days_var = tk.IntVar(value=1)
        self.start_date_var = tk.IntVar(value=201)
        self.step_size_var = tk.IntVar(value=900)
        self.sim_confirmed = False

        sim_fields = [
            ("sim_days", self.sim_days_var),
            ("start_date", self.start_date_var),
            ("step_size", self.step_size_var),
        ]
        for idx, (label, var) in enumerate(sim_fields):
            tk.Label(self.sim_content, text=label, bg="white").grid(row=idx, column=0, sticky="e", padx=(0, 5))
            tk.Entry(self.sim_content, textvariable=var, width=10, justify="center").grid(row=idx, column=1, pady=1)

        self.sim_content.grid_columnconfigure(0, weight=1)
        self.sim_content.grid_columnconfigure(1, weight=1)

        self.sim_controls, self.sim_confirm_btn, self.sim_reset_btn = self._build_controls(
            self.sim_section, self.confirm_sim, self.reset_sim, row=1
        )

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

        self.include_day_of_year_var = tk.BooleanVar(value=True)
        self.include_day_of_year_cb = tk.Checkbutton(
            self.space_frame,
            text="Include day of year (observation)",
            variable=self.include_day_of_year_var,
            bg="white",
        )
        self.include_day_of_year_cb.grid(row=2, column=0, columnspan=2, sticky="w")

        self.include_episode_progress_var = tk.BooleanVar(value=True)
        self.include_episode_progress_cb = tk.Checkbutton(
            self.space_frame,
            text="Include episode progress (observation)",
            variable=self.include_episode_progress_var,
            bg="white",
        )
        self.include_episode_progress_cb.grid(row=3, column=0, columnspan=2, sticky="w")

        self.normalize_observation_var = tk.BooleanVar(value=True)
        self.normalize_observation_cb = tk.Checkbutton(
            self.space_frame,
            text="Normalize observation",
            variable=self.normalize_observation_var,
            bg="white",
        )
        self.normalize_observation_cb.grid(row=4, column=0, columnspan=2, sticky="w")

        self.space_confirmed = False
        self.action_dim_label = tk.Label(self.space_frame, text="", bg="white", justify="left")
        self.action_dim_label.grid(row=5, column=0, columnspan=2, sticky="w")
        self.obs_dim_label = tk.Label(self.space_frame, text="", bg="white", justify="left")
        self.obs_dim_label.grid(row=6, column=0, columnspan=2, sticky="w")
        self.inputs_combo = None
        self.outputs_combo = None

        self.space_frame.grid_columnconfigure(0, weight=1, uniform="space_ctrl")
        self.space_frame.grid_columnconfigure(1, weight=1, uniform="space_ctrl")

        self.space_confirm_btn = tk.Button(
            self.space_frame, text="Confirm", command=self.confirm_spaces, bg="white", width=10
        )
        self.space_confirm_btn.grid(row=9, column=0, pady=2)

        self.space_reset_btn = tk.Button(
            self.space_frame, text="Reset", command=self.reset_spaces, bg="white", state="disabled", width=10
        )
        self.space_reset_btn.grid(row=9, column=1, pady=2)

        # ====== Reward Function ======
        self.reward_section = tk.LabelFrame(content, text="Reward Function", bg="white", labelanchor="n")
        self.reward_section.grid(row=3, column=0, padx=5, pady=5)

        self.reward_collapsed_label = tk.Label(
            self.reward_section,
            text="Reward function set. Click Reset to edit.",
            bg="white",
        )
        self.reward_collapsed_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.reward_collapsed_label.grid_remove()

        self.reward_content = tk.Frame(self.reward_section, bg="white")
        self.reward_content.grid(row=0, column=0, sticky="we")

        self.reward_confirmed = False
        self.reward_modes = list_available_reward_modes()
        if not self.reward_modes:
            self.reward_modes = ["example_reward"]
        default_mode = "example_reward" if "example_reward" in self.reward_modes else self.reward_modes[0]
        self.reward_mode = tk.StringVar(value=default_mode)
        self.reward_mode_buttons: list[tk.Radiobutton] = []

        for col, mode in enumerate(self.reward_modes):
            label = mode.replace("_", " ").title()
            radio = tk.Radiobutton(
                self.reward_content,
                text=label,
                variable=self.reward_mode,
                value=mode,
                command=self.update_reward_mode,
                bg="white",
            )
            radio.grid(row=0, column=col, sticky="w")
            self.reward_mode_buttons.append(radio)

        self.default_reward_frame = tk.Frame(self.reward_content, bg="white")
        self.default_reward_frame.grid(row=1, column=0, columnspan=2, pady=2)

        self.custom_reward_frame = tk.Frame(self.reward_content, bg="white")
        self.reward_source_label = tk.Label(
            self.custom_reward_frame,
            text="",
            bg="white",
            justify="left",
            anchor="w",
        )
        self.reward_source_label.grid(row=0, column=0, sticky="w")
        self.custom_reward_frame.grid_remove()

        for col in range(max(len(self.reward_modes), 1)):
            self.reward_content.grid_columnconfigure(col, weight=1)

        self.reward_controls = tk.Frame(self.reward_section, bg="white")
        self.reward_controls.grid(row=1, column=0, padx=5, pady=(0, 5), sticky="we")
        self.reward_controls.grid_columnconfigure(0, weight=1, uniform="reward_ctrl")
        self.reward_controls.grid_columnconfigure(1, weight=1, uniform="reward_ctrl")

        self.reward_confirm_btn = tk.Button(
            self.reward_controls, text="Confirm", command=self.confirm_reward, bg="white", width=10
        )
        self.reward_confirm_btn.grid(row=0, column=0, pady=2)

        self.reward_reset_btn = tk.Button(
            self.reward_controls, text="Reset", command=self.reset_reward, bg="white", state="disabled", width=10
        )
        self.reward_reset_btn.grid(row=0, column=1, pady=2, padx=(5, 0))
        self.update_reward_mode()

        # Environment name and create button ---------------------------------
        env_name_frame = tk.Frame(content, bg="white")
        env_name_frame.grid(row=4, column=0, pady=5, sticky="we")
        for col in range(3):
            env_name_frame.grid_columnconfigure(col, weight=1)

        tk.Label(env_name_frame, text="Env name", bg="white").grid(row=0, column=0, padx=(0, 5), sticky="e")
        self.env_name_var = tk.StringVar()
        tk.Entry(env_name_frame, textvariable=self.env_name_var, width=10).grid(row=0, column=1, padx=(0, 5), sticky="we")
        self.create_env_btn = tk.Button(
            env_name_frame, text="Create Env", command=self.create_env, bg="white", state="disabled"
        )
        self.create_env_btn.grid(row=0, column=2, sticky="w")

        # ========================= Run Baseline Tab ==========================
        run_frame = tk.Frame(self.run_body, bg="white")
        run_frame.pack(padx=10, pady=10, anchor="n")

        for col in range(2):
            run_frame.grid_columnconfigure(col, weight=1)

        self.env_options = self.load_env_options()
        option_labels = [lbl for lbl, _ in self.env_options]
        values = option_labels or ["No environments available"]
        self.selected_env_var = tk.StringVar(value=values[0])

        tk.Label(run_frame, text="Environment", bg="white").grid(row=0, column=0, sticky="w")
        self.env_menu = tk.OptionMenu(run_frame, self.selected_env_var, *values, command=self.on_env_select)
        if not option_labels:
            self.env_menu.config(state="disabled")
        self.env_menu.grid(row=0, column=1, sticky="w")

        self.run_settings_frame = tk.LabelFrame(run_frame, text="Run Settings", bg="white", labelanchor="n")
        self.run_settings_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky="ew")

        self.save_results_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            self.run_settings_frame,
            text="Save results",
            variable=self.save_results_var,
            bg="white",
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        self.max_steps_var = tk.StringVar(value="")
        tk.Label(self.run_settings_frame, text="Max steps", bg="white").grid(row=1, column=0, sticky="e", padx=(0, 5))
        tk.Entry(self.run_settings_frame, textvariable=self.max_steps_var, width=10, justify="center").grid(row=1, column=1, sticky="w")

        self.rl_control_window_only_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            self.run_settings_frame,
            text="RL control window only",
            variable=self.rl_control_window_only_var,
            bg="white",
        ).grid(row=2, column=0, columnspan=2, sticky="w")

        self.office_hour_start_var = tk.StringVar(value="08:00")
        tk.Label(self.run_settings_frame, text="Office hour start", bg="white").grid(row=3, column=0, sticky="e", padx=(0, 5))
        tk.Entry(self.run_settings_frame, textvariable=self.office_hour_start_var, width=10, justify="center").grid(row=3, column=1, sticky="w")

        self.office_hour_end_var = tk.StringVar(value="18:00")
        tk.Label(self.run_settings_frame, text="Office hour end", bg="white").grid(row=4, column=0, sticky="e", padx=(0, 5))
        tk.Entry(self.run_settings_frame, textvariable=self.office_hour_end_var, width=10, justify="center").grid(row=4, column=1, sticky="w")

        self.step_info_print_interval_var = tk.StringVar(value="10")
        tk.Label(self.run_settings_frame, text="Step info print interval", bg="white").grid(row=5, column=0, sticky="e", padx=(0, 5))
        tk.Entry(
            self.run_settings_frame,
            textvariable=self.step_info_print_interval_var,
            width=10,
            justify="center",
        ).grid(row=5, column=1, sticky="w")

        self.run_settings_frame.grid_columnconfigure(0, weight=1)
        self.run_settings_frame.grid_columnconfigure(1, weight=1)

        # Baseline action configuration ---------------------------------
        self.building_action_frame = tk.LabelFrame(run_frame, text="Building Actions", bg="white", labelanchor="n")
        self.building_action_frame.grid(row=2, column=0, columnspan=2, pady=5, sticky="ew")
        tk.Label(self.building_action_frame, text="Building", bg="white").grid(row=0, column=0, sticky="w")
        self.building_var = tk.StringVar()
        self.building_menu = tk.OptionMenu(self.building_action_frame, self.building_var, "")
        self.building_menu.config(state="disabled")
        self.building_menu.grid(row=0, column=1, sticky="w")
        self.building_action_entry_frame = tk.Frame(self.building_action_frame, bg="white")
        self.building_action_entry_frame.grid(row=1, column=0, columnspan=2)

        self.pv_action_frame = tk.LabelFrame(run_frame, text="PV Actions", bg="white", labelanchor="n")
        self.pv_action_frame.grid(row=3, column=0, columnspan=2, pady=5, sticky="ew")
        tk.Label(self.pv_action_frame, text="PV", bg="white").grid(row=0, column=0, sticky="w")
        self.pv_var = tk.StringVar()
        self.pv_menu = tk.OptionMenu(self.pv_action_frame, self.pv_var, "")
        self.pv_menu.config(state="disabled")
        self.pv_menu.grid(row=0, column=1, sticky="w")
        self.pv_action_entry_frame = tk.Frame(self.pv_action_frame, bg="white")
        self.pv_action_entry_frame.grid(row=1, column=0, columnspan=2)

        self.actions_confirm_btn = tk.Button(
            self.pv_action_frame, text="Confirm", command=self.confirm_actions, bg="white"
        )
        self.actions_confirm_btn.grid(row=2, column=0, columnspan=2, pady=2)

        tk.Button(run_frame, text="Run", command=self.run_selected_env, bg="white").grid(
            row=4, column=0, columnspan=2, pady=10
        )

        # storage for baseline actions
        self.current_fmu_configs: list[dict[str, str]] = []
        self.current_env_params: dict[str, object] = {}
        self.action_entries: list[dict[str, object]] = []
        self.building_labels: list[str] = []
        self.pv_labels: list[str] = []
        self.physical_actions: list[list[float]] = []
        self.actions_confirmed = False

        if option_labels:
            self.on_env_select(values[0])

        # ========================= Manage Envs Tab ==========================
        manage_frame = tk.Frame(self.manage_body, bg="white")
        manage_frame.pack(padx=10, pady=10, anchor="n", fill="x")

        tk.Label(manage_frame, text="Saved Environments", bg="white").pack(anchor="w")
        self.env_listbox = tk.Listbox(manage_frame, bg="white", height=12)
        self.env_listbox.pack(fill="both", expand=True)
        tk.Button(manage_frame, text="Delete Selected", command=self.delete_selected_env, bg="white").pack(pady=5)

        self.refresh_env_listbox()

        def _scroll_all_to_top():
            for sf in (self.create_scroll, self.run_scroll, self.manage_scroll):
                sf.canvas.yview_moveto(0)

        self.after_idle(lambda: (self.center_window(), _scroll_all_to_top()))

    # ------------------ layout helpers ------------------
    def center_window(self, width: int | None = None, height: int | None = None) -> None:
        """Center the main window on the current display."""
        self.update_idletasks()
        current_width = width if width is not None else self.winfo_width()
        current_height = height if height is not None else self.winfo_height()
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        x = (screen_w - current_width) // 2
        y = (screen_h - current_height) // 2
        self.geometry(f"{current_width}x{current_height}+{x}+{y}")

    def _build_section(
        self, parent: tk.Widget, title: str, collapse_text: str, row: int
    ) -> tuple[tk.LabelFrame, tk.Frame, tk.Label]:
        """Create a centered label frame with collapsible content."""
        section = tk.LabelFrame(parent, text=title, bg="white", labelanchor="n")
        section.grid(row=row, column=0, padx=5, pady=5, sticky="ew")
        section.grid_columnconfigure(0, weight=1)
        collapsed = tk.Label(section, text=collapse_text, bg="white", anchor="center", justify="center")
        collapsed.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        collapsed.grid_remove()
        content = tk.Frame(section, bg="white")
        content.grid(row=0, column=0, sticky="ew")
        content.grid_columnconfigure(0, weight=1)
        return section, content, collapsed

    def _build_controls(
        self, section: tk.LabelFrame, confirm_cmd: Callable[[], None], reset_cmd: Callable[[], None], row: int
    ) -> tuple[tk.Frame, tk.Button, tk.Button]:
        """Create a centered Confirm/Reset control row."""
        controls = tk.Frame(section, bg="white")
        controls.grid(row=row, column=0, padx=5, pady=(0, 5), sticky="ew")
        for col in range(2):
            controls.grid_columnconfigure(col, weight=1, uniform=f"ctrl_{section}")
        confirm_btn = tk.Button(controls, text="Confirm", command=confirm_cmd, bg="white", width=10)
        confirm_btn.grid(row=0, column=0, pady=2)
        reset_btn = tk.Button(controls, text="Reset", command=reset_cmd, bg="white", state="disabled", width=10)
        reset_btn.grid(row=0, column=1, pady=2, padx=(5, 0))
        return controls, confirm_btn, reset_btn

    # ------------------ file browsing ------------------
    def browse_file(self, desc: str, pattern: str) -> str:
        """Generic file dialog helper."""
        return filedialog.askopenfilename(filetypes=[(desc, pattern), ("All files", "*.*")])

    def browse_fmu(self, var: tk.StringVar) -> None:
        """Prompt for an FMU file and store the path in *var*."""
        path = self.browse_file("FMU", "*.fmu")
        if path:
            var.set(path)

    # ------------------ IO display ------------------
    def show_io_info(self, io_type: str) -> None:
        """Display input/output names for the selected FMU type."""
        info = IO_DEFINITIONS.get(io_type)
        if not info:
            messagebox.showerror("Error", f"No IO definition found for {io_type}")
            return

        inputs = "\n".join(info.get("INPUTS", [])) or "-"
        outputs = "\n".join(info.get("OUTPUTS", [])) or "-"
        message = f"Inputs:\n{inputs}\n\nOutputs:\n{outputs}"
        messagebox.showinfo(f"{io_type} IO", message)

    # ------------------ FMU section ------------------
    def _build_type_section(self, frame: tk.LabelFrame, types: list[str]) -> None:
        """Render FMU selection widgets for a list of FMU *types*."""
        row = 0
        for t in types:
            tk.Label(frame, text=t, bg="white").grid(row=row, column=0, sticky="w")
            var = tk.IntVar(value=0)
            self.type_count_vars[t] = var
            spin = tk.Spinbox(
                frame,
                from_=0,
                to=30,
                width=5,
                textvariable=var,
                command=self.update_fmu_entries,
            )
            spin.grid(row=row, column=1, sticky="w")
            self.type_spinboxes[t] = spin
            type_frame = tk.Frame(frame, bg="white")
            type_frame.grid(row=row + 1, column=0, columnspan=3, sticky="w")
            self.type_frames[t] = type_frame
            self.type_path_vars[t] = []
            row += 2

    def update_fmu_entries(self) -> None:
        """Render FMU path widgets based on user-selected counts."""
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
                tk.Button(frame, text="Browse", command=lambda v=var: self.browse_fmu(v), bg="white").grid(
                    row=i, column=2, padx=5
                )
                tk.Button(frame, text="View IO", command=partial(self.show_io_info, t), bg="white").grid(
                    row=i, column=3, padx=5
                )

    def _collapse_section(self, content: tk.Widget, placeholder: tk.Widget) -> None:
        content.grid_remove()
        placeholder.grid()

    def _expand_section(self, content: tk.Widget, placeholder: tk.Widget) -> None:
        placeholder.grid_remove()
        content.grid()

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
            self._collapse_section(self.fmu_content, self.fmu_collapsed_label)
            self.update_create_env_state()
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def reset_fmus(self) -> None:
        """Re-enable FMU configuration widgets."""
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
        self.include_day_of_year_cb.config(state="normal")
        self.include_episode_progress_cb.config(state="normal")
        self.normalize_observation_cb.config(state="normal")
        self.space_confirm_btn.config(state="normal")
        self.space_reset_btn.config(state="disabled")
        self._expand_section(self.fmu_content, self.fmu_collapsed_label)
        self.update_create_env_state()

    # ------------------ simulation params ------------------
    def confirm_sim(self) -> None:
        """Lock simulation parameter inputs."""
        self.sim_confirmed = True
        self.sim_confirm_btn.config(state="disabled")
        self.sim_reset_btn.config(state="normal")
        for widget in self.sim_content.grid_slaves():
            if isinstance(widget, tk.Entry):
                widget.config(state="disabled")
        self._collapse_section(self.sim_content, self.sim_collapsed_label)
        self.update_create_env_state()

    def reset_sim(self) -> None:
        """Unlock simulation parameter inputs."""
        self.sim_confirmed = False
        self.sim_confirm_btn.config(state="normal")
        self.sim_reset_btn.config(state="disabled")
        for widget in self.sim_content.grid_slaves():
            if isinstance(widget, tk.Entry):
                widget.config(state="normal")
        self._expand_section(self.sim_content, self.sim_collapsed_label)
        self.update_create_env_state()

    # ------------------ spaces ------------------
    def _time_feature_names(self) -> list[str]:
        """Return environment-provided time feature names."""
        names: list[str] = []
        if self.include_hour_var.get():
            names.extend(["hour_sin", "hour_cos"])
        if self.include_day_of_year_var.get():
            names.extend(["day_of_year_sin", "day_of_year_cos"])
        if self.include_episode_progress_var.get():
            names.append("episode_progress")
        return names

    def confirm_spaces(self) -> None:
        """Lock space settings and show preview."""
        try:
            if not self.fmu_confirmed:
                raise ValueError("FMUs not confirmed")
            self.action_type_menu.config(state="disabled")
            self.include_hour_cb.config(state="disabled")
            self.include_day_of_year_cb.config(state="disabled")
            self.include_episode_progress_cb.config(state="disabled")
            self.normalize_observation_cb.config(state="disabled")

            if self.action_type_var.get() == "continuous":
                action_dim = len(self.input_names)
                self.action_dim_label.config(text=f"Action dim: {action_dim}")
            else:
                action_dim = len(self.discrete_dims)
                self.action_dim_label.config(text=f"Action dim: {action_dim} ({self.discrete_dims})")

            time_features = self._time_feature_names()
            obs_dim = len(self.output_names) + len(time_features)
            obs_type = "normalized" if self.normalize_observation_var.get() else "raw"
            self.obs_dim_label.config(text=f"Observation dim: {obs_dim} ({obs_type})")

            if self.inputs_combo:
                self.inputs_combo.destroy()
            if self.outputs_combo:
                self.outputs_combo.destroy()

            self.inputs_var = tk.StringVar(value="Inputs")
            self.inputs_combo = ttk.Combobox(
                self.space_frame, textvariable=self.inputs_var, values=self.input_names, state="readonly"
            )
            self.inputs_combo.grid(row=7, column=0, columnspan=2, sticky="we")

            outputs = time_features + self.output_names[:]
            self.outputs_var = tk.StringVar(value="Outputs")
            self.outputs_combo = ttk.Combobox(
                self.space_frame, textvariable=self.outputs_var, values=outputs, state="readonly"
            )
            self.outputs_combo.grid(row=8, column=0, columnspan=2, sticky="we")

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
        self.include_day_of_year_cb.config(state="normal")
        self.include_episode_progress_cb.config(state="normal")
        self.normalize_observation_cb.config(state="normal")
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

    # ------------------ reward ------------------
    def update_reward_mode(self) -> None:
        """Switch reward option details by selected mode."""
        if self.reward_mode.get() == "default":
            self.custom_reward_frame.grid_remove()
            self.default_reward_frame.grid()
        else:
            self.default_reward_frame.grid_remove()
            self.custom_reward_frame.grid(row=1, column=0, columnspan=2, pady=2)
            source = reward_script_path(self.reward_mode.get())
            source_text = str(source) if source else "(script path unavailable)"
            self.reward_source_label.config(text=f"Reward script:\n{source_text}")

    def confirm_reward(self) -> None:
        """Lock reward configuration."""
        self.reward_confirm_btn.config(state="disabled")
        self.reward_reset_btn.config(state="normal")
        for widget in self.default_reward_frame.winfo_children():
            widget.config(state="disabled")
        for widget in self.custom_reward_frame.winfo_children():
            widget.config(state="disabled")
        for widget in self.reward_mode_buttons:
            widget.config(state="disabled")
        self.reward_confirmed = True
        self._collapse_section(self.reward_content, self.reward_collapsed_label)
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
        for widget in self.reward_mode_buttons:
            widget.config(state="normal")
        self.update_reward_mode()
        self._expand_section(self.reward_content, self.reward_collapsed_label)
        self.update_create_env_state()

    # ------------------ create env enablement ------------------
    def update_create_env_state(self) -> None:
        """Enable create-env button when all sections are confirmed."""
        if all([self.fmu_confirmed, self.sim_confirmed, self.space_confirmed, self.reward_confirmed]):
            self.create_env_btn.config(state="normal")
        else:
            self.create_env_btn.config(state="disabled")

    # ------------------ env persistence ------------------
    def create_env(self) -> None:
        """Save environment configuration to file with a custom name."""
        try:
            if not all([self.fmu_confirmed, self.sim_confirmed, self.space_confirmed, self.reward_confirmed]):
                raise ValueError("All sections must be confirmed")

            env_name = self.env_name_var.get().strip()
            if not re.fullmatch(r"[A-Za-z0-9]{1,10}", env_name):
                raise ValueError("Env name must be 1-10 characters: English letters and digits only (no spaces).")

            sim_days = self.sim_days_var.get()
            start_date = self.start_date_var.get()
            step_size = self.step_size_var.get()
            action_type = self.action_type_var.get()
            include_hour = self.include_hour_var.get()
            include_day_of_year = self.include_day_of_year_var.get()
            include_episode_progress = self.include_episode_progress_var.get()
            normalize_observation = self.normalize_observation_var.get()
            reward_mode = self.reward_mode.get()

            code = (
                "from src.env_wrapper import MuFlex\n"
                f"fmu_configs = {pprint.pformat(self.confirmed_fmu_configs)}\n"
                f"env = MuFlex(fmu_configs=fmu_configs, sim_days={sim_days}, start_date={start_date}, "
                f"step_size={step_size}, action_type='{action_type}', include_hour={include_hour}, "
                f"include_day_of_year={include_day_of_year}, "
                f"include_episode_progress={include_episode_progress}, "
                f"normalize_observation={normalize_observation}, "
                f"reward_mode='{reward_mode}')"
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
    def clear_action_entries(self, frame: tk.Frame) -> None:
        for w in frame.winfo_children():
            w.destroy()

    def _populate_action_entries(self, entry: dict[str, object], frame: tk.Frame) -> None:
        io_type = entry["io_type"]
        inputs = IO_DEFINITIONS[io_type]["INPUTS"]
        mins = entry["mins"]
        maxs = entry["maxs"]
        vars = entry["vars"]
        for j, (name, var, mn, mx) in enumerate(zip(inputs, vars, mins, maxs)):
            tk.Label(frame, text=name, bg="white").grid(row=j, column=0, sticky="w")
            tk.Entry(frame, textvariable=var, width=10).grid(row=j, column=1)
            tk.Label(frame, text=f"[{mn}, {mx}]", bg="white").grid(row=j, column=2, sticky="w")

    def on_building_select(self, label: str) -> None:
        if label not in self.building_labels:
            return
        self.building_var.set(label)
        self.clear_action_entries(self.building_action_entry_frame)
        entry = next(e for e in self.action_entries if e["label"] == label)
        self._populate_action_entries(entry, self.building_action_entry_frame)

    def on_pv_select(self, label: str) -> None:
        if label not in self.pv_labels:
            return
        self.pv_var.set(label)
        self.clear_action_entries(self.pv_action_entry_frame)
        entry = next(e for e in self.action_entries if e["label"] == label)
        self._populate_action_entries(entry, self.pv_action_entry_frame)

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
                "reward_mode": params.get("reward_mode", "example_reward"),
                "action_type": params.get("action_type", "continuous"),
                "include_hour": params.get("include_hour", True),
                "include_day_of_year": params.get("include_day_of_year", True),
                "include_episode_progress": params.get("include_episode_progress", True),
                "normalize_observation": params.get("normalize_observation", True),
                "rl_control_window_only": params.get("rl_control_window_only", True),
                "office_hour_start": params.get("office_hour_start", "08:00"),
                "office_hour_end": params.get("office_hour_end", "18:00"),
                "step_info_print_interval": params.get("step_info_print_interval", 10),
            }

            self.rl_control_window_only_var.set(bool(self.current_env_params["rl_control_window_only"]))
            self.office_hour_start_var.set(str(self.current_env_params["office_hour_start"]))
            self.office_hour_end_var.set(str(self.current_env_params["office_hour_end"]))
            self.step_info_print_interval_var.set(str(self.current_env_params["step_info_print_interval"]))

            self.action_entries = []
            self.building_labels = []
            self.pv_labels = []
            counts: dict[str, int] = {}
            for idx, cfg in enumerate(fmu_configs):
                io_type = cfg["io_type"]
                category = IO_DEFINITIONS[io_type].get("category", "building").lower()
                counts[category] = counts.get(category, 0) + 1
                label_prefix = "PV" if category == "pv" else "Building"
                lbl = f"{label_prefix} {counts[category]}"

                mins = IO_DEFINITIONS[io_type]["base_mins"]
                maxs = IO_DEFINITIONS[io_type]["base_maxs"]
                vars = [tk.StringVar() for _ in IO_DEFINITIONS[io_type]["INPUTS"]]

                self.action_entries.append({
                    "index": idx,
                    "label": lbl,
                    "category": category,
                    "io_type": io_type,
                    "mins": mins,
                    "maxs": maxs,
                    "vars": vars,
                })
                if category == "pv":
                    self.pv_labels.append(lbl)
                else:
                    self.building_labels.append(lbl)

            building_menu = self.building_menu["menu"]
            building_menu.delete(0, "end")
            if self.building_labels:
                for lbl in self.building_labels:
                    building_menu.add_command(label=lbl, command=lambda v=lbl: self.on_building_select(v))
                self.building_var.set(self.building_labels[0])
                self.on_building_select(self.building_labels[0])
                self.building_menu.config(state="normal")
            else:
                building_menu.add_command(label="", command=lambda: None)
                self.building_var.set("")
                self.building_menu.config(state="disabled")
                self.clear_action_entries(self.building_action_entry_frame)

            pv_menu = self.pv_menu["menu"]
            pv_menu.delete(0, "end")
            if self.pv_labels:
                for lbl in self.pv_labels:
                    pv_menu.add_command(label=lbl, command=lambda v=lbl: self.on_pv_select(v))
                self.pv_var.set(self.pv_labels[0])
                self.on_pv_select(self.pv_labels[0])
                self.pv_menu.config(state="normal")
            else:
                pv_menu.add_command(label="", command=lambda: None)
                self.pv_var.set("")
                self.pv_menu.config(state="disabled")
                self.clear_action_entries(self.pv_action_entry_frame)

            self.physical_actions = []
            self.actions_confirmed = False
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def confirm_actions(self) -> None:
        try:
            self.physical_actions = [None] * len(self.current_fmu_configs)
            for entry in self.action_entries:
                io_type = entry["io_type"]
                mins = entry["mins"]
                maxs = entry["maxs"]
                vars = entry["vars"]
                label = entry["label"]
                vals: list[float] = []
                for name, var, mn, mx in zip(IO_DEFINITIONS[io_type]["INPUTS"], vars, mins, maxs):
                    v_str = var.get()
                    if not v_str:
                        raise ValueError(f"Missing value for {label} {name}")
                    v = float(v_str)
                    if v < mn or v > mx:
                        raise ValueError(f"Value {v} out of bounds [{mn}, {mx}] for {label} {name}")
                    vals.append(v)
                self.physical_actions[entry["index"]] = vals

            if any(v is None for v in self.physical_actions):
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

            max_steps_text = self.max_steps_var.get().strip()
            if max_steps_text:
                max_steps = int(max_steps_text)
                if max_steps <= 0:
                    raise ValueError("Max steps must be a positive integer")
            else:
                max_steps = None

            office_hour_start = self.office_hour_start_var.get().strip()
            office_hour_end = self.office_hour_end_var.get().strip()
            hour_pattern = r"([01]\d|2[0-3]):([0-5]\d)"
            if not re.fullmatch(hour_pattern, office_hour_start):
                raise ValueError("Office hour start must use HH:MM format")
            if not re.fullmatch(hour_pattern, office_hour_end):
                raise ValueError("Office hour end must use HH:MM format")

            step_info_print_interval_text = self.step_info_print_interval_var.get().strip()
            if step_info_print_interval_text:
                step_info_print_interval = int(step_info_print_interval_text)
                if step_info_print_interval < 0:
                    raise ValueError("Step info print interval must be >= 0")
            else:
                step_info_print_interval = 0

            params = dict(
                fmu_configs=self.current_fmu_configs,
                sim_days=self.current_env_params.get("sim_days", 1),
                start_date=self.current_env_params.get("start_date", 1),
                step_size=self.current_env_params.get("step_size", 900),
                reward_mode=self.current_env_params.get("reward_mode", "example_reward"),
                save_results=self.save_results_var.get(),
                max_steps=max_steps,
                action_type=self.current_env_params.get("action_type", "continuous"),
                include_hour=self.current_env_params.get("include_hour", True),
                include_day_of_year=self.current_env_params.get("include_day_of_year", True),
                include_episode_progress=self.current_env_params.get("include_episode_progress", True),
                normalize_observation=self.current_env_params.get("normalize_observation", True),
                rl_control_window_only=self.rl_control_window_only_var.get(),
                office_hour_start=office_hour_start,
                office_hour_end=office_hour_end,
                step_info_print_interval=step_info_print_interval,
                physical_actions=self.physical_actions,
            )

            thread = threading.Thread(target=run_baseline, kwargs=params, daemon=True)
            thread.start()
        except Exception as exc:
            messagebox.showerror("Error", str(exc))


if __name__ == "__main__":
    app = MuFlexGUI()
    app.mainloop()
