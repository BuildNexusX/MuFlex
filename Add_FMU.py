"""Tkinter utility to manage FMU type definitions.

The tool allows users to add, edit and persist FMU I/O templates stored in
``fmu_config.json``.
"""

import json
import tkinter as tk
from tkinter import messagebox, simpledialog
from src.config import load_fmu_config, save_fmu_config

class FMUTypeGUI(tk.Tk):
    """Simple editor for adding or removing FMU archetypes."""
    def __init__(self) -> None:
        super().__init__()
        self.title("FMU Type Manager")
        self.geometry("800x600")
        self.config(bg="white")
        # built‑in types that cannot be deleted
        self.fixed_types = {"OfficeS", "OfficeM"}
        self.data = load_fmu_config()
        self.current_type = None

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=3)
        self.rowconfigure(0, weight=1)

        # List of types
        self.type_list = tk.Listbox(self)
        self.type_list.grid(row=0, column=0, sticky="ns")
        for t in self.data:
            self.type_list.insert(tk.END, t)
        self.type_list.bind("<<ListboxSelect>>", self.on_select)

        # Detail frame
        self.detail = tk.Frame(self, bg="white")
        self.detail.grid(row=0, column=1, sticky="nsew")
        self.fields = {}
        labels = [
            "INPUTS",
            "OUTPUTS",
            "ob_base_low",
            "ob_base_high",
            "dims",
            "intervals",
            "base_mins",
            "base_maxs",
        ]
        for i, name in enumerate(labels):
            tk.Label(self.detail, text=name, bg="white").grid(row=i, column=0, sticky="w")
            entry = tk.Entry(self.detail, width=60)
            entry.grid(row=i, column=1, sticky="w")
            self.fields[name] = entry

        # Buttons
        btn_frame = tk.Frame(self, bg="white")
        btn_frame.grid(row=1, column=0, columnspan=2, pady=5)
        tk.Button(btn_frame, text="Add Type", command=self.add_type, bg="white").pack(side="left", padx=5)
        tk.Button(btn_frame, text="Delete Type", command=self.delete_type, bg="white").pack(side="left", padx=5)
        tk.Button(btn_frame, text="Save", command=self.save, bg="white").pack(side="left", padx=5)

    def on_select(self, event) -> None:
        """Populate the detail pane when a type is chosen from the list."""
        if not self.type_list.curselection():
            return
        t = self.type_list.get(self.type_list.curselection())
        self.current_type = t
        cfg = self.data[t]
        for name, entry in self.fields.items():
            entry.delete(0, tk.END)
            entry.insert(0, json.dumps(cfg[name]))

    def add_type(self) -> None:
        """Create a blank FMU template and append it to the list."""
        name = simpledialog.askstring("Type name", "Enter new FMU type name:")
        if not name:
            return
        if name in self.data:
            messagebox.showerror("Error", "Type already exists")
            return
        self.data[name] = {
            "INPUTS": [],
            "OUTPUTS": [],
            "ob_base_low": [],
            "ob_base_high": [],
            "dims": [],
            "intervals": [],
            "base_mins": [],
            "base_maxs": [],
        }
        self.type_list.insert(tk.END, name)

    def delete_type(self) -> None:
        """Remove the selected FMU type unless it's protected."""
        if not self.type_list.curselection():
            return
        t = self.type_list.get(self.type_list.curselection())
        if t in self.fixed_types:
            messagebox.showerror("Error", "Cannot delete fixed type")
            return
        del self.data[t]
        self.type_list.delete(self.type_list.curselection())
        self.current_type = None
        for entry in self.fields.values():
            entry.delete(0, tk.END)

    def save(self) -> None:
        """Persist edited definitions back to ``fmu_config.json``."""
        if self.current_type is not None:
            try:
                for name, entry in self.fields.items():
                    self.data[self.current_type][name] = json.loads(entry.get() or "[]")
            except json.JSONDecodeError:
                messagebox.showerror("Error", "Invalid JSON format in fields")
                return
        save_fmu_config(self.data)
        messagebox.showinfo("Saved", "Configuration saved")

if __name__ == "__main__":
    gui = FMUTypeGUI()
    gui.mainloop()