"""Helpers for reading and writing FMU configuration files."""

import json
from pathlib import Path
from typing import Dict, Any

# Default location of the configuration JSON used by the GUI and command line tools.
# Projects embedding MuFlex may override this by assigning a new value to ``CONFIG_PATH`` before calling the helpers below.
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "fmu_config.json"

def load_fmu_config() -> Dict[str, Any]:
    """Return the parsed FMU configuration dictionary."""
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_fmu_config(data: Dict[str, Any]) -> None:
    """Persist the FMU configuration mapping to disk."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)