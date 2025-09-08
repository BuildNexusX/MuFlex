# MuFlex
**A Scalable, Physics-based Platform for Multi-Building Flexibility Analysis and Coordination** 🚀

<p align="center">
  <img src="figs/MUFLEX.png" alt="MUFLEX Logo" width="280">
</p>

<p align="center">
  🎥 <a href="https://www.bilibili.com/video/BV1aca9zPEVG/?share_source=copy_web&vd_source=177b394f89d5b57beb8201809652c5df" target="_blank"><b>Watch Demo Video on Bilibili</b></a>
</p>

## ⚙️ Installation & Dependencies (Python 3.12 recommended)

```bash
# Windows
# Clone the repository
git clone https://github.com/BuildNexusX/MUFLEX.git
cd MUFLEX

# Create and activate a Python 3.12 environment (Conda recommended)
conda create -n MUFLEX python=3.12 -y
conda activate MUFLEX
```


### Install packages
Run the commands listed in requirements/install.txt 

### Run Scripts

```bash
# Run Baseline without GUI
python -m algo.baseline
# Main Simulation GUI
python MuFlex.py 
# Model Management GUI
python Add_FMU.py
```


---

# ⚡️ Simulate Building Clusters in 5 Minutes
Start exploring building flexibility with just a few clicks using ▶️`python MuFlex.py`

MuFlex scales simulation from a single building to large clusters while keeping the UI simple and configuration lightweight.

## 🖥️ MuFlex GUI Overview - three tabs

### 🛠️ Create Env 
#### ❗ confirm simulation settings step by step
- **FMU Configuration:** choose building types (managed in `Add_FMU.py`) and provide FMU paths.
- **Simulation Parameters:** set the number of days (duration), start day of year (on which day of the year the simulation begins), and step size (must match the model inside timestep)
- **Action/Observation Space Preview:** choose the environment’s action space type and optionally include hour-of-day in the observation vector.
You will get a preview of all the observations and actions using the configuration you defined.
- **Reward Function:** use the default reward with tunable weights (used in Paper) or a custom script.
- **Env Name & Create:** name and save the environment once everything is configured.

### 🚀 Run Baseline 
#### for every environment you created
- **Environment Selector:** pick a saved environment.
- **Execution Options:** enable result saving and per-step logging.
- **Baseline Actions:** specify fixed actions for each FMU.
- **Run:** launch the baseline controller.

### 📂 Manage Envs
#### existing `example` is the environment used in paper
- **Saved Environments:** list all stored environments and their configurations.
- **Delete Selected:** remove outdated environments.

## 📊 Simulation Output Files
After a simulation completes, MuFlex saves several files in the output directory:

- `Output_EPExport_Model` – EnergyPlus simulation results.
- `simulation_data_<DATE>_<TIME>` – input and output records for each FMU.
- `rewards.xlsx` – per-step reward trace (columns vary with reward mode).
- `.txt` files – FMU execution logs.

---

## 🔌 Add Custom FMU Types
▶️`python Add_FMU.py` helps you register new building templates through a GUI. See `test_fmu` for an example.

### What is a “type”?
Each *type* represents an FMU model with its own input and output variables.
FMUs exposing different interfaces should be registered under different types.

### JSON fields
Every type entry in `config/fmu_config.json` stores the following lists:

- **`INPUTS`** – controllable set‑point variables accepted by the FMU.
- **`OUTPUTS`** – variables returned by the FMU and exposed to the agent.
- **`ob_base_low`** / **`ob_base_high`** – expected min/max for each output, used for observation normalisation.
- **`dims`** – number of discrete bins for each input (used when the action space type is `discrete`).
- **`intervals`** – physical step size for each input; continuous actions are snapped to multiples of this value.
- **`base_mins`** / **`base_maxs`** – physical lower and upper limits for each input.

### GUI workflow
#### 📋 Type List
Browse existing FMU types.

#### 📝 Detail Editor
Edit the lists above using JSON strings.

#### ⚙️ Management Buttons
Add new types, remove outdated ones, and save everything back to the configuration file.

---

## 📁 Repository Structure

```text
MuFlex/
├── MuFlex.py                ▶️Simulation GUI launcher
├── Add_FMU.py               ▶️GUI for registering custom FMU types
├── README.md
├── algo/                    📁Reward scripts and baseline
│   ├── baseline.py          # Rule‑based controller (can be used for all cases)
│   ├── custom_reward.py     # Edit this file to create your own reward calculation
│   └── default_reward.py    # Reward function used in paper
├── config/                  📁FMU type definitions
│   └── fmu_config.json
├── figs/                    📁Logos & diagrams
├── models/                  📁EnergyPlus → FMU building models
│   ├── add_fmu_test/        # Example FMU of using Add_FMU.py to integrate your own model
│   ├── medium_office/       # Medium office FMUs
│   ├── small_office/        # Small office FMUs
│   └── weather/             # Weather files
├── requirements/            📁Dependency files
│   └── install.txt          # Commands can be pasted for installing required packages
└── src/                     📁Core environment code
    ├── buffer.py            # Replay buffer for RL implementation
    ├── config.py            # Environment configuration helpers
    ├── env.py               # Gymnasium-style main environment
    └── env_list.txt         # Saved environments (created using MuFlex.py)
```

---

## 🤖 RL Implementation Example
### 4-Building Cluster Case used in Paper
You can go to src/env_list.txt to copy and paste a created environment for RL implementation with defined parameters.

```python

from src.env import MuFlex
fmu_configs = [{'io_type': 'OfficeS',
  'path': 'C:/Users/Administrator/Desktop/MuFlex/models/small_office/small_baseline_v1.fmu'},
 {'io_type': 'OfficeS',
  'path': 'C:/Users/Administrator/Desktop/MuFlex/models/small_office/small_baseline_v2.fmu'},
 {'io_type': 'OfficeM',
  'path': 'C:/Users/Administrator/Desktop/MuFlex/models/medium_office/medium_baseline_v1.fmu'},
 {'io_type': 'OfficeM',
  'path': 'C:/Users/Administrator/Desktop/MuFlex/models/medium_office/medium_baseline_v2.fmu'}]
env = MuFlex(fmu_configs=fmu_configs, sim_days=1, start_date=201, step_size=900, action_type='continuous', include_hour=True, reward_mode='default')
obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
env.close()

```
---

## 📚 Citation

If you use **MuFlex**, please cite the **arXiv preprint** first:

**Wu, Z., Korolija, I., & Tang, R. (2025). _MuFlex: A Scalable, Physics-based Platform for Multi-Building Flexibility Analysis and Coordination._**  
arXiv. https://doi.org/10.48550/arXiv.2508.13532 • [arXiv page](https://arxiv.org/abs/2508.13532)

**BibTeX**
```bibtex
@misc{wu2025muflex,
  title         = {MuFlex: A Scalable, Physics-based Platform for Multi-Building Flexibility Analysis and Coordination},
  author        = {Wu, Ziyan and Korolija, Ivan and Tang, Rui},
  year          = {2025},
  eprint        = {2508.13532},
  archivePrefix = {arXiv},
  doi           = {10.48550/arXiv.2508.13532},
  url           = {https://arxiv.org/abs/2508.13532}
}
```

---

## 📝 License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project is released under the **MIT License**. See [`LICENSE`](LICENSE) for details.

### *Feel free to open issues, submit pull requests, or start a discussion. Happy Coding!* 🎉
