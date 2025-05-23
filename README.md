# Quadrotor Payload and Disturbance Simulation (Isaac Gym)

This repository contains simulation experiments for mid-flight payload attachment, wind disturbances, and runtime mass modifications using the Aerial Gym framework (built on Isaac Gym Preview 4).

---

## Features
- Mid-flight payload attachment via runtime mass update
- Wind disturbance injection using external forces
- Visualization of trajectory dips due to disturbances
- CPU pipeline usage for runtime inertial property modification

---

## Installation Guide

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/quadrotor-payload-simulation.git
cd quadrotor-payload-simulation
```

### 2. Isaac Gym Setup
- Download **Isaac Gym Preview 4** from NVIDIA (requires NVIDIA developer access).
- Follow the [official installation instructions](https://developer.nvidia.com/isaac-gym) to set up the Python bindings.

Example setup:
```bash
cd isaacgym/python
pip install -e .
```

### 3. Set Up Environment (Optional)
We recommend using a conda environment:
```bash
conda create -n aerial-gym python=3.8
conda activate aerial-gym
pip install torch numpy matplotlib
```

---

## Folder Structure
```
quadrotor-payload-simulation/
├── collected_data/        # .npz output data
├── scripts/               # Simulation scripts
├── plots/                 # Generated visualizations
├── README.md              # This file
```

---

## Running Experiments

### Payload Mass Change Mid-flight
```bash
python scripts/hover_with_mass_update.py
```

### Wind Disturbance Mid-flight
```bash
python scripts/hover_with_wind_force.py
```

### Expected Output:
- `.npz` files in `collected_data/`
- Graphs of position and Euler angle changes (see `plots/`)

---

## Visualization
To generate plots from saved `.npz` files:
```bash
python scripts/plot_results.py
```

This will show the `z`-position dip at disturbance onset.

---

## Author
Pranav Kulkarni — UC San Diego ERL

---
