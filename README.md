# UAV Navigation with Obstacle Avoidance using Reinforcement Learning

> Reinforcement Learning-based autonomous UAV navigation with obstacle avoidance in 3D environments using curriculum learning – Bachelor Thesis.

**[Read the full thesis (PDF)](thesis.pdf)**

<!-- PLACEHOLDER: drag and drop an overview image here (e.g., architecture diagram or simulation screenshot) -->

---

## Table of Contents

- [Abstract](#abstract)
<!-- - [Demo](#demo) -->
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Training](#training)
- [Experiment Tracking](#experiment-tracking)
- [Troubleshooting](#troubleshooting)
- [Simulation Stack](#simulation-stack)
- [Acknowledgments](#acknowledgments)

---

## Abstract

This project trains a quadrotor UAV to autonomously navigate toward waypoints while avoiding obstacles in a 3D environment. The agent is trained using **Proximal Policy Optimization (PPO)** with **curriculum learning** that progressively increases environment difficulty — from open-space flight to dense obstacle fields. The UAV perceives its surroundings through a simulated **360-degree LiDAR** sensor and is controlled via velocity commands. Training and evaluation are tracked with **Weights & Biases**.

<!-- --- -->

<!-- ## Demo

### 3D Trajectory Plots

<!-- PLACEHOLDER: drag and drop here -->

|              Stage 0 — Open Space              |           Stage 1 — Random Obstacles           |              Stage 3 — Dense Field              |
| :----------------------------------------------: | :----------------------------------------------: | :----------------------------------------------: |
| <!-- PLACEHOLDER: stage 0 screenshot --> | <!-- PLACEHOLDER: stage 1 screenshot --> | <!-- PLACEHOLDER: stage 3 screenshot --> |

|             Stage 4 — Around Target             |           Stage 5 — Dense Near Target           |
| :----------------------------------------------: | :----------------------------------------------: |
| <!-- PLACEHOLDER: stage 4 screenshot --> | PLACEHOLDER: stage 5 screenshot --> |

---

## Architecture

The system is built around the following components:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Loop (PPO)                      │
│                     Stable Baselines 3 + W&B                    │
├─────────────────────────────────────────────────────────────────┤
│                      Wrapper Chain (outermost → innermost)      │
│                                                                 │
│  DummyVecEnv                                                    │
│   └─ LidarFlattenWrapper       (flatten dict obs → 1D vector)   │
│       └─ NormalizeObsWrapper   (static normalization)           │
│           └─ CustomRewardWrapper (lidar-aware reward shaping)   │
│               └─ LidarObsWrapper (raycasting → distance obs)    │
│                   └─ VectorVoyagerEnv (base environment)        │
│                       └─ PyFlyt QuadXBaseEnv                    │
│                           └─ PyBullet Physics Engine            │
├─────────────────────────────────────────────────────────────────┤
│                      Curriculum Controller                      │
│        Stages 0–5: progressively harder obstacle configs        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Requirement           | Version                                                                         |
| --------------------- | ------------------------------------------------------------------------------- |
| **Python**      | >= 3.12                                                                         |
| **uv**          | latest ([install guide](https://docs.astral.sh/uv/getting-started/installation/))  |
| **OS**          | macOS or Linux (PyBullet requires a display or `xvfb` for headless rendering) |
| **W&B Account** | Free account at [wandb.ai](https://wandb.ai) (for experiment tracking)             |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/uav-nav-obstacle-avoidance-rl.git
cd uav-nav-obstacle-avoidance-rl
```

### 2. Install dependencies with uv

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install all dependencies
uv sync
```

This creates a `.venv` in the project root and installs all dependencies from `pyproject.toml` and `uv.lock`.

### 3. Configure Weights & Biases

```bash
uv run wandb login
```

Paste your API key from [wandb.ai/authorize](https://wandb.ai/authorize) when prompted.

### 4. (Optional) Create a `.env` file

```bash
# .env
WANDB_API_KEY=your_api_key_here
```

### 5. Verify the installation

```bash
# Quick environment sanity check — creates the env, prints observation/action spaces, and exits
uv run python -m uav_nav_obstacle_avoidance_rl.test.base_env_test
```

---

## Project Structure

```
uav-nav-obstacle-avoidance-rl/
│
├── uav_nav_obstacle_avoidance_rl/       # Main Python package
│   ├── config.py                        # Paths, logging setup
│   ├── environment/                     # Environment modules
│   │   ├── vector_voyager_env.py        # Base quadrotor environment (PyFlyt)
│   │   ├── occupancy_grid.py            # 2D grid for obstacle placement
│   │   ├── lidar_wrapper.py             # Simulated 360° LiDAR (raycasting)
│   │   ├── reward_wrapper.py            # Custom lidar-aware reward function
│   │   ├── normalize_obs_wrapper.py     # Static observation normalization
│   │   ├── flaten_wrapper.py            # Dict → flat vector wrapper
│   │   └── cam_wrapper.py               # Third-person camera wrapper
│   │
│   ├── modeling/                        # Training pipeline
│   │   ├── train.py                     # CLI entry point (run-train, sweep, seed-sweep)
│   │   ├── config-defaults.yaml         # Default environment & PPO config
│   │   ├── exp-6-sweep.yaml             # Bayesian hyperparameter sweep config
│   │   └── seed-three.yaml              # Multi-seed grid sweep config
│   │
│   ├── utils/                           # Callbacks & utilities
│   │   ├── env_factory.py               # Environment builder (wrapper chain)
│   │   ├── curriculum_callback.py       # Curriculum learning scheduler
│   │   ├── eval_metrics_callback.py     # Evaluation metrics & trajectory analysis
│   │   └── train_metrics_callback.py    # Per-step training metrics to W&B
│   │
│   └── test/
│       └── base_env_test.py            # Environment analysis & tests
│
├── reports/                             # Training outputs (gitignored)
├── wandb/                               # W&B local data (gitignored)
├── pyproject.toml                       # Project metadata & dependencies
├── uv.lock                              # Locked dependency versions
└── README.md
```

---

## Configuration

All environment, reward, curriculum, and PPO settings are in a single YAML file:

**[`uav_nav_obstacle_avoidance_rl/modeling/config-defaults.yaml`](uav_nav_obstacle_avoidance_rl/modeling/config-defaults.yaml)**

---

## Training

All training commands use [uv](https://docs.astral.sh/uv/) to run within the project's virtual environment.

### Single Training Run

```bash
uv run python -m uav_nav_obstacle_avoidance_rl.modeling.train run-train \
  --exp-name "my-experiment" \
  --wandb-tags exp-6 \
  --timesteps 2000000 \
  --eval-freq 200000 \
  --n-envs 2
```

| Flag                  | Default       | Description                         |
| --------------------- | ------------- | ----------------------------------- |
| `--exp-name`        | `"exp"`     | Experiment name (appears in W&B)    |
| `--timesteps`       | `2,000,000` | Total training timesteps            |
| `--eval-freq`       | `200,000`   | Evaluate every N timesteps          |
| `--n-envs`          | `2`         | Parallel training environments      |
| `--n-eval-episodes` | `30`        | Episodes per evaluation cycle       |
| `--seed`            | `9`         | Random seed                         |
| `--wandb-tags`      | `None`      | Tags for W&B filtering (repeatable) |

### Hyperparameter Sweep (Bayesian)

```bash
# Create a new sweep and start an agent
uv run python -m uav_nav_obstacle_avoidance_rl.modeling.train sweep \
  --exp-name "exp-6-" \
  --sweep-config uav_nav_obstacle_avoidance_rl/modeling/exp-6-sweep.yaml \
  --wandb-tags exp-6 \
  --count 25

# Join an existing sweep from another terminal
uv run python -m uav_nav_obstacle_avoidance_rl.modeling.train sweep \
  --exp-name "exp-6-" \
  --sweep-id <SWEEP_ID> \
  --count 25
```

### Multi-Seed Sweep (Reproducibility)

```bash
uv run python -m uav_nav_obstacle_avoidance_rl.modeling.train seed-sweep \
  --exp-name "exp-6-" \
  --sweep-config uav_nav_obstacle_avoidance_rl/modeling/seed-three.yaml \
  --wandb-tags seed --wandb-tags exp-6 \
  --timesteps 2000000 \
  --eval-freq 200000
```

### Training Output

Training artifacts are saved under the W&B run directory inside `reports/`:

- `models/best_model.zip` — best model checkpoint (by evaluation success rate)
- `tensorboard/` — TensorBoard logs
- `run.log` — full training log

---

## Experiment Tracking

All experiments are tracked in **Weights & Biases** under the project `uav-nav-obstacle-avoidance-rl`.

### Viewing Results

After logging in to W&B, view your runs at:

```
https://wandb.ai/<your-username>/uav-nav-obstacle-avoidance-rl
```

---

## Simulation Stack

| Component                                                   | Role                                            |
| ----------------------------------------------------------- | ----------------------------------------------- |
| [PyBullet](https://pybullet.org)                               | Physics engine, collision detection, raycasting |
| [PyFlyt](https://github.com/jjshoots/PyFlyt)                   | Quadrotor dynamics, velocity control interface  |
| [Stable Baselines 3](https://stable-baselines3.readthedocs.io) | PPO implementation, vectorized environments     |
| [Gymnasium](https://gymnasium.farama.org)                      | Environment API standard                        |
| [Weights & Biases](https://wandb.ai)                          | Experiment tracking, hyperparameter sweeps      |

---

## Troubleshooting

### Common Issues

**`uv` command not found:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Then restart your shell or run: source ~/.bashrc
```

**W&B authentication error:**

```bash
uv run wandb login
```

**Out of memory during training:**

- Reduce `--n-envs` (default: 2)
- Reduce `batch_size` in config
- Training runs on CPU by default (`device="cpu"`)

---

## Acknowledgments

I would like to thank Prof. Dr. Christian Seidel for his support. I am also grateful to Konstantin Bake for his valuable feedback, support, and encouragement along the way!
