# btorch Examples & Tutorials

A collection of reproducible examples, tutorials, and reference implementations for **btorch** — a PyTorch-based spiking neural network (SNN) library for recurrent spiking neural networks (RSNNs).

This repository demonstrates how to use btorch across a range of scenarios, from single-neuron dynamics to large-scale connectome-based simulations with interactive visualization.

---

## Table of Contents

- [Overview](#overview)
- [What is btorch?](#what-is-btorch)
- [Repository Structure](#repository-structure)
- [Components](#components)
  - [Basic Examples](#basic-examples)
  - [brunel2000](#brunel2000)
  - [glif_net](#glif_net)
  - [mice_v1](#mice_v1)
  - [Interactive Visualizer](#interactive-visualizer)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core btorch Patterns](#core-btorch-patterns)
- [References](#references)

---

## Overview

This repository provides **five progressive learning tracks**:

| Track | Directory | Description |
|-------|-----------|-------------|
| **Fundamentals** | `basic_examples/` | Learn single-neuron and RSNN basics with btorch |
| **Classical Reproduction** | `brunel2000/` | Reproduce Brunel (2000) balanced E/I networks |
| **Cognitive Tasks** | `glif_net/` | Train GLIF-RSNNs on the 12AX working memory task |
| **Large-Scale Simulation** | `mice_v1/` | Connectome-based fly brain simulation with environment I/O |
| **Interactive Visualization** | `mice_v1/interactive_visualizer/` | Streamlit app for real-time SNN exploration |

Each component is self-contained and can be explored independently.

---

## What is btorch?

**btorch** is a PyTorch library for building and simulating recurrent spiking neural networks (RSNNs). It provides:

- **Neuron models**: LIF, GLIF3 (Generalized LIF with adaptation currents)
- **Synapse models**: ExponentialPSC, AlphaPSC, HeterSynapsePSC (heterogeneous receptor types)
- **Recurrent networks**: `RecurrentNN` with built-in state management, gradient checkpointing, and CPU offloading
- **Connectivity**: Sparse and dense connections with Dale's law enforcement
- **State management**: Memory registration, reset, and tracing via `register_memory`
- **Simulation utilities**: Poisson noise generation, environment context managers for `dt`

Learn more at: [Criticality-Cognitive-Computation-Lab/btorch](https://github.com/Criticality-Cognitive-Computation-Lab/btorch)

---

## Repository Structure

```
btorch-examples/
├── README.md                           # This file
├── environment.yaml                    # Conda environment specification
├── start                               # Binder/Colab startup script
│
├── basic_examples/                     # Fundamentals
│   ├── 01_single_neuron_dynamics.ipynb
│   └── 02_basic_rsnn.ipynb
│
├── brunel2000/                         # Classical balanced network
│   ├── README.md                       # Detailed component docs
│   ├── brunel2000_tutorial.ipynb       # Step-by-step notebook tutorial
│   ├── config.py                       # OmegaConf dataclass configs (Model A & B)
│   ├── simulate.py                     # Simulation loop
│   ├── sweep.py                        # Parameter sweep for phase diagram
│   ├── analysis.py                     # Firing rate, CV, raster analysis
│   ├── connection.py                   # Sparse connectivity builders
│   ├── run_model_a.py                  # Quick runner: identical E/I neurons
│   ├── run_model_b.py                  # Quick runner: heterogeneous E/I neurons
│   ├── brunel_sweep.py                 # Launcher for parallel sweeps
│   └── brunel_analysis.py              # Phase diagram & comparison plots
│
├── glif_net/                           # GLIF-RSNN cognitive task training
│   ├── notebooks/
│   │   ├── 12ax_tutorial.ipynb         # Full 12AX training pipeline
│   │   └── tutorial_12ax_utils.py      # Helper functions for notebook
│   ├── src/
│   │   ├── model.py                    # SingleLayerGLIFRSNN definition
│   │   ├── connectivity.py             # Sparse E/I connectivity with Dale's law
│   │   ├── checkpoint.py               # Save/load with memory snapshots
│   │   └── calibration_grad.py         # Gradient-based calibration
│   ├── tests/
│   │   └── test_12ax.py                # Unit tests for 12AX pipeline
│   └── run_12ax.slurm                  # SLURM batch script for cluster training
│
└── mice_v1/                            # Large-scale connectome simulation
    ├── src/
    │   ├── models/
    │   │   ├── brain.py                # FlyBrain: connectome-based RSNN
    │   │   ├── base.py                 # Base model utilities
    │   │   ├── weights.py              # Weight initialization
    │   │   ├── input_layers.py         # Environment input embeddings
    │   │   ├── glif_synapse.py         # GLIF-specific synapse wrappers
    │   │   └── compat_synapse.py       # Compatibility layer for HeterSynapsePSC
    │   ├── utils/
    │   │   ├── dataloader.py           # Data loading utilities
    │   │   ├── preprocess.py           # Preprocessing pipeline
    │   │   ├── simulated_dataset.py    # Synthetic dataset generation
    │   │   ├── vis_utils.py            # Visualization helpers
    │   │   ├── multitraces.py          # Multi-trace plotting
    │   │   └── device.py               # Device management
    │   └── runner.py                   # Experiment runner (train / simulate)
    ├── interactive_visualizer/         # Streamlit interactive visualizer
    │   ├── snn_visualizer.py           # Main Streamlit application
    │   ├── run.sh                      # Launch script
    │   └── lib/                        # Frontend libraries (vis.js, tom-select)
    ├── tutorial/
    │   └── mock_assets.py              # Mock data for tutorials
    ├── tutorial_generate_and_sim.yaml  # Config for tutorial pipeline
    ├── tutorial_companion_template.ipynb  # Cell-by-cell tutorial notebook
    └── run_tutorial_generate_and_sim_notebook.py  # Scripted tutorial runner
```

---

## Components

### Basic Examples

**Directory**: `basic_examples/`

Two foundational notebooks introducing btorch concepts:

- **`01_single_neuron_dynamics.ipynb`** — Explore LIF and GLIF3 neuron dynamics, membrane potential traces, refractory periods, and after-spike currents (ASC). Simulates a two-neuron system with dense connectivity.
- **`02_basic_rsnn.ipynb`** — Build a basic recurrent spiking neural network. Covers neuron models (LIF, GLIF), synapse models (AlphaPSC), state initialization (`init_net_state`, `reset_net`), and the `environ.context(dt=...)` pattern.

**Start here** if you are new to btorch or spiking neural networks.

---

### brunel2000

**Directory**: `brunel2000/`

Reproduces the classic **Brunel (2000)** paper: *Dynamics of Sparsely Connected Networks of Excitatory and Inhibitory Spiking Neurons*. This is the canonical benchmark for balanced E/I spiking networks.

**Features**:
- **Model A**: Identical E/I neurons with uniform parameters
- **Model B**: Heterogeneous E/I neurons with population-specific time constants, weights, and delays
- **Four dynamical regimes**: SR (Synchronous Regular), AR (Asynchronous Regular), AI (Asynchronous Irregular), SI (Synchronous Irregular)
- **Phase diagram sweeps**: Grid search over inhibitory strength `g` and external input rate `ν_ext`
- **OmegaConf configuration**: Dataclass-first configs with CLI overrides
- **Parallel execution**: `ProcessPoolExecutor` for parameter sweeps with resume capability

**Quick Start**:
```bash
# Run Model A (AI regime)
python brunel2000/run_model_a.py

# Run Model B (heterogeneous parameters)
python brunel2000/run_model_b.py neuron.tau_i=10.0 synapse.g_e=6.0

# Phase diagram sweep
python brunel2000/brunel_sweep.py \
    sweep.sweep_g="[3.0,4.0,5.0,6.0,7.0,8.0]" \
    sweep.sweep_nu_ext_ratio="[0.5,1.0,1.5,2.0,3.0,4.0]"

# Analyze results
python brunel2000/brunel_analysis.py ./outputs/sweep_default
```

**Notebook**: See `brunel2000/brunel2000_tutorial.ipynb` for a guided walkthrough.

**Detailed Docs**: See `brunel2000/README.md` for full parameter tables, model equations, and figure reproduction recipes.

---

### glif_net

**Directory**: `glif_net/`

A complete training pipeline for a **single-layer GLIF-RSNN** on the **12AX working memory task** — a classic cognitive neuroscience benchmark that requires maintaining context over time to produce correct responses.

**Features**:
- **SingleLayerGLIFRSNN**: Recurrent GLIF layer with adaptation currents, Dale's law constrained sparse connectivity, and readout layer
- **Poisson spike encoding**: Symbolic inputs converted to spike trains
- **Learnable input/output scaling**: `LearnableScale` modules with trajectory tracking
- **Checkpointing**: Full state save/load including memory reset values (`memories_rv`) and runtime memory snapshots (`memory_values`)
- **Gradient checkpointing & CPU offloading**: Memory-efficient training for large networks
- **Calibration**: Gradient-based calibration for robust inference

**Quick Start**:
```bash
# Run the tutorial notebook
jupyter notebook glif_net/notebooks/12ax_tutorial.ipynb

# Or run tests
pytest glif_net/tests/test_12ax.py
```

**Notebook**: `glif_net/notebooks/12ax_tutorial.ipynb` rebuilds the full pipeline cell-by-cell, keeping all btorch-critical steps explicit.

---

### mice_v1

**Directory**: `mice_v1/`

A **large-scale connectome-based simulation** framework for building biologically constrained neural network models. Built around a `FlyBrain` class that integrates real connectome data with btorch RSNNs.

**Features**:
- **Connectome integration**: Load real neuron populations and connectivity matrices
- **Heterogeneous synapses**: `HeterSynapsePSC` with receptor-type-specific time constants and E/I decomposition
- **Environment I/O layers**: `EnvInputLayer` (vision, wind/gravity, ascending neuron inputs) and `EnvOutputLayer` (descending neuron and MBON readouts)
- **Neuron embedding maps**: Modular input/output registration per neuron population
- **Full state tracing**: Track `neuron.v`, `neuron.Iasc`, `synapse.psc`, `synapse.psc_e`, `synapse.psc_i`
- **Experiment runner**: Unified training and simulation modes with visualization

**Quick Start**:
```bash
# Follow the tutorial notebook
jupyter notebook mice_v1/tutorial_companion_template.ipynb

# Or run the scripted tutorial
python mice_v1/run_tutorial_generate_and_sim_notebook.py
```

---

### Interactive Visualizer

**Directory**: `mice_v1/interactive_visualizer/`

A **Streamlit-based interactive application** for exploring SNN simulation results in real time. Visualize network topology, neuron dynamics, spike rasters, and synaptic currents through an interactive web interface.

**Features**:
- **Network graph**: Interactive vis.js visualization of connectome topology with E/I coloring
- **Neuron inspection**: Click any neuron to see its presynaptic partners, weights, and dynamics
- **Spike raster**: Real-time raster plots with population highlighting
- **Dynamics plots**: Membrane potential, PSC, EPSC, and IPSC traces over time
- **GLIF parameter integration**: Loads per-cell-class GLIF model parameters for physical voltage scaling

**Quick Start**:
```bash
cd mice_v1/interactive_visualizer
streamlit run snn_visualizer.py
```

Or use the launch script:
```bash
bash mice_v1/interactive_visualizer/run.sh
```

The app expects simulation outputs (parquet files) and network topology (`neurons.csv.gz`, `connections.csv.gz`) in the configured data directory.

---

## Installation

### Option 1: Conda (Recommended)

```bash
conda env create -f environment.yaml
conda activate btorch-binder
```

### Option 2: pip

```bash
pip install git+https://github.com/Criticality-Cognitive-Computation-Lab/btorch.git
pip install omegaconf streamlit pyvis networkx
# Install other dependencies as needed (see environment.yaml)
```

### Dependencies

Key dependencies (see `environment.yaml` for full list):
- Python ≥ 3.12
- PyTorch ≥ 2.8
- btorch (from source)
- OmegaConf (fork with enhanced features)
- Streamlit, pyvis, networkx (for visualizer)
- matplotlib, seaborn, scipy, pandas, numpy
- Jupyter (for notebooks)

---

## Quick Start

### 1. Learn the Basics
```bash
jupyter notebook basic_examples/01_single_neuron_dynamics.ipynb
```

### 2. Run a Balanced Network
```bash
python brunel2000/run_model_a.py
```

### 3. Train a GLIF Network
```bash
jupyter notebook glif_net/notebooks/12ax_tutorial.ipynb
```

### 4. Launch the Interactive Visualizer
```bash
bash mice_v1/interactive_visualizer/run.sh
```

---

## Core btorch Patterns

These patterns appear across all components:

### State Management
```python
from btorch.models import functional

# Initialize and reset all recurrent states
functional.init_net_state(model, batch_size=batch_size, device=device)
functional.reset_net(model, batch_size=batch_size, device=device)

# Initialize membrane voltages
functional.uniform_v_(model.neuron, low=-70.0, high=-40.0, set_reset_value=True)
```

### Simulation Context
```python
from btorch.models import environ

# Always wrap simulation in dt context
with environ.context(dt=0.1):
    spikes, states = model(input_current)
```

### Memory Tracing
```python
# States dict contains all registered memories
states = {
    "neuron.v": voltages,      # membrane potential
    "neuron.Iasc": asc,        # after-spike currents
    "synapse.psc": psc,        # postsynaptic currents
}
```

### Checkpointing (glif_net)
```python
from glif_net.src.checkpoint import save_checkpoint, load_checkpoint

save_checkpoint(model, optimizer, epoch, best_acc, path="checkpoint.pt")
epoch, best_acc = load_checkpoint("checkpoint.pt", model, optimizer)
```

---

## References

- **Brunel, N. (2000)**. Dynamics of sparsely connected networks of excitatory and inhibitory spiking neurons. *Journal of Computational Neuroscience*, 8(3), 183–208.
- **btorch**: [Criticality-Cognitive-Computation-Lab/btorch](https://github.com/Criticality-Cognitive-Computation-Lab/btorch)

---

*This repository is maintained by the Criticality-Cognitive-Computation Lab. For issues or contributions, please open a GitHub issue or pull request.*
