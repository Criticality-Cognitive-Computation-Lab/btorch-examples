# GLIF RSNN Tutorials: Implementation Summary

## Project Overview

A clean, educational tutorial suite for training Generalized Leaky Integrate-and-Fire (GLIF) recurrent spiking neural networks using the btorch framework. Implements Sequential MNIST with partial adaptation support, plus three additional benchmark tasks.

**Key Achievement:** Single-layer GLIF RSNN with partial adaptation neurons (n_adapt parameter) following the Sequential MNIST paper scheme (220 neurons, 100 adaptive).

---

## What This Tutorial Teaches (First Principles)

### 1. Stateful Neuron Management (btorch Core Pattern)
Spiking neural networks maintain internal state (membrane voltage, synaptic currents) across timesteps. btorch handles this through:
- **Memory buffers**: Dynamic state not stored in `state_dict()`
- **Reset values**: Deterministic initialization per batch via `uniform_v_(set_reset_value=True)`
- **Context management**: All forward passes wrapped in `environ.context(dt=...)`

**Critical insight:** Stateful neurons require explicit state reset before each batch to prevent gradient flow across training examples.

### 2. Partial Adaptation Architecture
Biological neurons exhibit heterogeneous adaptation properties. We model this via:
- **n_adapt parameter**: Controls how many neurons have spike-frequency adaptation (SFA)
- **Per-neuron asc_amps**: First n_adapt neurons receive `asc_amp`, others receive 0
- **Paper configuration**: Sequential MNIST uses 220 neurons with 100 adaptive (asc_amp=-1.8, tau=700ms)

**Biological motivation:** Not all neurons adapt equally; partial adaptation matches biological reality and improves task performance.

### 3. Regularization Calibration
Two complementary approaches:
- **I/O calibration**: `LearnableScale` modules normalize input/output ranges for stable training
- **Gradient calibration**: Match regularization gradient magnitudes to ~10% of task loss gradients

**Why it matters:** Unbalanced regularization either dominates training (too strong) or has no effect (too weak).

### 4. Checkpointing for Stateful Models
Standard PyTorch checkpointing is insufficient:
- `state_dict()` excludes dynamic buffers (voltage, synaptic currents)
- Must save/restore `memories_rv` (memory reset values) separately
- `functional.named_memory_reset_values()` and `functional.set_memory_reset_values()` handle this

---

## Implementation Architecture

### Source Code (`glif_net/src/`)

| File | Purpose | Key Components |
|------|---------|----------------|
| `model.py` | Core GLIF RSNN | `SingleLayerGLIFRSNN`, E/I connectivity via `build_sparse_mat()`, partial adaptation via `asc_amps` array, `LearnableScale` I/O |
| `loss.py` | Combined loss | CE + voltage regularization + firing rate regularization (NO ASC current loss per requirements) |
| `checkpoint.py` | Save/load utilities | Handles `state_dict()` + `memories_rv` for proper state recovery |
| `calibration.py` | I/O scale calibration | `LearnableScale` adjustment based on data statistics |
| `calibration_grad.py` | Gradient calibration | Matches reg gradient norms to ~10% of task gradients |
| `viz.py` | Visualization | Spike raster, voltage traces, firing rate statistics |
| `gru_baseline.py` | Comparison model | GRU with same readout architecture for baseline comparison |

### Task Modules (`glif_net/tutorials/`)

| Task | Input Encoding | Paper Config | Output Classes |
|------|---------------|--------------|----------------|
| **seqmnist** | Threshold-based (80 neurons, thresholds in [0,1]) | 220 neurons, 100 adaptive, batch=256, lr=0.01 | 10 |
| **shd** | Pre-spiking (700 channels) | 100 timesteps | 20 |
| **speech_command** | MFCC + threshold encoding | 160 timesteps, 20 MFCCs | 35 |
| **poisson_mnist** | Poisson rate encoding | 100 timesteps, max 100 Hz | 10 |

### Notebooks (`glif_net/notebooks/`)

| Notebook | Focus | Key Demonstrations |
|----------|-------|-------------------|
| `01_quickstart.ipynb` | btorch fundamentals | State init/reset, `environ.context(dt=...)`, training loop, checkpointing, raster/traces plots |
| `seqmnist.ipynb` | Paper implementation | Threshold encoding visualization, 220/100 neuron configuration, training dynamics |
| `shd.ipynb` | Event-based input | Tonic dataset integration, cochlear spike visualization |
| `checkpointing.ipynb` | State recovery | Save/load mechanics, `memories_rv` verification, resuming training |
| `grad_calibration.ipynb` | Advanced regularization | Gradient norm matching, loss component balancing |

### Entry Points

- **`run_tutorial.py`**: Unified CLI entry (`python -m glif_net.run_tutorial task=seqmnist n_neuron=220`)
- **`run_all_slurm.sh`**: Array job for all 4 tasks (SBATCH array=0-3)

---

## Configuration Design

**No YAML files** - Pure Python dataclasses with CLI overrides per omegaconf-config skill:

```python
@dataclass
class Config:
    task: str = "seqmnist"
    n_neuron: int = 256
    n_adapt: int = 0  # 0=all, -1=half, N=first N
    asc_amp: float = -0.2
    # ... all defaults in Python
```

**Priority**: CLI arguments > Task defaults > Base defaults

Each task module provides `get_task_defaults()` returning paper-specific overrides.

---

## Technical Specifications

### GLIF3 Neuron Parameters (Sequential MNIST Paper)
- **n_neuron**: 220 (80% E, 20% I via `build_sparse_mat()`)
- **n_adapt**: 100 neurons with SFA
- **asc_amp**: -1.8 mV (adaptation strength)
- **tau_adapt**: 700 ms
- **v_threshold**: -10 mV
- **tau_mem**: 20 ms, **tau_syn**: 5 ms

### Training Configuration
- **Batch size**: 256
- **Learning rate**: 0.01 with StepLR (decay 0.8 every 2500 steps)
- **Loss**: CE (weight=1.0) + voltage (weight=0.0, not used) + rate (weight=0.1)
- **Readout**: Low-pass filter (tau=20ms) on last 80% of timesteps

### Key Dependencies
- btorch (neuron models, synapses, RNN wrappers)
- torch, torchvision, torchaudio
- tonic (for SHD dataset)
- matplotlib (visualization)
- tensorboard (logging)

---

## File Structure

```
glif_net/
├── src/
│   ├── __init__.py
│   ├── model.py                  # SingleLayerGLIFRSNN
│   ├── loss.py                   # CombinedLoss (CE + voltage + rate)
│   ├── checkpoint.py             # save_checkpoint(), load_checkpoint()
│   ├── calibration.py            # calibrate_io_scales()
│   ├── calibration_grad.py       # calibrate_regularization_weights()
│   ├── viz.py                    # visualize_spike_raster(), etc.
│   └── gru_baseline.py           # GRU comparison
├── tutorials/
│   ├── __init__.py
│   ├── seqmnist.py               # Sequential MNIST (paper scheme)
│   ├── shd.py                    # Spiking Heidelberg Digits
│   ├── speech_command.py         # Google Speech Commands
│   └── poisson_mnist.py          # MNIST with Poisson encoding
├── notebooks/
│   ├── 01_quickstart.ipynb       # btorch fundamentals
│   ├── seqmnist.ipynb            # Paper implementation
│   ├── shd.ipynb                 # SHD tutorial
│   ├── checkpointing.ipynb       # State management
│   └── grad_calibration.ipynb    # Gradient calibration
├── run_tutorial.py               # Main entry point
├── run_all_slurm.sh              # SLURM array job
└── README.md                     # Documentation
```

---

## Usage Examples

```bash
# Sequential MNIST with paper defaults
python -m glif_net.run_tutorial task=seqmnist

# Custom configuration
python -m glif_net.run_tutorial task=seqmnist n_adapt=50 asc_amp=-1.0

# All tasks via SLURM
sbatch glif_net/run_all_slurm.sh

# Single task via SLURM
sbatch --array=2 glif_net/run_all_slurm.sh  # Just seqmnist
```

---

## Key Research Insights

1. **Partial adaptation** (n_adapt < n_neuron) matches biological heterogeneity and improves generalization
2. **Threshold encoding** for Sequential MNIST (80 thresholds) outperforms direct pixel input
3. **Low-pass filter readout** (tau=20ms) provides stable classification from spike trains
4. **Gradient calibration** prevents regularization from overwhelming task learning

---

## Deliverables

- **Source**: 8 Python modules (model, loss, checkpoint, calibration ×2, viz, baseline, tasks ×4)
- **Notebooks**: 5 tutorial notebooks covering fundamentals to advanced calibration
- **Scripts**: Unified CLI entry + SLURM array job
- **Documentation**: README + inline docstrings

**Total**: ~2000 lines of documented Python code + notebook tutorials
