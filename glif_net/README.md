# GLIF RSNN Tutorials

Clean tutorial implementations for training single-layer GLIF (Generalized Leaky Integrate-and-Fire) recurrent spiking neural networks on multiple tasks.

## Features

- **Single-layer GLIF RSNN** with partial adaptation support (`n_adapt` parameter)
- **No YAML configs** - pure Python dataclass configuration with CLI overrides
- **Four benchmark tasks**: Sequential MNIST (paper scheme), Poisson MNIST, SHD, Speech Commands
- **Proper btorch patterns**: state management, checkpointing with `memories_rv`, `environ.context(dt=...)`
- **Loss**: CE + voltage regularization + firing rate regularization (no ASC current loss)
- **Unified SLURM array job** for running all tasks

## Quick Start

```bash
# Sequential MNIST with paper defaults (220 neurons, 100 adaptive)
python -m glif_net.run_tutorial task=seqmnist

# Override specific parameters
python -m glif_net.run_tutorial task=seqmnist n_adapt=50 asc_amp=-1.0

# SHD with custom settings
python -m glif_net.run_tutorial task=shd n_neuron=256 n_adapt=128

# All tasks via SLURM
sbatch glif_net/run_all_slurm.sh

# Single task via SLURM
sbatch --array=2 glif_net/run_all_slurm.sh  # Just seqmnist
```

## Directory Structure

```
glif_net/
├── src/
│   ├── model.py           # SingleLayerGLIFRSNN with partial adaptation
│   ├── loss.py            # CE + voltage + rate loss
│   ├── checkpoint.py      # Save/load with memories_rv
│   ├── calibration.py     # I/O scale calibration
│   ├── viz.py             # Visualization helpers
│   └── gru_baseline.py    # GRU comparison model
├── tutorials/
│   ├── seqmnist.py        # Sequential MNIST (paper scheme)
│   ├── poisson_mnist.py   # MNIST with Poisson encoding
│   ├── shd.py             # Spiking Heidelberg Digits
│   └── speech_command.py  # Google Speech Commands
├── run_tutorial.py        # Main entry point
├── run_all_slurm.sh       # SLURM array job script
└── README.md
```

## Configuration

All configuration is done via Python dataclasses with CLI overrides:

```python
# Key parameters
n_neuron: int = 256          # Number of recurrent neurons
n_adapt: int = 0             # Neurons with adaptation (0=all, -1=half, N=first N)
asc_amp: float = -0.2        # Adaptation strength
response_window: float = 0.8 # Fraction of timesteps for readout
readout_tau: float = 20.0    # Low-pass filter for readout (ms)
```

### Task-Specific Defaults

Each task module provides `get_task_defaults()`:

**Sequential MNIST (paper scheme)**:
- 220 neurons, 100 with SFA
- asc_amp=-1.8, tau_adapt=700ms
- batch_size=256, lr=0.01, decay every 2500 steps
- 80 input neurons with threshold encoding

## Implementation Details

### Partial Adaptation

The `n_adapt` parameter controls how many neurons have adaptation:
- `n_adapt=0`: All neurons adapt (default `asc_amp`)
- `n_adapt=-1`: Half adapt
- `n_adapt=N`: First N neurons adapt, rest don't

### Threshold Encoding (SeqMNIST)

Per paper: 80 input neurons with thresholds evenly spaced in [0,1]. Neuron i fires when pixel value crosses threshold[i] from previous to current pixel.

### Checkpointing

Saves both `state_dict()` and `memories_rv` (memory reset values) required by btorch stateful neurons:

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "memories_rv": functional.named_memory_reset_values(model),
    ...
}
```

### TensorBoard Logging

| Category | Tag |
|----------|-----|
| Metrics | train/loss, train/acc, test/loss, test/acc |
| Loss | loss/ce, loss/voltage, loss/rate |
| Stats | stats/firing_rate_hz |
| Scales | scales/input_scale, scales/output_scale |

## Requirements

```bash
# Core dependencies (from btorch environment)
torch
torchvision
torchaudio
btorch

# Optional
tonic        # For SHD dataset
matplotlib   # For visualization
```

## Citation

This implementation follows the GLIF paper scheme for Sequential MNIST:
- 220 recurrent neurons with 100 adaptive
- Threshold-based input encoding (80 neurons)
- Low-pass filter readout (tau=20ms)
