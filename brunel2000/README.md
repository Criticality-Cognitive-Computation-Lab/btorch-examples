# Brunel 2000 Tutorial: Sparsely Connected Networks with btorch + OmegaConf

This tutorial reproduces the classic Brunel 2000 paper "Dynamics of Sparsely Connected Networks of Excitatory and Inhibitory Spiking Neurons" using btorch RSNN simulation with OmegaConf for configuration management.

## Quick Start

```bash
# Run Model A with defaults
python run_model_a.py

# Run with custom parameters
python run_model_a.py synapse.g=5.0 synapse.nu_ext=20.0 sim.duration=1000.0

# Run Model B (heterogeneous E/I)
python run_model_b.py neuron.tau_i=10.0 synapse.g_e=6.0

# Parameter sweep for phase diagram
python brunel_sweep.py sweep.sweep_g=[3,4,5,6,7,8] sweep.sweep_nu_ext_ratio=[0.5,1,2,3,4]

# Analyze sweep results
python brunel_analysis.py ./outputs/sweep_default
```

## Paper Overview

Brunel 2000 analyzes the dynamics of sparsely connected networks of leaky integrate-and-fire (LIF) neurons. The paper identifies four distinct dynamical states:

| State | Name | Characteristics |
|-------|------|-----------------|
| SR | Synchronous Regular | High firing rates, low CV, strong synchrony |
| AR | Asynchronous Regular | Moderate rates, low CV, weak synchrony |
| AI | Asynchronous Irregular | Low rates, CV ≈ 1, weak synchrony |
| SI | Synchronous Irregular | Low rates, CV ≈ 1, oscillatory synchrony |

## Implementation

### File Structure

```
brunel2000/
├── brunel_config.py      # OmegaConf dataclass configuration (union types)
├── brunel_model.py       # btorch RSNN model (Model A + B)
├── brunel_simulation.py  # Single simulation runner
├── brunel_sweep.py       # Parameter sweep pipeline
├── brunel_analysis.py    # Visualization and analysis
├── run_model_a.py        # Quick runner for Model A
├── run_model_b.py        # Quick runner for Model B
└── README.md             # This file
```

### Model Selection (Dataclass Union Types)

Uses proper OmegaConf union types instead of string `model_type`:

```python
from brunel_config import (
    load_config,
    ModelANeuronConfig, ModelASynapseConfig,  # Model A
    ModelBNeuronConfig, ModelBSynapseConfig,  # Model B
)

# Model A: Identical E/I neurons (default)
cfg = load_config()  # Uses ModelA* configs by default

# Model B: Heterogeneous E/I neurons
cfg = load_config()
cfg.neuron = ModelBNeuronConfig(tau_e=20.0, tau_i=10.0)
cfg.synapse = ModelBSynapseConfig(g_e=6.0, J_i=0.15)
```

### Model A: Identical E/I Neurons

All neurons share the same parameters:
- Single membrane time constant `tau_m = 20 ms`
- Single EPSP amplitude `J = 0.1 mV`
- Single inhibitory strength `g` (IPSP = -g × J)
- Single external rate `nu_ext`
- Single delay `D` for all connections

**Key equation** (firing rate in AI state):
```
ν_0 = (ν_ext - ν_thr) / (g × γ - 1)
```
where `ν_thr = θ / (C_E × J × τ)` and `γ = N_I/N_E = 0.25`.

### Model B: Heterogeneous E/I Neurons

Different parameters for E and I populations:
- Different time constants: `tau_e` vs `tau_i`
- Different EPSP amplitudes: `J_e` vs `J_i`
- Different inhibitory strengths: `g_e` (I→E), `g_i` (I→I)
- Different external rates: `nu_ext_e`, `nu_ext_i`
- **Single delay D** for all connections (heterogeneous delays deferred)

## Configuration (OmegaConf)

### Dataclass-First Pattern

All configuration is defined in Python dataclasses (not YAML):

```python
@dataclass
class SynapseConfig:
    # Model A
    J: float = 0.1      # Model A only
    g: float = 5.0
    nu_ext: float = 10.0
    delay: float = 1.5
```

```python
@dataclass
class ModelBSynapseConfig:
    """Model B: heterogeneous synaptic parameters"""
    J_e: float = 0.1    # EPSP for E neurons
    J_i: float = 0.1    # EPSP for I neurons
    g_e: float = 5.0    # I→E strength
    g_i: float = 5.0    # I→I strength
    nu_ext_e: float = 10.0
    nu_ext_i: float = 10.0
    delay: float = 1.5
```

```python
# Union type for synapse configs
SynapseConfig = Union[ModelASynapseConfig, ModelBSynapseConfig]
```

### CLI Overrides

Override any parameter via command line:

```bash
python run_model_a.py \
    synapse.g=6.0 \
    synapse.nu_ext=25.0 \
    synapse.delay=2.0 \
    neuron.tau_m=15.0 \
    sim.duration=2000.0 \
    output.output_path=./outputs/experiment_1
```

### Union Type Pattern (Model A vs B)

Uses dataclass union types instead of string `model_type`:

```python
from dataclasses import dataclass, field
from typing import Union

@dataclass
class ModelANeuronConfig:
    tau_m: float = 20.0
    v_thresh: float = 20.0
    v_reset: float = 10.0
    tau_rp: float = 2.0

@dataclass
class ModelBNeuronConfig:
    tau_e: float = 20.0
    tau_i: float = 20.0
    v_thresh: float = 20.0
    v_reset: float = 10.0
    tau_rp: float = 2.0

# Union type
NeuronConfig = Union[ModelANeuronConfig, ModelBNeuronConfig]

@dataclass
class BrunelConfig:
    neuron: NeuronConfig = field(default_factory=ModelANeuronConfig)
    synapse: SynapseConfig = field(default_factory=ModelASynapseConfig)
    # ... other configs
```

## btorch RSNN Patterns

### 1. Model Creation

```python
from brunel_model import create_model

cfg = load_config()  # No class argument needed
model = create_model(cfg)  # Handles initialization
```

### 2. Simulation Loop

```python
from btorch.models import environ, functional

# Reset states before each run
functional.reset_net_state(model, batch_size=1)
functional.uniform_v_(model.neuron, set_reset_value=True)

# Always use dt context manager
with environ.context(dt=cfg.sim.dt):
    spikes, states = model(external_input)
```

### 3. Network Architecture

```python
class BrunelNetwork(nn.Module):
    def __init__(self, cfg):
        # Sparse connectivity
        self.conn = SparseConn(conn=weight_matrix)
        
        # Neurons (Model A: single type, Model B: separate E/I)
        self.neuron = GLIF3(n_neuron=N, tau=cfg.neuron.tau_m, ...)
        
        # Custom RNN with delay handling
        self.rnn = _BrunelRNN(neuron, conn, delay_steps)
```

## Parameter Sweep

### Worker/Launcher Pattern

**Worker** (`brunel_simulation.py`): Runs single simulation
**Launcher** (`brunel_sweep.py`): Distributes work across parameters

### Grid Search

```python
# Generate all combinations
for g in cfg.sweep.sweep_g:
    for ratio in cfg.sweep.sweep_nu_ext_ratio:
        trial_cfg = deepcopy(cfg)
        trial_cfg.synapse.g = g
        trial_cfg.synapse.nu_ext = ratio * nu_thr
        # Run trial...
```

### Parallel Execution

```bash
python brunel_sweep.py sweep.max_workers=4
```

Uses `ProcessPoolExecutor` to run simulations in parallel.

### Resume Capability

```bash
python brunel_sweep.py sweep.resume=true
```

Skips completed runs based on existing `metrics.json` files.

## Analysis

### Phase Diagram

Visualize the four dynamical states in (g, ν_ext/ν_thr) space:

```python
from brunel_analysis import plot_phase_diagram

results = load_sweep_results("./outputs/sweep_a")
plot_phase_diagram(results, save_path="phase_diagram.png")
```

### Firing Rate Comparison

Compare simulation with theoretical predictions:

```python
from brunel_analysis import plot_firing_rate_comparison

plot_firing_rate_comparison(results, save_path="firing_rates.png")
```

### Spike Raster

Visualize spike times (similar to Fig 8 in paper):

```python
from brunel_analysis import plot_raster

plot_raster(spikes_e, spikes_i, dt=0.1, save_path="raster.png")
```

## Examples

### Example 1: Single Simulation

```bash
python run_model_a.py \
    synapse.g=5.0 \
    synapse.nu_ext=20.0 \
    sim.duration=1000.0 \
    output.output_path=./outputs/run_01
```

Results saved to `./outputs/run_01/`:
- `metrics.json`: Firing rates, CV, state classification
- `spikes.npz`: Spike raster data
- `config.yaml`: Full configuration

### Example 2: Phase Diagram Sweep

```bash
python brunel_sweep.py \
    sweep.sweep_g="[3.0, 4.0, 5.0, 6.0, 7.0, 8.0]" \
    sweep.sweep_nu_ext_ratio="[0.5, 1.0, 1.5, 2.0, 3.0, 4.0]" \
    sweep.max_workers=4 \
    output.output_path=./outputs/phase_diagram

# Generate plots
python brunel_analysis.py ./outputs/phase_diagram
```

### Example 3: Model B - Fast Inhibition

```bash
python run_model_b.py \
    neuron.tau_i=10.0 \
    synapse.g_e=6.0 \
    synapse.J_i=0.15 \
    sim.duration=1000.0 \
    output.output_path=./outputs/model_b_fast
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `network.n_exc` | 10000 | Number of excitatory neurons |
| `network.n_inh` | 2500 | Number of inhibitory neurons |
| `network.conn_prob` | 0.1 | Connection probability ε |
| `neuron.tau_m` | 20.0 | Membrane time constant (ms) |
| `neuron.v_thresh` | 20.0 | Threshold potential (mV) |
| `synapse.J` | 0.1 | EPSP amplitude (mV) |
| `synapse.g` | 5.0 | Inhibitory strength ratio |
| `synapse.nu_ext` | 10.0 | External Poisson rate (Hz) |
| `synapse.delay` | 1.5 | Synaptic delay (ms) |
| `sim.dt` | 0.1 | Time step (ms) |
| `sim.duration` | 1000.0 | Simulation duration (ms) |

## Reproducing Paper Figures

### Figure 1: Bifurcation Diagram and CV

Sweep over g at fixed ν_ext ratios to see:
- Bifurcation between high (H) and low (L) activity states
- CV jump at g ≈ 4 (balanced point)

```bash
python brunel_sweep.py \
    sweep.sweep_g="[3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0]" \
    synapse.nu_ext=20.0

python brunel_analysis.py ./outputs/sweep_default
# Check cv_analysis.png
```

### Figure 2: Phase Diagram

Full phase diagram in (g, ν_ext/ν_thr) space:

```bash
python brunel_sweep.py \
    sweep.sweep_g="[3.0, 4.0, 5.0, 6.0, 7.0, 8.0]" \
    sweep.sweep_nu_ext_ratio="[0.5, 1.0, 1.5, 2.0, 3.0, 4.0]"

python brunel_analysis.py ./outputs/sweep_default
# Check phase_diagram.png
```

### Figure 8: Example Raster Plots

Four parameter sets showing different states:

```bash
# SR state (g < 4, high nu_ext)
python run_model_a.py synapse.g=3.0 synapse.nu_ext=30.0

# AI state (g > 4, moderate nu_ext)
python run_model_a.py synapse.g=5.0 synapse.nu_ext=20.0

# SI state (g > 4, nu_ext ~ nu_thr)
python run_model_a.py synapse.g=5.0 synapse.nu_ext=12.0
```

## Notes

### Model B Limitations

- **Heterogeneous delays** (D_EE, D_EI, D_IE, D_II) are not yet supported
- When btorch adds native delay support, Model B will be updated
- For now, single constant delay D is used for all connections

### Performance Tips

1. **Smaller networks for testing**: Use `network.n_exc=1000 network.n_inh=250`
2. **Shorter simulations**: Use `sim.duration=500.0` for quick tests
3. **GPU acceleration**: Set `sim.device=cuda` (if available)
4. **Parallel sweeps**: Use `sweep.max_workers=4` or higher

### References

Brunel, N. (2000). Dynamics of sparsely connected networks of excitatory and inhibitory spiking neurons. *Journal of Computational Neuroscience*, 8(3), 183-208.
