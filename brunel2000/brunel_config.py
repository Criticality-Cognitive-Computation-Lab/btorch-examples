"""Brunel 2000 configuration using OmegaConf dataclass-first pattern.

This module defines all configuration parameters for reproducing the Brunel 2000 paper.
Uses proper dataclass union types for Model A vs Model B selection.

Usage:
    # Model A (identical E/I neurons)
    cfg = load_config(BrunelConfig)

    # Model B (heterogeneous E/I neurons)
    cfg = load_config(BrunelConfig)
    cfg.neuron = ModelBNeuronConfig(tau_e=20.0, tau_i=10.0)
    cfg.synapse = ModelBSynapseConfig(g_e=6.0, J_i=0.15)

    # Override via CLI:
    python script.py synapse.g=6.0 sim.duration=2000.0
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Any
from omegaconf import OmegaConf, MISSING


@dataclass
class NetworkConfig:
    """Network architecture parameters."""

    n_exc: int = 10000  # Excitatory neurons (80%)
    n_inh: int = 2500  # Inhibitory neurons (20%)
    conn_prob: float = 0.1  # Connection probability ε

    @property
    def n_neurons(self) -> int:
        return self.n_exc + self.n_inh


@dataclass
class CommonNeuronConfig:
    """Common neuron parameters for all models."""

    v_thresh: float = 20.0  # Threshold θ (mV)
    v_reset: float = 10.0  # Reset potential V_r (mV)
    tau_rp: float = 2.0  # Refractory period (ms)


@dataclass
class ModelANeuronConfig(CommonNeuronConfig):
    """Model A: Identical E/I neurons."""

    tau_m: float = 20.0  # Membrane time constant τ (ms)


@dataclass
class ModelBNeuronConfig(CommonNeuronConfig):
    """Model B: Different time constants for E/I populations."""

    tau_e: float = 20.0  # Excitatory neuron τ (ms)
    tau_i: float = 20.0  # Inhibitory neuron τ (ms)


# Union type for neuron configs
NeuronConfig = Union[ModelANeuronConfig, ModelBNeuronConfig]


@dataclass
class CommonSynapseConfig:
    """Common synaptic parameters for all models."""

    delay: float = 1.5  # Synaptic delay D (ms) - single constant delay


@dataclass
class ModelASynapseConfig(CommonSynapseConfig):
    """Model A: Identical synaptic parameters."""

    J: float = 0.1  # EPSP amplitude (mV)
    g: float = 5.0  # Relative inhibitory strength
    nu_ext: float = 10.0  # External Poisson rate (Hz)


@dataclass
class ModelBSynapseConfig(CommonSynapseConfig):
    """Model B: Heterogeneous synaptic parameters."""

    J_e: float = 0.1  # EPSP for excitatory neurons (mV)
    J_i: float = 0.1  # EPSP for inhibitory neurons (mV)
    g_e: float = 5.0  # I→E relative inhibitory strength
    g_i: float = 5.0  # I→I relative inhibitory strength
    nu_ext_e: float = 10.0  # External rate to E population (Hz)
    nu_ext_i: float = 10.0  # External rate to I population (Hz)


# Union type for synapse configs
SynapseConfig = Union[ModelASynapseConfig, ModelBSynapseConfig]


@dataclass
class SimConfig:
    """Simulation parameters."""

    dt: float = 0.1  # Time step (ms)
    duration: float = 1000.0  # Simulation duration (ms)
    warmup: float = 100.0  # Warmup period to discard (ms)
    seed: int = 42  # Random seed
    device: str = "cpu"  # "cpu" or "cuda"


@dataclass
class OutputConfig:
    """Output configuration."""

    output_path: Path = Path("./outputs")
    save_spikes: bool = True  # Save spike raster data
    save_states: bool = False  # Save neuron states (memory intensive)
    save_plots: bool = True  # Generate and save plots


@dataclass
class SweepConfig:
    """Parameter sweep configuration."""

    # Model A sweep candidates
    sweep_g: list[float] = field(default_factory=lambda: [3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    sweep_nu_ext_ratio: list[float] = field(
        default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    )

    # Model B sweep candidates
    sweep_g_e: list[float] = field(default_factory=lambda: [4.0, 5.0, 6.0, 7.0, 8.0])
    sweep_j_i: list[float] = field(default_factory=lambda: [0.05, 0.1, 0.15, 0.2])

    # Sweep execution
    max_workers: int = 4
    resume: bool = True  # Skip completed runs


@dataclass
class BrunelConfig:
    """Top-level configuration for Brunel 2000 simulation.

    Uses OmegaConf composition pattern with union types for model selection.
    Default is Model A (identical E/I neurons).

    CLI overrides use dot notation: synapse.g=6.0 neuron.tau_m=15.0

    To use Model B via CLI, specify the full type:
        neuron="ModelBNeuronConfig(tau_i=10.0)" synapse="ModelBSynapseConfig(g_e=6.0)"
    """

    network: NetworkConfig = field(default_factory=NetworkConfig)
    neuron: NeuronConfig = field(default_factory=ModelANeuronConfig)
    synapse: SynapseConfig = field(default_factory=ModelASynapseConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)


def load_config(return_cli: bool = False) -> BrunelConfig | tuple[BrunelConfig, Any]:
    """Load configuration from dataclass with CLI overrides.

    Args:
        return_cli: If True, also return the raw CLI config for forwarding

    Returns:
        cfg: The populated BrunelConfig instance
        cli_cfg: (optional) The raw CLI OmegaConf object

    Example:
        >>> cfg = load_config()
        >>> print(cfg.synapse.J)  # Default or CLI override

        >>> cfg, cli = load_config(return_cli=True)
        >>> # Forward CLI overrides to worker processes
    """
    from typing import Any

    # Create default config from dataclass
    defaults = OmegaConf.structured(BrunelConfig())

    # Parse CLI overrides (e.g., "synapse.g=6.0")
    cli_cfg = OmegaConf.from_cli()

    # Merge CLI overrides into defaults
    cfg = OmegaConf.unsafe_merge(defaults, cli_cfg)

    # Convert back to dataclass instance
    cfg_obj = OmegaConf.to_object(cfg)
    assert isinstance(cfg_obj, BrunelConfig)

    if return_cli:
        return cfg_obj, cli_cfg
    return cfg_obj


def is_model_a(cfg: BrunelConfig) -> bool:
    """Check if config is using Model A (identical neurons)."""
    return isinstance(cfg.neuron, ModelANeuronConfig) and isinstance(
        cfg.synapse, ModelASynapseConfig
    )


def is_model_b(cfg: BrunelConfig) -> bool:
    """Check if config is using Model B (heterogeneous neurons)."""
    return isinstance(cfg.neuron, ModelBNeuronConfig) and isinstance(
        cfg.synapse, ModelBSynapseConfig
    )


def compute_derived_params(cfg: BrunelConfig) -> dict:
    """Compute derived parameters from base config.

    Returns dict with:
        - c_exc: Number of E connections per neuron
        - c_inh: Number of I connections per neuron
        - c_ext: Number of external connections
        - nu_thr: Threshold frequency (Hz)
        - j_exc: Excitatory weight (mV)
        - j_inh: Inhibitory weight (mV, negative)
        - gamma: N_I/N_E ratio (0.25)
    """
    net = cfg.network
    neu = cfg.neuron
    syn = cfg.synapse

    # Connection counts
    c_exc = int(net.conn_prob * net.n_exc)  # C_E
    c_inh = int(net.conn_prob * net.n_inh)  # C_I
    c_ext = c_exc  # External connections = recurrent E connections

    # Gamma ratio
    gamma = net.n_inh / net.n_exc  # Should be 0.25 for 80/20 split

    # Get parameters based on model type
    if isinstance(neu, ModelANeuronConfig) and isinstance(syn, ModelASynapseConfig):
        # Model A: identical parameters
        tau_m = neu.tau_m
        j_exc = syn.J
        j_inh = -syn.g * syn.J
        nu_ext = syn.nu_ext
    elif isinstance(neu, ModelBNeuronConfig) and isinstance(syn, ModelBSynapseConfig):
        # Model B: use excitatory neuron parameters as reference
        tau_m = neu.tau_e
        j_exc = syn.J_e
        j_inh = -syn.g_e * syn.J_e
        nu_ext = syn.nu_ext_e
    else:
        raise ValueError(
            "Neuron and synapse config types must match (Model A or Model B)"
        )

    # Threshold frequency: ν_thr = θ / (C_E * J * τ)
    nu_thr = neu.v_thresh / (c_exc * j_exc * tau_m / 1000)  # Convert τ to seconds

    return {
        "c_exc": c_exc,
        "c_inh": c_inh,
        "c_ext": c_ext,
        "gamma": gamma,
        "nu_thr": nu_thr,
        "j_exc": j_exc,
        "j_inh": j_inh,
        "tau_m": tau_m,
        "nu_ext": nu_ext,
    }


def get_model_params(cfg: BrunelConfig) -> dict:
    """Get unified parameter dict from config.

    Returns model-specific parameters in a unified format regardless of model type.
    """
    net = cfg.network
    neu = cfg.neuron
    syn = cfg.synapse

    if isinstance(neu, ModelANeuronConfig) and isinstance(syn, ModelASynapseConfig):
        return {
            "tau_e": neu.tau_m,
            "tau_i": neu.tau_m,
            "J_e": syn.J,
            "J_i": syn.J,
            "g_e": syn.g,
            "g_i": syn.g,
            "nu_ext_e": syn.nu_ext,
            "nu_ext_i": syn.nu_ext,
            "delay": syn.delay,
        }
    elif isinstance(neu, ModelBNeuronConfig) and isinstance(syn, ModelBSynapseConfig):
        return {
            "tau_e": neu.tau_e,
            "tau_i": neu.tau_i,
            "J_e": syn.J_e,
            "J_i": syn.J_i,
            "g_e": syn.g_e,
            "g_i": syn.g_i,
            "nu_ext_e": syn.nu_ext_e,
            "nu_ext_i": syn.nu_ext_i,
            "delay": syn.delay,
        }
    else:
        raise ValueError(
            "Neuron and synapse config types must match (Model A or Model B)"
        )


if __name__ == "__main__":
    # Test config loading
    cfg = load_config()

    print("Configuration:")
    print(
        f"  Model type: {'A' if is_model_a(cfg) else 'B' if is_model_b(cfg) else 'Unknown'}"
    )
    print(f"  Neurons: {cfg.network.n_exc} E, {cfg.network.n_inh} I")
    print(f"  Connection prob: {cfg.network.conn_prob}")

    if isinstance(cfg.neuron, ModelANeuronConfig):
        print(f"  tau_m: {cfg.neuron.tau_m} ms (Model A)")
    else:
        print(f"  tau_e: {cfg.neuron.tau_e} ms, tau_i: {cfg.neuron.tau_i} ms (Model B)")

    if isinstance(cfg.synapse, ModelASynapseConfig):
        print(f"  J: {cfg.synapse.J} mV, g: {cfg.synapse.g} (Model A)")
    else:
        print(f"  J_e: {cfg.synapse.J_e} mV, J_i: {cfg.synapse.J_i} mV (Model B)")

    print(f"  dt: {cfg.sim.dt} ms")
    print(f"  duration: {cfg.sim.duration} ms")

    # Test derived params
    derived = compute_derived_params(cfg)
    print("\nDerived parameters:")
    print(f"  C_E: {derived['c_exc']}")
    print(f"  C_I: {derived['c_inh']}")
    print(f"  C_ext: {derived['c_ext']}")
    print(f"  gamma: {derived['gamma']:.2f}")
    print(f"  nu_thr: {derived['nu_thr']:.2f} Hz")
