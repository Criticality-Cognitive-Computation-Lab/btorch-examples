"""Brunel 2000 single simulation runner (worker).

Run a single Brunel network simulation with given configuration.
Saves results including spike raster, firing rates, CV, and state classification.

Usage:
    python brunel_simulation.py \
        network.model_type=A \
        synapse.g=5.0 \
        synapse.nu_ext=20.0 \
        sim.duration=1000.0 \
        output.output_path=./outputs/run_01

The script uses OmegaConf for configuration and follows the worker pattern
for integration with parameter sweeps.
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from btorch.models import environ, functional, init

from brunel_config import (
    BrunelConfig,
    ModelANeuronConfig,
    load_config,
    compute_derived_params,
    get_model_params,
)
from brunel_model import create_model, BrunelNetwork


def run_simulation(
    cfg: BrunelConfig,
    save_results: bool = True,
    verbose: bool = True,
) -> dict:
    """Run single Brunel network simulation.

    Args:
        cfg: Configuration object
        save_results: Whether to save results to disk
        verbose: Whether to print progress

    Returns:
        results: Dictionary with simulation results and metrics
    """
    if verbose:
        model_type = "A" if isinstance(cfg.neuron, ModelANeuronConfig) else "B"
        print(f"\n{'=' * 60}")
        print(f"Brunel 2000 Simulation")
        print(f"{'=' * 60}")
        print(f"Model type: {model_type}")
        print(
            f"Neurons: {cfg.network.n_exc} E + {cfg.network.n_inh} I = {cfg.network.n_exc + cfg.network.n_inh}"
        )
        print(f"Duration: {cfg.sim.duration} ms, dt: {cfg.sim.dt} ms")
        print(f"Seed: {cfg.sim.seed}")

    # Set random seed
    torch.manual_seed(cfg.sim.seed)
    np.random.seed(cfg.sim.seed)

    # Setup device
    device = torch.device(cfg.sim.device)
    if verbose:
        print(f"Device: {device}")

    # Create model
    model = create_model(cfg).to(device)

    # Generate external input
    ext_input = model.generate_external_input(batch_size=1).to(device)

    if verbose:
        print(f"External input shape: {ext_input.shape}")

    # Reset and initialize model
    functional.reset_net_state(model, batch_size=1)
    init.uniform_v_(model.neuron, set_reset_value=True)

    # Run simulation
    if verbose:
        print("Running simulation...")

    with environ.context(dt=cfg.sim.dt):
        spikes, states = model(ext_input)

    if verbose:
        print("Simulation complete. Computing metrics...")

    # Compute metrics
    results = compute_metrics(cfg, model, spikes, states)

    if verbose:
        print(f"\nResults:")
        print(f"  E firing rate: {results['rate_e']:.2f} Hz")
        print(f"  I firing rate: {results['rate_i']:.2f} Hz")
        print(f"  CV (E): {results['cv_e']:.3f}")
        print(f"  CV (I): {results['cv_i']:.3f}")
        print(f"  State: {results['state_classification']}")

    # Save results
    if save_results:
        save_simulation_results(cfg, results, spikes, states)

    return results


def compute_metrics(
    cfg: BrunelConfig,
    model: BrunelNetwork,
    spikes: torch.Tensor,
    states: dict,
) -> dict:
    """Compute simulation metrics.

    Args:
        cfg: Configuration
        model: Network model
        spikes: Spike tensor [T, batch, n_neurons]
        states: Neuron states dict

    Returns:
        results: Dictionary with metrics
    """
    # Convert to numpy for analysis
    spikes_np = spikes.detach().cpu().numpy()

    # Remove batch dimension
    spikes_np = spikes_np[:, 0, :]  # [T, n_neurons]

    # Split into populations
    spikes_e = spikes_np[:, : cfg.network.n_exc]
    spikes_i = spikes_np[:, cfg.network.n_exc :]

    # Simulation duration (excluding warmup)
    warmup_steps = min(int(cfg.sim.warmup / cfg.sim.dt), len(spikes_e) - 1)
    duration_eff = cfg.sim.duration - warmup_steps * cfg.sim.dt

    # Ensure we have at least some data points
    if warmup_steps >= len(spikes_e):
        warmup_steps = max(0, len(spikes_e) // 10)  # Use 10% as warmup fallback
        duration_eff = cfg.sim.duration - warmup_steps * cfg.sim.dt

    # Firing rates (Hz)
    spikes_e_eff = spikes_e[warmup_steps:]
    spikes_i_eff = spikes_i[warmup_steps:]

    rate_e = spikes_e_eff.sum() / (cfg.network.n_exc * duration_eff / 1000.0)
    rate_i = spikes_i_eff.sum() / (cfg.network.n_inh * duration_eff / 1000.0)

    # CV of ISI
    cv_e = compute_cv_isi(spikes_e_eff)
    cv_i = compute_cv_isi(spikes_i_eff)

    # Population activity (instantaneous firing rate)
    pop_e = spikes_e_eff.sum(axis=1) / (cfg.network.n_exc * cfg.sim.dt / 1000.0)
    pop_i = spikes_i_eff.sum(axis=1) / (cfg.network.n_inh * cfg.sim.dt / 1000.0)

    # Detect oscillations
    osc_freq_e, osc_power_e = detect_oscillations(pop_e, cfg.sim.dt)
    osc_freq_i, osc_power_i = detect_oscillations(pop_i, cfg.sim.dt)

    # Synchrony index (population activity std / mean)
    sync_e = pop_e.std() / (pop_e.mean() + 1e-10)
    sync_i = pop_i.std() / (pop_i.mean() + 1e-10)

    # State classification
    state = classify_state(rate_e, rate_i, cv_e, cv_i, sync_e, sync_i, osc_power_e)

    # Get derived params for comparison with theory
    derived = compute_derived_params(cfg)
    params = get_model_params(cfg)

    model_type = "A" if isinstance(cfg.neuron, ModelANeuronConfig) else "B"
    results = {
        # Configuration
        "model_type": model_type,
        "g": params.get("g", params.get("g_e")),
        "nu_ext": params.get("nu_ext", params.get("nu_ext_e")),
        "nu_thr": derived["nu_thr"],
        "nu_ext_ratio": params.get("nu_ext", params.get("nu_ext_e"))
        / derived["nu_thr"],
        "delay": params["delay"],
        "J": params.get("J", params.get("J_e")),
        "c_exc": derived["c_exc"],
        "c_inh": derived["c_inh"],
        # Firing rates
        "rate_e": float(rate_e),
        "rate_i": float(rate_i),
        # CV of ISI
        "cv_e": float(cv_e),
        "cv_i": float(cv_i),
        # Oscillations
        "osc_freq_e": float(osc_freq_e),
        "osc_freq_i": float(osc_freq_i),
        "osc_power_e": float(osc_power_e),
        "osc_power_i": float(osc_power_i),
        # Synchrony
        "sync_e": float(sync_e),
        "sync_i": float(sync_i),
        # Classification
        "state_classification": state,
        # Raw data (for analysis)
        "spikes_e": spikes_e_eff,
        "spikes_i": spikes_i_eff,
        "pop_e": pop_e,
        "pop_i": pop_i,
    }

    return results


def compute_cv_isi(spikes: np.ndarray) -> float:
    """Compute coefficient of variation of inter-spike intervals.

    Args:
        spikes: Spike array [T, n_neurons]

    Returns:
        CV of ISI (averaged across neurons)
    """
    n_neurons = spikes.shape[1]
    cvs = []

    for i in range(n_neurons):
        # Find spike times
        spike_times = np.where(spikes[:, i] > 0)[0]

        if len(spike_times) > 2:
            # Compute ISIs
            isis = np.diff(spike_times)

            # CV = std / mean
            if isis.mean() > 0:
                cv = isis.std() / isis.mean()
                cvs.append(cv)

    if len(cvs) == 0:
        return 0.0

    return float(np.mean(cvs))


def detect_oscillations(
    pop_activity: np.ndarray,
    dt: float,
    max_freq: float = 500.0,
) -> tuple:
    """Detect oscillations in population activity using FFT.

    Args:
        pop_activity: Population activity time series
        dt: Time step (ms)
        max_freq: Maximum frequency to consider (Hz)

    Returns:
        peak_freq: Peak frequency (Hz)
        peak_power: Relative power at peak
    """
    # Handle empty array
    if len(pop_activity) == 0:
        return 0.0, 0.0

    # FFT
    fft = np.fft.rfft(pop_activity)
    freqs = np.fft.rfftfreq(len(pop_activity), d=dt / 1000.0)  # Hz
    power = np.abs(fft) ** 2

    # Find peak in relevant range
    valid_idx = freqs <= max_freq
    valid_freqs = freqs[valid_idx]
    valid_power = power[valid_idx]

    if len(valid_power) <= 1:
        return 0.0, 0.0

    # Peak frequency
    peak_idx = np.argmax(valid_power[1:]) + 1  # Exclude DC
    peak_freq = valid_freqs[peak_idx]
    peak_power = valid_power[peak_idx] / valid_power.sum()

    return peak_freq, peak_power


def classify_state(
    rate_e: float,
    rate_i: float,
    cv_e: float,
    cv_i: float,
    sync_e: float,
    sync_i: float,
    osc_power: float,
) -> str:
    """Classify network state based on metrics.

    States (from Brunel 2000):
        SR: Synchronous Regular - high rate, low CV, high synchrony
        AR: Asynchronous Regular - moderate rate, low CV, low synchrony
        AI: Asynchronous Irregular - low rate, CV ~ 1, low synchrony
        SI: Synchronous Irregular - low rate, CV ~ 1, high synchrony

    Args:
        rate_e: Excitatory firing rate (Hz)
        rate_i: Inhibitory firing rate (Hz)
        cv_e: CV of ISI for E population
        cv_i: CV of ISI for I population
        sync_e: Synchrony index for E population
        sync_i: Synchrony index for I population
        osc_power: Oscillation power

    Returns:
        state: State classification string
    """
    # Use average across populations
    rate = (rate_e + rate_i) / 2
    cv = (cv_e + cv_i) / 2
    sync = (sync_e + sync_i) / 2

    # Thresholds (approximate from paper)
    # High rate: > 50 Hz (near saturation)
    # Low CV: < 0.5 (regular firing)
    # High CV: > 0.8 (irregular firing, Poisson ~ 1)
    # High synchrony: power > threshold

    high_rate = rate > 50.0
    low_cv = cv < 0.5
    high_cv = cv > 0.8
    high_sync = osc_power > 0.1  # Relative power threshold

    if high_rate and low_cv:
        return "SR"  # Synchronous Regular
    elif not high_rate and low_cv and not high_sync:
        return "AR"  # Asynchronous Regular
    elif not high_rate and high_cv and not high_sync:
        return "AI"  # Asynchronous Irregular
    elif not high_rate and high_cv and high_sync:
        return "SI"  # Synchronous Irregular
    else:
        return "UNKNOWN"


def save_simulation_results(
    cfg: BrunelConfig,
    results: dict,
    spikes: torch.Tensor,
    states: dict,
):
    """Save simulation results to disk.

    Args:
        cfg: Configuration
        results: Metrics dictionary
        spikes: Spike tensor
        states: Neuron states
    """
    output_path = Path(cfg.output.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_path / "config.yaml"
    from omegaconf import OmegaConf

    OmegaConf.save(OmegaConf.structured(cfg), config_path)

    # Save metrics (without large arrays)
    metrics = {k: v for k, v in results.items() if not isinstance(v, np.ndarray)}
    metrics_path = output_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save spikes if requested
    if cfg.output.save_spikes:
        spikes_path = output_path / "spikes.npz"
        np.savez(
            spikes_path,
            spikes_e=results["spikes_e"],
            spikes_i=results["spikes_i"],
            pop_e=results["pop_e"],
            pop_i=results["pop_i"],
        )

    # Save full results
    results_path = output_path / "results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Results saved to: {output_path}")


def main():
    """Main entry point for single simulation."""
    # Load config with CLI overrides
    cfg, cli_cfg = load_config(return_cli=True)

    # Run simulation
    results = run_simulation(cfg, save_results=True, verbose=True)

    return results


if __name__ == "__main__":
    main()
