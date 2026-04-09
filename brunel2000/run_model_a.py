#!/usr/bin/env python
"""Quick runner for Model A (identical E/I neurons).

This is the tutorial entry point - demonstrates basic btorch RSNN
simulation with OmegaConf configuration.

Usage:
    # Basic run with defaults (Model A)
    python run_model_a.py

    # Custom parameters
    python run_model_a.py \
        synapse.g=5.0 \
        synapse.nu_ext=20.0 \
        synapse.delay=1.5 \
        sim.duration=1000.0 \
        output.output_path=./outputs/my_run

    # Smaller network for testing
    python run_model_a.py \
        network.n_exc=1000 \
        network.n_inh=250 \
        sim.duration=500.0

Model A Parameters:
    - synapse.g: Relative inhibitory strength (default: 5.0)
    - synapse.nu_ext: External Poisson rate in Hz (default: 10.0)
    - synapse.J: EPSP amplitude in mV (default: 0.1)
    - synapse.delay: Synaptic delay in ms (default: 1.5)
    - neuron.tau_m: Membrane time constant in ms (default: 20.0)
    - sim.duration: Simulation duration in ms (default: 1000.0)
"""

from brunel_config import load_config
from brunel_simulation import run_simulation


def main():
    """Run Model A simulation."""
    # Load config - CLI overrides are automatically parsed
    # Example: python run_model_a.py synapse.g=6.0 sim.duration=2000.0
    cfg = load_config()

    print("=" * 60)
    print("Brunel 2000 - Model A (Identical E/I Neurons)")
    print("=" * 60)

    # Run simulation
    results = run_simulation(cfg, save_results=True, verbose=True)

    print("\n" + "=" * 60)
    print("Simulation Complete!")
    print(f"Output: {cfg.output.output_path}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
