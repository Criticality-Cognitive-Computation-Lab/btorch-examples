#!/usr/bin/env python
"""Quick runner for Model B (heterogeneous E/I neurons).

Extension of Model A where E and I populations have different parameters:
- Different membrane time constants (tau_e vs tau_i)
- Different synaptic efficacies (J_e vs J_i)
- Different inhibitory strengths (g_e for I→E, g_i for I→I)
- Different external input rates (nu_ext_e, nu_ext_i)

Note: Heterogeneous delays (D_EE, D_EI, D_IE, D_II) are not yet supported
and will be added when btorch has native delay support.

Usage:
    # Basic run with defaults (Model B)
    python run_model_b.py

    # Faster inhibitory neurons
    python run_model_b.py \
        neuron.tau_i=10.0 \
        synapse.g_e=6.0

    # Stronger inhibition on I population
    python run_model_b.py \
        synapse.g_e=5.0 \
        synapse.g_i=8.0 \
        synapse.J_i=0.15

Model B Specific Parameters:
    - neuron.tau_e: E neuron time constant (default: 20.0 ms)
    - neuron.tau_i: I neuron time constant (default: 20.0 ms)
    - synapse.J_e: EPSP for E neurons (default: 0.1 mV)
    - synapse.J_i: EPSP for I neurons (default: 0.1 mV)
    - synapse.g_e: I→E inhibitory strength (default: 5.0)
    - synapse.g_i: I→I inhibitory strength (default: 5.0)
    - synapse.nu_ext_e: External rate to E (default: 10.0 Hz)
    - synapse.nu_ext_i: External rate to I (default: 10.0 Hz)
"""

from brunel_config import (
    load_config,
    ModelBNeuronConfig,
    ModelBSynapseConfig,
)
from brunel_simulation import run_simulation


def main():
    """Run Model B simulation."""
    # Load base config
    cfg = load_config()

    # Switch to Model B configs
    cfg.neuron = ModelBNeuronConfig()
    cfg.synapse = ModelBSynapseConfig()

    print("=" * 60)
    print("Brunel 2000 - Model B (Heterogeneous E/I Neurons)")
    print("=" * 60)
    print("Note: Single constant delay for all connections.")
    print("      Heterogeneous delays (D_EE, D_EI, etc.) will be")
    print("      supported when btorch adds native delay support.")
    print()

    # Run simulation
    results = run_simulation(cfg, save_results=True, verbose=True)

    print("\n" + "=" * 60)
    print("Simulation Complete!")
    print(f"Output: {cfg.output.output_path}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
