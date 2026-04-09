"""Brunel 2000 parameter sweep pipeline (launcher).

Grid search over (g, nu_ext) parameter space to map phase diagram.
Uses ProcessPoolExecutor for parallel execution.

Usage:
    # Model A sweep over g and nu_ext
    python brunel_sweep.py \
        network.model_type=A \
        sweep.sweep_g=[3,4,5,6,7,8] \
        sweep.sweep_nu_ext_ratio=[0.5,1,2,3,4] \
        sweep.max_workers=4 \
        output.output_path=./outputs/sweep_a

    # Model B sweep over J_I
    python brunel_sweep.py \
        network.model_type=B \
        sweep.sweep_j_i=[0.05,0.1,0.15,0.2] \
        sweep.sweep_g_e=[4,5,6,7,8] \
        sweep.max_workers=4 \
        output.output_path=./outputs/sweep_b

Follows the OmegaConf worker/launcher pattern from the skill guide.
"""

import itertools
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Optional, cast

from omegaconf import OmegaConf

from brunel_config import (
    BrunelConfig,
    ModelANeuronConfig,
    ModelASynapseConfig,
    ModelBNeuronConfig,
    ModelBSynapseConfig,
    load_config,
    compute_derived_params,
)
from brunel_simulation import run_simulation


def generate_sweep_configs(cfg: BrunelConfig) -> list:
    """Generate all parameter combinations for sweep.

    Args:
        cfg: Base configuration with sweep candidates

    Returns:
        List of (config, label) tuples for each sweep point
    """
    configs = []

    if isinstance(cfg.neuron, ModelANeuronConfig) and isinstance(
        cfg.synapse, ModelASynapseConfig
    ):
        # Model A: sweep over g and nu_ext_ratio
        sweep_g = cfg.sweep.sweep_g
        sweep_nu_ratios = cfg.sweep.sweep_nu_ext_ratio

        # Compute nu_thr for converting ratios to actual rates
        derived = compute_derived_params(cfg)
        nu_thr = derived["nu_thr"]

        for g in sweep_g:
            for ratio in sweep_nu_ratios:
                # Create trial config
                trial_cfg = deepcopy(cfg)
                trial_cfg.synapse.g = g
                trial_cfg.synapse.nu_ext = ratio * nu_thr

                label = f"g_{g:.1f}_nu_{ratio:.2f}"
                configs.append((trial_cfg, label))

    elif isinstance(cfg.neuron, ModelBNeuronConfig) and isinstance(
        cfg.synapse, ModelBSynapseConfig
    ):
        # Model B: sweep over g_e and J_i (effect on phase diagram)
        sweep_g_e = cfg.sweep.sweep_g_e
        sweep_j_i = cfg.sweep.sweep_j_i

        for g_e in sweep_g_e:
            for j_i in sweep_j_i:
                trial_cfg = deepcopy(cfg)
                trial_cfg.synapse.g_e = g_e
                trial_cfg.synapse.g_i = g_e  # Use same g for I→E and I→I
                trial_cfg.synapse.J_i = j_i

                label = f"ge_{g_e:.1f}_ji_{j_i:.2f}"
                configs.append((trial_cfg, label))
    else:
        raise ValueError(
            "Neuron and synapse config types must match (Model A or Model B)"
        )

    return configs


def run_single_sweep(
    trial_cfg: BrunelConfig,
    label: str,
    output_base: Path,
    verbose: bool = True,
) -> dict:
    """Run a single simulation in the sweep.

    Args:
        trial_cfg: Configuration for this trial
        label: Trial label
        output_base: Base output directory
        verbose: Print progress

    Returns:
        results: Dictionary with results
    """
    # Set output path for this trial
    trial_cfg.output.output_path = output_base / label

    # Check if already completed (resume capability)
    if trial_cfg.sweep.resume:
        metrics_path = trial_cfg.output.output_path / "metrics.json"
        if metrics_path.exists():
            if verbose:
                print(f"  {label}: Already completed, skipping")
            with open(metrics_path) as f:
                return json.load(f)

    if verbose:
        print(f"  {label}: Running...")

    try:
        # Run simulation
        results = run_simulation(trial_cfg, save_results=True, verbose=False)
        results["label"] = label
        results["success"] = True

        if verbose:
            print(
                f"  {label}: Done (rate_e={results['rate_e']:.1f}, state={results['state_classification']})"
            )

        return results

    except Exception as e:
        if verbose:
            print(f"  {label}: Failed - {e}")
        return {
            "label": label,
            "success": False,
            "error": str(e),
        }


def run_sweep_parallel(
    cfg: BrunelConfig,
    max_workers: Optional[int] = None,
    verbose: bool = True,
) -> list:
    """Run parameter sweep in parallel.

    Args:
        cfg: Configuration with sweep candidates
        max_workers: Number of parallel workers (default: cfg.sweep.max_workers)
        verbose: Print progress

    Returns:
        List of results dictionaries
    """
    if max_workers is None:
        max_workers = cfg.sweep.max_workers

    # Generate sweep configs
    sweep_configs = generate_sweep_configs(cfg)

    if verbose:
        model_type = "A" if isinstance(cfg.neuron, ModelANeuronConfig) else "B"
        print(f"\n{'=' * 60}")
        print(f"Parameter Sweep: Model {model_type}")
        print(f"{'=' * 60}")
        print(f"Total trials: {len(sweep_configs)}")
        print(f"Max workers: {max_workers}")
        print(f"Output: {cfg.output.output_path}")
        print()

    output_base = Path(cfg.output.output_path)
    output_base.mkdir(parents=True, exist_ok=True)

    results = []

    # Run in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                run_single_sweep,
                trial_cfg,
                label,
                output_base,
                verbose,
            ): label
            for trial_cfg, label in sweep_configs
        }

        # Collect results as they complete
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    if verbose:
        # Summary
        successful = sum(1 for r in results if r.get("success", False))
        print(f"\n{'=' * 60}")
        print(f"Sweep Complete: {successful}/{len(results)} successful")
        print(f"{'=' * 60}")

    return results


def run_sweep_sequential(
    cfg: BrunelConfig,
    verbose: bool = True,
) -> list:
    """Run parameter sweep sequentially (for debugging).

    Args:
        cfg: Configuration with sweep candidates
        verbose: Print progress

    Returns:
        List of results dictionaries
    """
    # Generate sweep configs
    sweep_configs = generate_sweep_configs(cfg)

    if verbose:
        model_type = "A" if isinstance(cfg.neuron, ModelANeuronConfig) else "B"
        print(f"\n{'=' * 60}")
        print(f"Parameter Sweep (Sequential): Model {model_type}")
        print(f"{'=' * 60}")
        print(f"Total trials: {len(sweep_configs)}")
        print(f"Output: {cfg.output.output_path}")
        print()

    output_base = Path(cfg.output.output_path)
    output_base.mkdir(parents=True, exist_ok=True)

    results = []

    for trial_cfg, label in sweep_configs:
        result = run_single_sweep(trial_cfg, label, output_base, verbose)
        results.append(result)

    if verbose:
        successful = sum(1 for r in results if r.get("success", False))
        print(f"\n{'=' * 60}")
        print(f"Sweep Complete: {successful}/{len(results)} successful")
        print(f"{'=' * 60}")

    return results


def save_sweep_summary(results: list, output_path: Path):
    """Save aggregated sweep results.

    Args:
        results: List of result dictionaries
        output_path: Output directory
    """
    # Save full results
    results_path = output_path / "sweep_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Create summary table
    successful_results = [r for r in results if r.get("success", False)]

    if successful_results:
        import pandas as pd

        df = pd.DataFrame(successful_results)

        # Save CSV
        csv_path = output_path / "sweep_summary.csv"
        df.to_csv(csv_path, index=False)

        # State distribution
        print("\nState Distribution:")
        print(df["state_classification"].value_counts())

    print(f"\nSweep summary saved to: {output_path}")


def main():
    """Main entry point for sweep."""
    # Load config with CLI overrides
    cfg = load_config()

    # Run sweep
    if cfg.sweep.max_workers == 1:
        results = run_sweep_sequential(cfg, verbose=True)
    else:
        results = run_sweep_parallel(
            cfg, max_workers=cfg.sweep.max_workers, verbose=True
        )

    # Save summary
    save_sweep_summary(results, Path(cfg.output.output_path))

    return results


if __name__ == "__main__":
    main()
