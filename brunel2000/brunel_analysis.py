"""Brunel 2000 analysis and visualization utilities.

Functions for analyzing simulation results and creating plots
matching the figures in Brunel 2000 paper.

Usage:
    from brunel_analysis import plot_phase_diagram, plot_raster

    results = load_results("./outputs/sweep_a")
    plot_phase_diagram(results, save_path="phase_diagram.png")
"""

import json
import pickle
from pathlib import Path
from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt


def load_results(result_path: Union[str, Path]) -> dict:
    """Load simulation results from directory.

    Args:
        result_path: Path to results directory

    Returns:
        Dictionary with metrics
    """
    result_path = Path(result_path)

    # Try metrics.json first
    metrics_path = result_path / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)

    # Try results.pkl
    pkl_path = result_path / "results.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    raise FileNotFoundError(f"No results found in {result_path}")


def load_sweep_results(sweep_path: Union[str, Path]) -> list:
    """Load all results from a sweep directory.

    Args:
        sweep_path: Path to sweep directory

    Returns:
        List of result dictionaries
    """
    sweep_path = Path(sweep_path)

    # Try sweep_results.json
    summary_path = sweep_path / "sweep_results.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)

    # Load individual results
    results = []
    for subdir in sweep_path.iterdir():
        if subdir.is_dir():
            try:
                result = load_results(subdir)
                results.append(result)
            except FileNotFoundError:
                pass

    return results


def plot_raster(
    spikes_e: np.ndarray,
    spikes_i: np.ndarray,
    dt: float,
    n_show: int = 100,
    save_path: Optional[Path] = None,
):
    """Plot spike raster (similar to Fig 8 in Brunel 2000).

    Args:
        spikes_e: E spikes [T, n_exc]
        spikes_i: I spikes [T, n_inh]
        dt: Time step (ms)
        n_show: Number of neurons to show per population
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    time = np.arange(spikes_e.shape[0]) * dt

    # E neurons (top)
    n_show_e = min(n_show, spikes_e.shape[1])
    for i in range(n_show_e):
        spike_times = time[spikes_e[:, i] > 0]
        ax1.scatter(spike_times, np.full_like(spike_times, i), c="red", s=1, marker="|")

    ax1.set_ylabel("E neuron index")
    ax1.set_title(f"Excitatory neurons (showing {n_show_e}/{spikes_e.shape[1]})")
    ax1.set_ylim(-1, n_show_e + 1)

    # I neurons (bottom)
    n_show_i = min(n_show, spikes_i.shape[1])
    for i in range(n_show_i):
        spike_times = time[spikes_i[:, i] > 0]
        ax2.scatter(
            spike_times, np.full_like(spike_times, i), c="blue", s=1, marker="|"
        )

    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("I neuron index")
    ax2.set_title(f"Inhibitory neurons (showing {n_show_i}/{spikes_i.shape[1]})")
    ax2.set_ylim(-1, n_show_i + 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved raster plot to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_population_activity(
    pop_e: np.ndarray,
    pop_i: np.ndarray,
    dt: float,
    save_path: Optional[Path] = None,
):
    """Plot population activity over time.

    Args:
        pop_e: E population activity [T]
        pop_i: I population activity [T]
        dt: Time step (ms)
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    time = np.arange(len(pop_e)) * dt

    # E population
    ax1.plot(time, pop_e, "r-", linewidth=0.5)
    ax1.axhline(y=pop_e.mean(), color="r", linestyle="--", label="Mean")
    ax1.set_ylabel("E rate (Hz)")
    ax1.set_title("Excitatory population activity")
    ax1.legend()

    # I population
    ax2.plot(time, pop_i, "b-", linewidth=0.5)
    ax2.axhline(y=pop_i.mean(), color="b", linestyle="--", label="Mean")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("I rate (Hz)")
    ax2.set_title("Inhibitory population activity")
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved population activity to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_phase_diagram(
    results: list,
    save_path: Optional[Path] = None,
):
    """Plot phase diagram (similar to Fig 2 in Brunel 2000).

    Shows state classification in (g, nu_ext/nu_thr) parameter space.

    Args:
        results: List of result dictionaries from sweep
        save_path: Optional path to save figure
    """
    # Filter successful results
    results = [r for r in results if r.get("success", False)]

    if not results:
        print("No successful results to plot")
        return

    # Extract data
    g_vals = [r["g"] for r in results]
    nu_ratios = [r["nu_ext_ratio"] for r in results]
    states = [r["state_classification"] for r in results]
    rates = [r["rate_e"] for r in results]

    # State to color mapping
    state_colors = {
        "SR": "red",
        "AR": "orange",
        "AI": "green",
        "SI": "blue",
        "UNKNOWN": "gray",
    }
    colors = [state_colors.get(s, "gray") for s in states]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: State classification
    scatter1 = ax1.scatter(g_vals, nu_ratios, c=colors, s=50, alpha=0.7)
    ax1.axvline(x=4, color="k", linestyle="--", alpha=0.5, label="Balanced (g=4)")
    ax1.set_xlabel("Inhibitory strength g")
    ax1.set_ylabel("External input ν_ext/ν_thr")
    ax1.set_title("Phase Diagram: State Classification")
    ax1.set_xlim(min(g_vals) - 0.5, max(g_vals) + 0.5)
    ax1.set_ylim(min(nu_ratios) - 0.2, max(nu_ratios) + 0.2)
    ax1.legend()

    # Create legend for states
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="red", label="SR (Synchronous Regular)"),
        Patch(facecolor="orange", label="AR (Asynchronous Regular)"),
        Patch(facecolor="green", label="AI (Asynchronous Irregular)"),
        Patch(facecolor="blue", label="SI (Synchronous Irregular)"),
    ]
    ax1.legend(handles=legend_elements, loc="upper left")

    # Right: Firing rate heatmap
    # Create grid for contour plot
    g_unique = sorted(set(g_vals))
    nu_unique = sorted(set(nu_ratios))

    if len(g_unique) > 1 and len(nu_unique) > 1:
        # Grid data
        rate_grid = np.zeros((len(nu_unique), len(g_unique)))
        for i, nu in enumerate(nu_unique):
            for j, g in enumerate(g_unique):
                # Find matching result
                for r in results:
                    if abs(r["g"] - g) < 0.01 and abs(r["nu_ext_ratio"] - nu) < 0.01:
                        rate_grid[i, j] = r["rate_e"]
                        break

        im = ax2.imshow(
            rate_grid,
            aspect="auto",
            origin="lower",
            extent=[min(g_unique), max(g_unique), min(nu_unique), max(nu_unique)],
            cmap="viridis",
        )
        ax2.set_xlabel("Inhibitory strength g")
        ax2.set_ylabel("External input ν_ext/ν_thr")
        ax2.set_title("E Population Firing Rate (Hz)")
        plt.colorbar(im, ax=ax2)
    else:
        ax2.text(
            0.5,
            0.5,
            "Need grid data for heatmap",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved phase diagram to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_firing_rate_comparison(
    results: list,
    save_path: Optional[Path] = None,
):
    """Compare simulated firing rates with theoretical predictions.

    Args:
        results: List of result dictionaries
        save_path: Optional path to save figure
    """
    # Filter successful AI state results
    ai_results = [
        r
        for r in results
        if r.get("success", False) and r.get("state_classification") == "AI"
    ]

    if not ai_results:
        print("No AI state results for comparison")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by g value
    g_groups = {}
    for r in ai_results:
        g = r["g"]
        if g not in g_groups:
            g_groups[g] = {"nu_ratio": [], "rate": []}
        g_groups[g]["nu_ratio"].append(r["nu_ext_ratio"])
        g_groups[g]["rate"].append(r["rate_e"])

    # Plot for each g value
    colors = plt.cm.viridis(np.linspace(0, 1, len(g_groups)))

    for (g, data), color in zip(sorted(g_groups.items()), colors):
        nu_ratios = np.array(data["nu_ratio"])
        rates = np.array(data["rate"])

        # Sort by nu_ratio
        idx = np.argsort(nu_ratios)

        ax.plot(
            nu_ratios[idx],
            rates[idx],
            "o-",
            color=color,
            label=f"g={g:.1f} (sim)",
            markersize=4,
        )

        # Theoretical prediction: ν_0 = (ν_ext - ν_thr) / (g*γ - 1)
        # For large g, this gives linear relationship
        gamma = 0.25
        theory_rate = (nu_ratios[idx] - 1) / (g * gamma - 1)
        theory_rate = np.maximum(theory_rate, 0)  # Can't be negative

        ax.plot(
            nu_ratios[idx],
            theory_rate,
            "--",
            color=color,
            alpha=0.5,
            label=f"g={g:.1f} (theory)",
        )

    ax.set_xlabel("External input ν_ext/ν_thr")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title("Firing Rate: Simulation vs Theory (AI state)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved firing rate comparison to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_cv_analysis(
    results: list,
    save_path: Optional[Path] = None,
):
    """Plot CV of ISI vs g (similar to Fig 1C).

    Args:
        results: List of result dictionaries
        save_path: Optional path to save figure
    """
    results = [r for r in results if r.get("success", False)]

    if not results:
        print("No results to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by nu_ext ratio
    nu_groups = {}
    for r in results:
        nu_ratio = round(r["nu_ext_ratio"], 1)
        if nu_ratio not in nu_groups:
            nu_groups[nu_ratio] = {"g": [], "cv": []}
        nu_groups[nu_ratio]["g"].append(r["g"])
        nu_groups[nu_ratio]["cv"].append(r["cv_e"])

    # Plot for each nu_ext ratio
    for nu_ratio in sorted(nu_groups.keys()):
        data = nu_groups[nu_ratio]
        g_vals = np.array(data["g"])
        cvs = np.array(data["cv"])

        # Sort by g
        idx = np.argsort(g_vals)

        ax.plot(
            g_vals[idx],
            cvs[idx],
            "o-",
            label=f"ν_ext/ν_thr={nu_ratio:.1f}",
            markersize=4,
        )

    ax.axvline(x=4, color="k", linestyle="--", alpha=0.5, label="Balanced (g=4)")
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5, label="Poisson (CV=1)")
    ax.set_xlabel("Inhibitory strength g")
    ax.set_ylabel("CV of ISI")
    ax.set_title("Coefficient of Variation vs g")
    ax.legend()
    ax.set_xlim(left=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved CV analysis to: {save_path}")
    else:
        plt.show()

    plt.close()


def create_summary_plots(sweep_path: Union[str, Path]):
    """Create all summary plots for a sweep.

    Args:
        sweep_path: Path to sweep directory
    """
    sweep_path = Path(sweep_path)

    print(f"Creating summary plots for: {sweep_path}")

    # Load results
    results = load_sweep_results(sweep_path)

    if not results:
        print("No results found")
        return

    print(f"Loaded {len(results)} results")

    # Create plots
    plot_phase_diagram(results, save_path=sweep_path / "phase_diagram.png")
    plot_firing_rate_comparison(results, save_path=sweep_path / "firing_rates.png")
    plot_cv_analysis(results, save_path=sweep_path / "cv_analysis.png")

    print(f"Plots saved to: {sweep_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python brunel_analysis.py <sweep_path>")
        sys.exit(1)

    sweep_path = sys.argv[1]
    create_summary_plots(sweep_path)
