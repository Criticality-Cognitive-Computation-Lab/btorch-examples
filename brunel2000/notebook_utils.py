"""Pure plotting / formatting helpers for the Brunel 2000 tutorial notebook.

These functions contain NO btorch simulation logic and NO wrapper indirection.
They only operate on arrays/DataFrames and return matplotlib objects.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _spike_raster(
    spikes: np.ndarray,
    dt_ms: float,
    n_e: int,
    ax: plt.Axes,
    max_neurons: int = 400,
    title: str = "",
):
    """Plot a simple raster showing E and I populations."""
    T, n_tot = spikes.shape
    if n_tot > max_neurons:
        n_show_e = max(1, int(max_neurons * (n_e / n_tot)))
        n_show_i = max(1, max_neurons - n_show_e)
        e_idx = np.linspace(0, n_e - 1, num=n_show_e, dtype=int)
        i_idx = np.linspace(n_e, n_tot - 1, num=n_show_i, dtype=int)
        keep = np.unique(np.concatenate([e_idx, i_idx]))
        spikes = spikes[:, keep]
        n_e = len(e_idx)
    else:
        keep = np.arange(n_tot)

    t_ms = np.arange(spikes.shape[0]) * dt_ms
    e_mask = keep < n_e

    for i, col in enumerate(keep):
        times = t_ms[spikes[:, i] > 0]
        color = "#54a24b" if e_mask[i] else "#e45756"
        ax.scatter(times, np.full_like(times, col), marker="|", s=8, c=color, lw=0.8)

    ax.axhline(n_e - 0.5, color="gray", ls="--", lw=0.8)
    ax.text(
        0.02, 0.95, "E", transform=ax.transAxes, color="#54a24b", va="top", fontsize=10
    )
    ax.text(
        0.02,
        0.05,
        "I",
        transform=ax.transAxes,
        color="#e45756",
        va="bottom",
        fontsize=10,
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron index")
    ax.set_title(title)
    ax.set_xlim(t_ms.min(), t_ms.max())
    ax.set_ylim(-1, spikes.shape[1])


def _pop_rate(
    spikes: np.ndarray,
    dt_ms: float,
    n_e: int,
    bin_ms: float = 1.0,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot population firing rate (overall, E, I)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 2.2))
    bin_steps = max(int(round(bin_ms / dt_ms)), 1)
    T = spikes.shape[0]
    bins = np.arange(0, T + bin_steps, bin_steps)
    rate_overall = np.empty(len(bins) - 1)
    rate_e = np.empty(len(bins) - 1)
    rate_i = np.empty(len(bins) - 1)
    t_centers = np.empty(len(bins) - 1)
    for k in range(len(bins) - 1):
        lo, hi = bins[k], bins[k + 1]
        rate_overall[k] = spikes[lo:hi].sum() / (spikes.shape[1] * bin_ms * 1e-3)
        rate_e[k] = spikes[lo:hi, :n_e].sum() / (max(n_e, 1) * bin_ms * 1e-3)
        rate_i[k] = spikes[lo:hi, n_e:].sum() / (
            max(spikes.shape[1] - n_e, 1) * bin_ms * 1e-3
        )
        t_centers[k] = (lo + hi) / 2 * dt_ms

    ax.plot(t_centers, rate_overall, color="tab:blue", lw=1.2, label="Overall")
    ax.plot(t_centers, rate_e, color="#54a24b", lw=1.0, label="E", alpha=0.7)
    ax.plot(t_centers, rate_i, color="#e45756", lw=1.0, label="I", alpha=0.7)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Rate (Hz)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Population firing rate")
    return ax


# ---------------------------------------------------------------------------
# Helpers used by the notebook directly
# ---------------------------------------------------------------------------


def plot_raster_and_rate(
    spikes: np.ndarray, dt_ms: float, n_e: int, duration_ms: float | None = None
):
    """Return a figure with raster (top) and rate (bottom)."""
    fig, axes = plt.subplots(
        2, 1, figsize=(8, 5), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]}
    )
    _spike_raster(spikes, dt_ms, n_e, ax=axes[0])
    _pop_rate(spikes, dt_ms, n_e, ax=axes[1])
    fig.tight_layout()
    return fig, axes


def plot_raster_grid(
    results: list[tuple[np.ndarray, float, int, str]],
    dt_ms: float,
    figsize=(10, 8),
):
    """Plot a grid of rasters for multiple regimes.

    Args
    ----
    results : list of (spikes, dt_ms, n_e, title)
    """
    n = len(results)
    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n == 1:
        axes = np.array([[axes]])
    axes = axes.flatten()
    for ax, (spikes, _, n_e, title) in zip(axes, results):
        _spike_raster(spikes, dt_ms, n_e, ax=ax, title=title)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig, axes


def plot_phase_map_overlay(
    df: pd.DataFrame,
    model: str = "a",
    j_i: float | None = None,
    overlay_df: pd.DataFrame | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot a single phase map (regime background + optional overlay points).

    Parameters
    ----------
    df : DataFrame with columns g, eta, regime (or other metrics)
    model : str label for title
    j_i : optional float to filter df
    overlay_df : DataFrame with columns g, eta to overlay as scatter
    ax : optional Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    df = df.copy()
    if "regime" not in df.columns:
        # we expect classify_regime to be applied before calling this
        raise ValueError("DataFrame needs a 'regime' column")

    if j_i is not None and "j_i" in df.columns:
        df = df[df["j_i"] == j_i]

    regime_order = ["silent", "SR", "AI", "SI-slow", "SI-fast"]
    regime_to_code = {name: i for i, name in enumerate(regime_order)}
    cmap = ListedColormap(["#f0f0f0", "#4c78a8", "#54a24b", "#f58518", "#b279a2"])

    piv = (
        df.assign(regime_code=df["regime"].map(regime_to_code))
        .pivot_table(index="eta", columns="g", values="regime_code", aggfunc="first")
        .sort_index()
    )

    x_vals = np.asarray(piv.columns, dtype=float)
    y_vals = np.asarray(piv.index, dtype=float)
    extent = [
        float(np.nanmin(x_vals)),
        float(np.nanmax(x_vals)),
        float(np.nanmin(y_vals)),
        float(np.nanmax(y_vals)),
    ]
    if extent[0] == extent[1]:
        extent[0] -= 0.5
        extent[1] += 0.5
    if extent[2] == extent[3]:
        extent[2] -= 0.5
        extent[3] += 0.5

    im = ax.imshow(
        piv.values,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        extent=extent,
        vmin=0,
        vmax=len(regime_order) - 1,
    )
    ax.set_xlabel("g")
    ax.set_ylabel(r"$\\eta = \\nu_{ext} / \\nu_{thr}$")
    title = f"Model {model.upper()} — Regime Map"
    if j_i is not None:
        title += f" ($J_I$={j_i:.2f})"
    ax.set_title(title)

    legend_handles = [
        plt.Line2D(
            [0], [0], marker="s", color="w", markerfacecolor=cmap(i), markersize=10
        )
        for i in range(len(regime_order))
    ]
    ax.legend(
        legend_handles, regime_order, loc="upper right", framealpha=0.95, fontsize=8
    )

    if overlay_df is not None and not overlay_df.empty:
        ax.scatter(
            overlay_df["g"],
            overlay_df["eta"],
            c="k",
            marker="x",
            s=40,
            lw=1.2,
            label="sampled",
            zorder=10,
        )
        ax.legend(loc="upper left", fontsize=8)

    return ax


def metrics_table(regime_rows: list[dict]) -> pd.DataFrame:
    """Format a small metrics DataFrame for display."""
    df = pd.DataFrame(regime_rows)
    return df


# ---------------------------------------------------------------------------
# cleanup helper
# ---------------------------------------------------------------------------


def close_all():
    plt.close("all")
