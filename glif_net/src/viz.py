"""Visualization wrappers built on top of btorch visualisation APIs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from btorch.visualisation.timeseries import (
    SimulationStates,
    plot_neuron_traces,
    plot_raster,
)

from .pop_connectivity import plot_population_graph


def visualize_spike_raster(
    spikes: torch.Tensor,
    dt: float = 1.0,
    save_path: Optional[Path] = None,
    title: str = "Spike Raster",
    max_neurons: int = 100,
) -> np.ndarray:
    """Render raster via btorch and return plotted spike array."""
    if spikes.ndim == 3:
        spikes = spikes[:, 0, :]
    spikes_np = spikes.detach().cpu().numpy()[:, :max_neurons]

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_raster(spikes_np, dt=dt, ax=ax, title=title)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180)
    plt.close(fig)
    return spikes_np


def visualize_voltage_traces(
    voltage: torch.Tensor,
    spikes: torch.Tensor | None,
    v_threshold: float,
    v_reset: float,
    dt: float = 1.0,
    save_path: Optional[Path] = None,
    title: str = "Voltage Traces",
    max_neurons: int = 10,
) -> np.ndarray:
    """Render traces via SimulationStates + plot_neuron_traces."""
    states = SimulationStates(
        voltage=voltage,
        spikes=spikes,
        dt=dt,
        v_threshold=v_threshold,
        v_reset=v_reset,
    )
    fig = plot_neuron_traces(
        states=states, sample_size=max_neurons, show_asc=False, show_psc=False
    )
    if isinstance(fig, dict):
        first = next(iter(fig.values()))
        fig_obj = first
    else:
        fig_obj = fig
    fig_obj.suptitle(title)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig_obj.savefig(save_path, dpi=180)
    plt.close(fig_obj)

    if voltage.ndim == 3:
        return voltage[:, 0, :max_neurons].detach().cpu().numpy()
    return voltage[:, :max_neurons].detach().cpu().numpy()


def visualize_population_connectivity(model: torch.nn.Module, save_path: Path) -> Path:
    """Plot population-level graph (Fig.1C-style)."""
    return plot_population_graph(model, save_path=save_path)


def compute_firing_rate_stats(spikes: torch.Tensor, dt: float = 1.0) -> dict:
    """Compute firing-rate summary stats from (T, B, N) spikes."""
    T, batch, _ = spikes.shape
    duration_sec = T * dt / 1000.0
    spike_counts = spikes.sum(dim=(0, 1))
    firing_rates = spike_counts / max(batch * duration_sec, 1e-12)
    return {
        "mean_rate_hz": firing_rates.mean().item(),
        "std_rate_hz": firing_rates.std().item(),
        "min_rate_hz": firing_rates.min().item(),
        "max_rate_hz": firing_rates.max().item(),
        "total_spikes": spike_counts.sum().item(),
        "spikes_per_neuron": spike_counts.mean().item(),
    }


def log_tensorboard_viz(
    writer,
    spikes: torch.Tensor,
    voltage: torch.Tensor,
    epoch: int,
    dt: float = 1.0,
    tag_prefix: str = "viz",
):
    """Log raster image and firing-rate histogram to tensorboard."""
    try:
        raster = visualize_spike_raster(spikes[:, 0, :], dt=dt)
        writer.add_image(
            f"{tag_prefix}/spike_raster",
            raster[np.newaxis, :, :],
            epoch,
            dataformats="CHW",
        )
        T, _, _ = spikes.shape
        duration_sec = T * dt / 1000.0
        rates = spikes.sum(dim=(0, 1)) / max(duration_sec, 1e-12)
        writer.add_histogram(f"{tag_prefix}/firing_rates", rates, epoch)
    except Exception:
        pass
