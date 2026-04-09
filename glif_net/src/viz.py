"""Visualization utilities for GLIF RSNN."""

import torch
import numpy as np
from pathlib import Path
from typing import Optional


def visualize_spike_raster(
    spikes: torch.Tensor,
    save_path: Optional[Path] = None,
    title: str = "Spike Raster",
    max_neurons: int = 100,
) -> np.ndarray:
    """
    Create a spike raster plot.

    Args:
        spikes: (T, n_neuron) or (T, batch, n_neuron) spike tensor
        save_path: Optional path to save figure
        title: Plot title
        max_neurons: Maximum neurons to display

    Returns:
        raster: Binary array (T, min(n_neuron, max_neurons))
    """
    if spikes.ndim == 3:
        # Average over batch
        spikes = spikes.mean(dim=1)

    spikes = spikes.cpu().numpy()
    T, n_neuron = spikes.shape

    # Limit neurons for display
    n_display = min(n_neuron, max_neurons)
    spikes = spikes[:, :n_display]

    # Convert to binary
    raster = (spikes > 0).astype(np.float32)

    # Save if path provided
    if save_path is not None:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(raster.T, aspect="auto", cmap="binary", interpolation="nearest")
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Neuron")
            ax.set_title(title)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()
        except ImportError:
            pass

    return raster


def visualize_voltage_traces(
    voltage: torch.Tensor,
    v_threshold: float,
    v_reset: float,
    save_path: Optional[Path] = None,
    title: str = "Voltage Traces",
    max_neurons: int = 10,
) -> np.ndarray:
    """
    Plot voltage traces for selected neurons.

    Args:
        voltage: (T, n_neuron) or (T, batch, n_neuron) voltage tensor
        v_threshold: Firing threshold
        v_reset: Reset voltage
        save_path: Optional path to save figure
        title: Plot title
        max_neurons: Maximum neurons to plot

    Returns:
        traces: Voltage array (T, min(n_neuron, max_neurons))
    """
    if voltage.ndim == 3:
        # Take first batch
        voltage = voltage[:, 0, :]

    voltage = voltage.cpu().numpy()
    T, n_neuron = voltage.shape

    # Select neurons
    n_display = min(n_neuron, max_neurons)
    traces = voltage[:, :n_display]

    if save_path is not None:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 6))
            time = np.arange(T)

            for i in range(n_display):
                ax.plot(time, traces[:, i], alpha=0.7, label=f"Neuron {i}")

            ax.axhline(y=v_threshold, color="r", linestyle="--", label="Threshold")
            ax.axhline(y=v_reset, color="g", linestyle="--", label="Reset")

            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Voltage (mV)")
            ax.set_title(title)
            ax.legend()
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()
        except ImportError:
            pass

    return traces


def compute_firing_rate_stats(spikes: torch.Tensor, dt: float = 1.0) -> dict:
    """
    Compute firing rate statistics.

    Args:
        spikes: (T, batch, n_neuron) spike tensor
        dt: Timestep in ms

    Returns:
        stats: Dictionary with firing rate statistics
    """
    T, batch, n_neuron = spikes.shape
    duration_sec = T * dt / 1000.0

    # Firing rates per neuron (Hz)
    spike_counts = spikes.sum(dim=(0, 1))  # (n_neuron,)
    firing_rates = spike_counts / (batch * duration_sec)

    stats = {
        "mean_rate_hz": firing_rates.mean().item(),
        "std_rate_hz": firing_rates.std().item(),
        "min_rate_hz": firing_rates.min().item(),
        "max_rate_hz": firing_rates.max().item(),
        "total_spikes": spike_counts.sum().item(),
        "spikes_per_neuron": spike_counts.mean().item(),
    }

    return stats


def log_tensorboard_viz(
    writer,
    spikes: torch.Tensor,
    voltage: torch.Tensor,
    epoch: int,
    tag_prefix: str = "viz",
):
    """Log visualizations to tensorboard."""
    try:
        from torch.utils.tensorboard import SummaryWriter

        if not isinstance(writer, SummaryWriter):
            return

        # Log spike raster as image
        if spikes.ndim == 3:
            raster = visualize_spike_raster(spikes[:, 0, :])  # First sample
            writer.add_image(
                f"{tag_prefix}/spike_raster",
                raster[np.newaxis, :, :],  # Add channel dim
                epoch,
                dataformats="CHW",
            )

        # Log firing rate histogram
        T, batch, n_neuron = spikes.shape
        dt = 1.0
        duration_sec = T * dt / 1000.0
        spike_counts = spikes.sum(dim=(0, 1))
        firing_rates = spike_counts / duration_sec

        writer.add_histogram(
            f"{tag_prefix}/firing_rates",
            firing_rates,
            epoch,
        )
    except Exception:
        pass
