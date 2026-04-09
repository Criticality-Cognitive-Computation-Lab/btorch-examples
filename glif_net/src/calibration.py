"""Input/output calibration utilities."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def calibrate_io_scales(
    model: torch.nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    num_batches: int = 10,
    target_input_std: float = 1.0,
    target_output_std: float = 1.0,
) -> None:
    """
    Calibrate input and output scales based on data statistics.

    This helps normalize the input current and output activations to
    reasonable ranges for stable training.
    """
    model.eval()

    input_activations = []
    output_activations = []

    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break

        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)

        # Get input projection (before neuron)
        if hasattr(model, "input_linear"):
            x_flat = x.reshape(-1, x.shape[-1])
            input_proj = model.input_linear(x_flat)
            input_activations.append(input_proj.std().item())

        # Get output activations
        output, _ = model(x)
        output_activations.append(output.std().item())

    # Adjust scales to target std (LearnableScale has `scale` parameter)
    if input_activations and hasattr(model, "input_scale"):
        current_input_std = sum(input_activations) / len(input_activations)
        if current_input_std > 0:
            new_scale = target_input_std / current_input_std
            model.input_scale.scale.data = torch.tensor(new_scale, device=device)

    if output_activations and hasattr(model, "output_scale"):
        current_output_std = sum(output_activations) / len(output_activations)
        if current_output_std > 0:
            new_scale = target_output_std / current_output_std
            model.output_scale.scale.data = torch.tensor(new_scale, device=device)

    model.train()


@torch.no_grad()
def calibrate_thresholds(
    model: torch.nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    num_batches: int = 10,
    target_firing_rate: float = 10.0,  # Hz
) -> None:
    """
    Calibrate neuron thresholds to achieve target firing rate.

    This adjusts v_threshold based on observed activity.
    """
    model.eval()

    all_spike_counts = []

    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break

        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)

        output, states = model(x)
        spikes = states["spikes"]  # (T, batch, n_neuron)

        # Count spikes per neuron
        spike_count = spikes.sum(dim=(0, 1))  # (n_neuron,)
        all_spike_counts.append(spike_count)

    if not all_spike_counts:
        return

    # Average firing rates
    avg_spike_count = torch.stack(all_spike_counts).mean(dim=0)
    T = spikes.shape[0]
    dt = 1.0  # ms
    duration_sec = T * dt / 1000.0
    firing_rates = avg_spike_count / duration_sec  # Hz

    # Adjust thresholds (increase threshold if firing too much, decrease if too little)
    if hasattr(model, "neuron"):
        current_threshold = model.neuron.v_threshold
        avg_rate = firing_rates.mean().item()

        if avg_rate > 0:
            ratio = target_firing_rate / avg_rate
            # Adjust threshold based on ratio (simple heuristic)
            if ratio < 1.0:  # Firing too much
                model.neuron.v_threshold = current_threshold + 2.0
            elif ratio > 2.0:  # Firing too little
                model.neuron.v_threshold = current_threshold - 2.0

    model.train()
