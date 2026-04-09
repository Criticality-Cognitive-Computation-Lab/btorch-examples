"""Gradient-based calibration utilities for regularization weights."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def calibrate_regularization_weights(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn,
    device: torch.device,
    num_batches: int = 5,
    target_ratio: float = 0.1,
    loss_components: list[str] | None = None,
) -> dict[str, float]:
    """
    Calibrate regularization weights by matching gradient magnitudes.

    This implements the "Precise Calibration" method from btorch regularization
    patterns: backward each loss separately and compare param grad magnitudes.

    Args:
        model: The model to calibrate
        train_loader: Training data loader
        loss_fn: Loss function that returns (total_loss, loss_dict)
        device: torch device
        num_batches: Number of batches to average over
        target_ratio: Target ratio of reg_grad_norm / task_grad_norm (default 0.1)
        loss_components: List of loss component names to calibrate (e.g., ['voltage', 'rate'])

    Returns:
        Dictionary of calibrated weights for each component
    """
    from btorch.models import environ, functional
    from btorch.models.init import uniform_v_

    if loss_components is None:
        loss_components = ["voltage", "rate"]

    model.eval()

    # Collect gradient norms for each component
    task_grad_norms = []
    component_grad_norms = {comp: [] for comp in loss_components}

    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break

        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)

        # Handle input shape
        if (
            x.ndim == 3 and x.shape[0] != loss_fn.config.T
            if hasattr(loss_fn, "config")
            else True
        ):
            x = x.transpose(0, 1)

        T = x.shape[0]
        batch_size = x.shape[1]

        # Dummy target (we only care about gradient magnitudes)
        target = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Reset state
        functional.reset_net(model.rnn, batch_size=batch_size)
        uniform_v_(model.neuron, set_reset_value=True)

        # Forward pass
        with environ.context(dt=model.dt):
            output, states = model(x)

        # Compute all losses
        total_loss, loss_dict = loss_fn(output, target, states, T)

        # Measure task loss (CE) gradient norm
        model.zero_grad()
        ce_loss = torch.tensor(loss_dict["ce"], device=device, requires_grad=True)
        # Need to recompute with actual tensor
        ce_loss_actual = nn.functional.cross_entropy(output, target)
        ce_loss_actual.backward(retain_graph=True)

        task_grad_norm = sum(
            p.grad.norm().item() for p in model.parameters() if p.grad is not None
        )
        task_grad_norms.append(task_grad_norm)

        # Measure each component's gradient norm
        for comp in loss_components:
            if comp not in loss_dict:
                continue

            model.zero_grad()

            # Recompute component loss
            if comp == "voltage":
                voltage = states["voltage"]
                v_reset = (
                    loss_fn.config.v_reset if hasattr(loss_fn, "config") else -60.0
                )
                comp_loss = torch.mean((voltage - v_reset) ** 2)
            elif comp == "rate":
                spikes = states["spikes"]
                spike_count = spikes.sum(dim=0)
                dt = loss_fn.dt if hasattr(loss_fn, "dt") else 1.0
                duration_sec = T * dt / 1000.0
                firing_rate = spike_count / duration_sec

                rate_min = (
                    loss_fn.config.rate_target_min
                    if hasattr(loss_fn, "config")
                    else 2.0
                )
                rate_max = (
                    loss_fn.config.rate_target_max
                    if hasattr(loss_fn, "config")
                    else 30.0
                )

                rate_penalty = torch.where(
                    firing_rate < rate_min,
                    (rate_min - firing_rate) ** 2,
                    torch.where(
                        firing_rate > rate_max,
                        (firing_rate - rate_max) ** 2,
                        torch.zeros_like(firing_rate),
                    ),
                ).mean()
                comp_loss = rate_penalty
            else:
                continue

            if comp_loss.requires_grad:
                comp_loss.backward()
                comp_grad_norm = sum(
                    p.grad.norm().item()
                    for p in model.parameters()
                    if p.grad is not None
                )
                component_grad_norms[comp].append(comp_grad_norm)

    # Calculate calibrated weights
    calibrated_weights = {}
    avg_task_norm = (
        sum(task_grad_norms) / len(task_grad_norms) if task_grad_norms else 1.0
    )

    for comp, norms in component_grad_norms.items():
        if norms and avg_task_norm > 0:
            avg_comp_norm = sum(norms) / len(norms)
            # weight = target_ratio * task_grad / comp_grad
            calibrated_weight = target_ratio * avg_task_norm / (avg_comp_norm + 1e-8)
            calibrated_weights[comp] = calibrated_weight
        else:
            calibrated_weights[comp] = 0.01  # Default fallback

    model.train()

    return calibrated_weights


@torch.no_grad()
def quick_calibration_check(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn,
    device: torch.device,
    num_batches: int = 3,
) -> dict:
    """
    Quick calibration check: run samples and check loss component ratios.

    Target: reg losses should be ~10-20% of task loss.

    Returns:
        Dictionary with loss statistics and recommended weight adjustments
    """
    from btorch.models import environ, functional
    from btorch.models.init import uniform_v_

    model.eval()

    ce_losses = []
    voltage_losses = []
    rate_losses = []

    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break

        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)

        if x.ndim == 3 and x.shape[0] != getattr(loss_fn, "T", 100):
            x = x.transpose(0, 1)

        T = x.shape[0]
        batch_size = x.shape[1]
        target = torch.zeros(batch_size, dtype=torch.long, device=device)

        functional.reset_net(model.rnn, batch_size=batch_size)
        uniform_v_(model.neuron, set_reset_value=True)

        with environ.context(dt=getattr(model, "dt", 1.0)):
            output, states = model(x)

        _, loss_dict = loss_fn(output, target, states, T)

        ce_losses.append(loss_dict.get("ce", 0))
        voltage_losses.append(loss_dict.get("voltage", 0))
        rate_losses.append(loss_dict.get("rate", 0))

    avg_ce = sum(ce_losses) / len(ce_losses) if ce_losses else 1.0
    avg_voltage = sum(voltage_losses) / len(voltage_losses) if voltage_losses else 0
    avg_rate = sum(rate_losses) / len(rate_losses) if rate_losses else 0

    # Calculate current ratios
    voltage_ratio = avg_voltage / avg_ce if avg_ce > 0 else 0
    rate_ratio = avg_rate / avg_ce if avg_ce > 0 else 0

    # Recommendations (target ~10-20%)
    recommendations = {
        "ce_loss": avg_ce,
        "voltage_loss": avg_voltage,
        "rate_loss": avg_rate,
        "voltage_ratio": voltage_ratio,
        "rate_ratio": rate_ratio,
        "voltage_weight_recommendation": 0.15 / (voltage_ratio + 1e-8),
        "rate_weight_recommendation": 0.15 / (rate_ratio + 1e-8),
    }

    model.train()

    return recommendations
