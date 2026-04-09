"""Loss functions for GLIF RSNN training."""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossConfig:
    """Loss component weights."""

    ce_weight: float = 1.0
    voltage_weight: float = 0.01
    rate_weight: float = 0.1
    rate_target_min: float = 2.0  # Hz
    rate_target_max: float = 30.0  # Hz
    v_reset: float = -60.0  # mV


class CombinedLoss(nn.Module):
    """
    Combined loss: CE + voltage regularization + firing rate regularization.

    NO ASC current regularization (removed per plan).
    """

    def __init__(self, config: LossConfig, dt: float = 1.0):
        super().__init__()
        self.config = config
        self.dt = dt

    def forward(
        self,
        output: torch.Tensor,  # (batch, num_classes) logits
        target: torch.Tensor,  # (batch,) class labels
        states: dict,  # Contains voltage, spikes
        T: int,  # Total timesteps
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute combined loss.

        Returns:
            total_loss: weighted sum
            loss_dict: component breakdown for logging
        """
        # CE loss
        ce_loss = F.cross_entropy(output, target)

        # Voltage regularization: penalize voltages far from reset
        voltage = states["voltage"]  # (T, batch, n_neuron)
        v_reg = torch.mean((voltage - self.config.v_reset) ** 2)

        # Rate regularization: encourage firing rates in target range
        spikes = states["spikes"]  # (T, batch, n_neuron)
        spike_count = spikes.sum(dim=0)  # (batch, n_neuron)
        duration_sec = T * self.dt / 1000.0  # Convert ms to s
        firing_rate = spike_count / duration_sec  # Hz

        # Penalize rates outside target range
        rate_penalty = torch.where(
            firing_rate < self.config.rate_target_min,
            (self.config.rate_target_min - firing_rate) ** 2,
            torch.where(
                firing_rate > self.config.rate_target_max,
                (firing_rate - self.config.rate_target_max) ** 2,
                torch.zeros_like(firing_rate),
            ),
        ).mean()

        # Total loss
        total = (
            self.config.ce_weight * ce_loss
            + self.config.voltage_weight * v_reg
            + self.config.rate_weight * rate_penalty
        )

        loss_dict = {
            "ce": ce_loss.item(),
            "voltage": v_reg.item(),
            "rate": rate_penalty.item(),
            "total": total.item(),
        }

        return total, loss_dict
