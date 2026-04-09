"""GRU baseline model for comparison."""

import torch
import torch.nn as nn


class GRUBaseline(nn.Module):
    """GRU baseline for comparison with GLIF RSNN."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        response_window: float = 0.8,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.response_window = response_window

        # GRU layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=False,
            bidirectional=bidirectional,
        )

        # Output layer
        multiplier = 2 if bidirectional else 1
        self.output_linear = nn.Linear(hidden_size * multiplier, output_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Args:
            x: (T, batch, input_dim) input

        Returns:
            output: (batch, output_dim) logits
            states: dict with hidden states
        """
        T, batch_size, _ = x.shape

        # GRU forward
        gru_out, hidden = self.gru(x)  # gru_out: (T, batch, hidden)

        # Use response window for readout
        start = int(T * (1 - self.response_window))
        response = gru_out[start:]  # (window_T, batch, hidden)

        # Average over time and apply output layer
        avg_response = response.mean(dim=0)  # (batch, hidden)
        output = self.output_linear(avg_response)

        # Dummy states for compatibility
        states = {
            "spikes": torch.zeros_like(gru_out[..., 0]),  # (T, batch)
            "voltage": gru_out,  # Use GRU output as proxy
            "psc": gru_out,
            "filtered_spikes": gru_out,
        }

        return output, states
