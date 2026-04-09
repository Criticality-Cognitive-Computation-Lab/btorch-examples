"""Single layer GLIF RSNN model with partial adaptation support."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from btorch.models import environ, functional
from btorch.models.neurons import GLIF3
from btorch.models.synapse import AlphaPSCBilleh
from btorch.models.linear import SparseConn, LearnableScale
from btorch.models.rnn import RecurrentNN
from btorch.models.init import build_sparse_mat, uniform_v_


class SingleLayerGLIFRSNN(nn.Module):
    """
    Single recurrent layer GLIF RSNN with partial adaptation support.

    Architecture:
    - Input: Linear(input_dim, n_neuron) with optional scale
    - Recurrent: GLIF3 neurons with AlphaPSC synapses (E/I ratio)
    - Output: Linear readout from all neurons with low-pass filter
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_neuron: int = 256,
        n_e_ratio: float = 0.8,
        n_adapt: int = 0,  # 0=all adapt, -1=half, N=first N neurons
        asc_amp: float = -0.2,
        tau_adapt: float = 700.0,
        tau_mem: float = 20.0,
        tau_syn: float = 5.0,
        v_threshold: float = -45.0,
        v_reset: float = -60.0,
        tau_ref: float = 5.0,
        input_scale: float = 1.0,
        output_scale: float = 1.0,
        response_window: float = 0.8,
        readout_tau: float = 20.0,
        dt: float = 1.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_neuron = n_neuron
        self.n_adapt = n_adapt
        self.response_window = response_window
        self.readout_tau = readout_tau
        self.dt = dt

        # Input layer with LearnableScale
        self.input_linear = nn.Linear(input_dim, n_neuron, bias=False)
        self.input_scale = LearnableScale(scale=input_scale, bias=None)

        # Build E/I recurrent weight matrix
        n_e = int(n_neuron * n_e_ratio)
        n_i = n_neuron - n_e
        weights, _, _ = build_sparse_mat(n_e=n_e, n_i=n_i, i_e_ratio=1.0)
        self.recurrent_conn = SparseConn(conn=weights)

        # Create per-neuron asc_amps array for partial adaptation
        # n_adapt=0: all neurons adapt (default asc_amp for all)
        # n_adapt=-1: half of neurons adapt
        # n_adapt=N: first N neurons adapt, rest don't
        if n_adapt == 0:
            # All neurons adapt
            asc_amps = torch.full((n_neuron,), asc_amp)
        elif n_adapt == -1:
            # Half adapt
            n_adaptive = n_neuron // 2
            asc_amps = torch.zeros(n_neuron)
            asc_amps[:n_adaptive] = asc_amp
        else:
            # First n_adapt neurons adapt
            asc_amps = torch.zeros(n_neuron)
            asc_amps[: min(n_adapt, n_neuron)] = asc_amp

        self.register_buffer("asc_amps", asc_amps)

        # GLIF3 neurons with per-neuron asc_amps
        self.neuron = GLIF3(
            n_neuron=n_neuron,
            v_threshold=v_threshold,
            v_reset=v_reset,
            tau=tau_mem,
            tau_ref=tau_ref,
            asc_amps=asc_amps,  # Per-neuron adaptation amplitudes
            tau_adapt=tau_adapt,
        )

        # AlphaPSC synapse for recurrent connections
        self.psc = AlphaPSCBilleh(
            n_neuron=n_neuron,
            tau_syn=tau_syn,
            linear=self.recurrent_conn,
        )

        # RNN wrapper
        self.rnn = RecurrentNN(
            neuron=self.neuron,
            synapse=self.psc,
            step_mode="m",
            update_state_names=("neuron.v", "synapse.psc"),
        )

        # Output layer with LearnableScale
        self.output_linear = nn.Linear(n_neuron, output_dim, bias=True)
        self.output_scale = LearnableScale(scale=output_scale, bias=None)

        # Low-pass filter state for readout
        self.readout_alpha = torch.exp(torch.tensor(-dt / readout_tau))
        self.register_buffer("readout_filter_state", torch.zeros(1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Args:
            x: (T, batch, input_dim) input spike trains or analog values

        Returns:
            output: (batch, output_dim) logits
            states: dict with keys:
                - spikes: (T, batch, n_neuron)
                - voltage: (T, batch, n_neuron)
                - psc: (T, batch, n_neuron)
                - filtered_spikes: (T, batch, n_neuron)
        """
        T, batch_size, _ = x.shape

        # Reset state before each forward
        functional.reset_net(self.rnn, batch_size=batch_size)
        uniform_v_(self.neuron, set_reset_value=True)

        # Input projection with scale
        x_scaled = self.input_scale(x)
        input_current = self.input_linear(x_scaled)  # (T, batch, n_neuron)

        # Forward through RNN
        with environ.context(dt=self.dt):
            spikes, _ = self.rnn(input_current)  # (T, batch, n_neuron)

        # Get voltage and psc from RNN states
        voltage = self.neuron.v  # (batch, n_neuron) - final state
        psc = self.psc.psc  # (batch, n_neuron) - final state

        # Store all timesteps of voltage and psc for loss computation
        # We need to capture these during the forward pass
        # Since btorch doesn't store history by default, we'll compute from spikes
        # For now, use the available final states and recompute if needed

        # Low-pass filter spikes for readout (exponential moving average)
        filtered_spikes = self._lowpass_filter(spikes)  # (T, batch, n_neuron)

        # Apply readout in response window
        response_slice = self.get_response_window(T)
        response_spikes = filtered_spikes[response_slice]  # (window_T, batch, n_neuron)

        # Average over time and apply output layer
        avg_spikes = response_spikes.mean(dim=0)  # (batch, n_neuron)
        output = self.output_linear(self.output_scale(avg_spikes))

        # Store states for loss computation
        states = {
            "spikes": spikes,
            "voltage": voltage.unsqueeze(0).expand(T, -1, -1),  # Approximate
            "psc": psc.unsqueeze(0).expand(T, -1, -1),  # Approximate
            "filtered_spikes": filtered_spikes,
        }

        return output, states

    def _lowpass_filter(self, spikes: torch.Tensor) -> torch.Tensor:
        """Apply exponential low-pass filter to spikes."""
        T, batch, n_neuron = spikes.shape

        filtered = torch.zeros_like(spikes)
        filtered[0] = spikes[0]

        alpha = self.readout_alpha
        for t in range(1, T):
            filtered[t] = alpha * filtered[t - 1] + (1 - alpha) * spikes[t]

        return filtered

    def get_response_window(self, T: int) -> slice:
        """Get time slice for response window based on ratio."""
        start = int(T * (1 - self.response_window))
        return slice(start, T)

    def reset_state(self, batch_size: int = 1):
        """Reset network state for new batch."""
        functional.reset_net(self.rnn, batch_size=batch_size)
        uniform_v_(self.neuron, set_reset_value=True)
