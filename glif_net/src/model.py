"""Single-layer GLIF RSNN with adaptation currents and state tracing."""

from __future__ import annotations

import torch
import torch.nn as nn

from btorch.models import environ, functional
from btorch.models.init import uniform_v_
from btorch.models.linear import LearnableScale, SparseConn
from btorch.models.neurons import GLIF3
from btorch.models.rnn import RecurrentNN
from btorch.models.synapse import ExponentialPSC

from .connectivity import (
    ConnectivitySpec,
    build_sparse_mat,
    sparseconn_dale_violations,
)


class SingleLayerGLIFRSNN(nn.Module):
    """Single recurrent GLIF layer with readout and adaptation currents."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_neuron: int = 300,
        n_e_ratio: float = 0.8,
        n_adapt: int = 120,
        asc_amp: float = -0.2,
        tau_adapt: float = 700.0,
        tau_adapt_min: float | None = None,
        tau_adapt_max: float | None = None,
        tau: float = 20.0,
        tau_syn: float = 5.0,
        v_threshold: float = -45.0,
        v_reset: float = -60.0,
        tau_ref: float | None = 0.0,
        input_scale: float = 1.0,
        output_scale: float = 1.0,
        response_window: float = 0.8,
        readout_tau: float = 20.0,
        dt: float = 1.0,
        connectivity_density: float = 1.0,
        i_e_ratio: float = 100.0,
        e_to_e_mean: float = 4.0e-3,
        e_to_e_std: float = 1.9e-3,
        e_i_mean: float = 5.0e-2,
        i_i_mean: float = 25e-4,
        conn_seed: int | None = None,
        grad_checkpoint: bool = False,
        unroll: int | bool = 8,
        chunk_size: int | None = None,
        cpu_offload: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_neuron = n_neuron
        self.n_adapt = n_adapt
        self.response_window = response_window
        self.readout_tau = readout_tau
        self.dt = float(dt)

        self.n_e = int(round(n_neuron * n_e_ratio))
        self.n_i = n_neuron - self.n_e
        if self.n_e <= 0 or self.n_i < 0:
            raise ValueError("invalid E/I split")

        self.connectivity_spec = ConnectivitySpec(
            density=connectivity_density,
            i_e_ratio=i_e_ratio,
            e_to_e_mean=e_to_e_mean,
            e_to_e_std=e_to_e_std,
            e_i_mean=e_i_mean,
            i_i_mean=i_i_mean,
        )
        weights, self.e_idx, self.i_idx = build_sparse_mat(
            n_e_neurons=self.n_e,
            n_i_neurons=self.n_i,
            density=connectivity_density,
            i_e_ratio=i_e_ratio,
            e_to_e_mean=e_to_e_mean,
            e_to_e_std=e_to_e_std,
            e_i_mean=e_i_mean,
            i_i_mean=i_i_mean,
            seed=conn_seed,
        )

        self.input_linear = nn.Linear(input_dim, n_neuron, bias=False)
        self.input_scale = LearnableScale(scale=input_scale, bias=None)

        self.recurrent_conn = SparseConn(conn=weights, enforce_dale=True)

        asc_amps = self._build_adaptation_amps(n_neuron, n_adapt, asc_amp)
        k = self._build_k_values(
            n_neuron=n_neuron,
            n_adapt=n_adapt,
            tau_adapt=tau_adapt,
            tau_adapt_min=tau_adapt_min,
            tau_adapt_max=tau_adapt_max,
        )
        self.neuron = GLIF3(
            n_neuron=n_neuron,
            v_threshold=v_threshold,
            v_reset=v_reset,
            tau=tau,
            tau_ref=tau_ref,
            asc_amps=asc_amps,
            k=k,
        )
        self.synapse = ExponentialPSC(
            n_neuron=n_neuron,
            tau_syn=tau_syn,
            linear=self.recurrent_conn,
        )

        self.rnn = RecurrentNN(
            neuron=self.neuron,
            synapse=self.synapse,
            step_mode="m",
            update_state_names=("neuron.v", "neuron.Iasc", "synapse.psc"),
            grad_checkpoint=grad_checkpoint,
            unroll=unroll,
            chunk_size=chunk_size,
            cpu_offload=cpu_offload,
        )

        self.output_linear = nn.Linear(n_neuron, output_dim, bias=True)
        self.output_scale = LearnableScale(scale=output_scale, bias=None)

        self.readout_alpha = float(
            torch.exp(torch.tensor(-self.dt / readout_tau)).item()
        )
        self.register_buffer("readout_filter_state", torch.zeros(1), persistent=False)
        self._state_initialized = False

    def _build_adaptation_amps(
        self,
        n_neuron: int,
        n_adapt: int,
        asc_amp: float,
    ) -> torch.Tensor:
        asc_amps = torch.zeros(n_neuron, 1, dtype=torch.float32)
        if n_adapt == 0:
            asc_amps[:, 0] = asc_amp
            return asc_amps
        if n_adapt == -1:
            asc_amps[: n_neuron // 2, 0] = asc_amp
            return asc_amps
        if n_adapt > 0:
            asc_amps[: min(n_adapt, n_neuron), 0] = asc_amp
        return asc_amps

    def _build_k_values(
        self,
        n_neuron: int,
        n_adapt: int,
        tau_adapt: float,
        tau_adapt_min: float | None,
        tau_adapt_max: float | None,
    ) -> torch.Tensor:
        k = torch.full((n_neuron, 1), 1.0 / tau_adapt, dtype=torch.float32)
        if tau_adapt_min is None or tau_adapt_max is None:
            return k
        if tau_adapt_min <= 0 or tau_adapt_max <= 0:
            raise ValueError("tau_adapt_min/tau_adapt_max must be positive")
        n_heter = (
            n_neuron if n_adapt == 0 else (n_neuron // 2 if n_adapt == -1 else n_adapt)
        )
        n_heter = max(0, min(n_neuron, n_heter))
        if n_heter == 0:
            return k
        taus = torch.empty(n_heter).uniform_(tau_adapt_min, tau_adapt_max)
        k[:n_heter, 0] = 1.0 / taus
        return k

    def get_response_window(self, T: int) -> slice:
        start = int(T * (1.0 - self.response_window))
        return slice(start, T)

    def _lowpass_filter(self, spikes: torch.Tensor) -> torch.Tensor:
        filtered = torch.zeros_like(spikes)
        filtered[0] = spikes[0]
        alpha = self.readout_alpha
        for t in range(1, spikes.shape[0]):
            filtered[t] = alpha * filtered[t - 1] + (1.0 - alpha) * spikes[t]
        return filtered

    def apply_dale_projection(self) -> None:
        # Native btorch SparseConn Dale support.
        if hasattr(self.recurrent_conn, "constrain"):
            self.recurrent_conn.constrain()

    def dale_violations(self) -> dict[str, float]:
        return sparseconn_dale_violations(self.recurrent_conn, n_e_neurons=self.n_e)

    def reset_state(self, batch_size: int, device: torch.device | None = None) -> None:
        if not self._state_initialized:
            functional.init_net_state(self.rnn, batch_size=batch_size, device=device)
            self._state_initialized = True
        functional.reset_net(self.rnn, batch_size=batch_size, device=device)

    def init_voltage(self, low: float = -70.0, high: float = -40.0) -> None:
        if not self._state_initialized:
            functional.init_net_state(self.rnn, batch_size=1)
            self._state_initialized = True
        uniform_v_(self.neuron, low=low, high=high, set_reset_value=True)

    def forward(
        self,
        x: torch.Tensor,
        *,
        reset_state: bool = True,
        return_sequence: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward sequence through RSNN.

        Args:
            x: (T, batch, input_dim)
            reset_state: reset recurrent states before forward
            return_sequence: if True return (T, batch, output_dim)
                else average over response window to (batch, output_dim)
        """
        if x.ndim != 3:
            raise ValueError("x must be (T, batch, input_dim)")
        T, batch_size, _ = x.shape

        if reset_state:
            self.reset_state(batch_size=batch_size, device=x.device)

        x_scaled = self.input_scale(x)
        input_current = self.input_linear(x_scaled)

        with environ.context(dt=self.dt):
            spikes, rnn_states = self.rnn(input_current)

        filtered_spikes = self._lowpass_filter(spikes)
        out_dtype = self.output_linear.weight.dtype
        filtered_spikes_out = filtered_spikes.to(dtype=out_dtype)
        sequence_output = self.output_linear(self.output_scale(filtered_spikes_out))

        if return_sequence:
            output = sequence_output
        else:
            response_spikes = filtered_spikes_out[self.get_response_window(T)]
            avg_spikes = response_spikes.mean(dim=0)
            output = self.output_linear(self.output_scale(avg_spikes))

        states = {
            "spikes": spikes,
            "voltage": rnn_states["neuron.v"],
            "psc": rnn_states["synapse.psc"],
            "Iasc": rnn_states["neuron.Iasc"],
            "filtered_spikes": filtered_spikes,
            "sequence_output": sequence_output,
        }
        return output, states
