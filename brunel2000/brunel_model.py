"""Brunel 2000 network model using btorch RSNN.

Implements the sparsely connected E/I network from Brunel 2000 using btorch
stateful neuron and synapse modules with RNN wrappers.

Model A: Identical E/I neurons (single neuron type, single delay)
Model B: Heterogeneous E/I neurons (different τ_m for E/I populations)

Usage:
    cfg = load_config(BrunelConfig)
    model = BrunelNetwork(cfg)

    functional.reset_net_state(model, batch_size=1)
    uniform_v_(model.neuron, set_reset_value=True)

    with environ.context(dt=cfg.sim.dt):
        spikes, states = model(external_input)
"""

import math
import torch
import torch.nn as nn
import scipy.sparse
from typing import Tuple, Optional

from btorch.models import environ, functional, init
from btorch.models.neurons import LIF
from btorch.models.linear import SparseConn
from btorch.models.rnn import RecurrentNN

from brunel_config import (
    BrunelConfig,
    ModelANeuronConfig,
    ModelASynapseConfig,
    ModelBNeuronConfig,
    ModelBSynapseConfig,
    compute_derived_params,
    get_model_params,
)


class BrunelNetwork(nn.Module):
    """Brunel 2000 sparsely connected E/I network.

    Architecture:
        - N_E excitatory neurons, N_I inhibitory neurons
        - Sparse random connectivity with probability ε
        - Delta-function PSCs with constant delay D
        - External Poisson input

    Model A: Single neuron type for all neurons
    Model B: Separate GLIF3 instances for E and I populations (different τ_m)

    Args:
        cfg: BrunelConfig with all network parameters

    Example:
        >>> cfg = load_config(BrunelConfig)
        >>> model = BrunelNetwork(cfg)
        >>> functional.reset_net_state(model, batch_size=1)
        >>> uniform_v_(model.neuron, set_reset_value=True)
        >>> with environ.context(dt=0.1):
        ...     spikes, states = model(external_input)
    """

    def __init__(self, cfg: BrunelConfig):
        super().__init__()
        self.cfg = cfg

        # Compute derived parameters
        self.derived = compute_derived_params(cfg)
        self.p = get_model_params(cfg)

        net = cfg.network
        neu = cfg.neuron

        self.n_exc = net.n_exc
        self.n_inh = net.n_inh
        self.n_total = net.n_exc + net.n_inh
        self.delay_steps = int(round(cfg.synapse.delay / cfg.sim.dt))

        # Create sparse connectivity matrices
        self._create_connectivity()

        # Create neurons (Model A: single type, Model B: separate E/I)
        # Using LIF with c_m=1 so that input current directly adds to voltage
        if isinstance(neu, ModelANeuronConfig):
            # Model A: Single neuron type for all neurons
            self.neuron = LIF(
                n_neuron=self.n_total,
                tau=neu.tau_m,
                v_threshold=neu.v_thresh,
                v_reset=neu.v_reset,
                tau_ref=neu.tau_rp,
                c_m=1.0,  # Direct current-to-voltage conversion
            )
            self.e_neuron = None
            self.i_neuron = None
        elif isinstance(neu, ModelBNeuronConfig):
            # Model B: Separate E and I neuron populations
            self.e_neuron = LIF(
                n_neuron=self.n_exc,
                tau=neu.tau_e,
                v_threshold=neu.v_thresh,
                v_reset=neu.v_reset,
                tau_ref=neu.tau_rp,
                c_m=1.0,
            )
            self.i_neuron = LIF(
                n_neuron=self.n_inh,
                tau=neu.tau_i,
                v_threshold=neu.v_thresh,
                v_reset=neu.v_reset,
                tau_ref=neu.tau_rp,
                c_m=1.0,
            )
            # Wrapper for unified interface
            self.neuron = _HeterogeneousNeurons(self.e_neuron, self.i_neuron)
        else:
            raise ValueError(f"Unknown neuron config type: {type(neu)}")

        # Create synapses (delta-function PSCs)
        self._create_synapses()

        # Create RNN wrapper
        self._create_rnn()

    def _create_connectivity(self):
        """Create sparse connectivity matrices for E→E, E→I, I→E, I→I."""
        net = self.cfg.network
        p = self.p

        c_exc = self.derived["c_exc"]
        c_inh = self.derived["c_inh"]

        # Weight matrices: rows = targets, cols = sources
        # Shape: (n_targets, n_sources)

        # E→E connections (excitatory to excitatory)
        w_ee = self._create_sparse_weights(
            n_pre=self.n_exc,
            n_post=self.n_exc,
            n_conn=c_exc,
            weight=p["J_e"],
        )

        # E→I connections (excitatory to inhibitory)
        w_ei = self._create_sparse_weights(
            n_pre=self.n_exc,
            n_post=self.n_inh,
            n_conn=c_exc,
            weight=p["J_i"],
        )

        # I→E connections (inhibitory to excitatory)
        w_ie = self._create_sparse_weights(
            n_pre=self.n_inh,
            n_post=self.n_exc,
            n_conn=c_inh,
            weight=-p["g_e"] * p["J_e"],  # Negative = inhibitory
        )

        # I→I connections (inhibitory to inhibitory)
        w_ii = self._create_sparse_weights(
            n_pre=self.n_inh,
            n_post=self.n_inh,
            n_conn=c_inh,
            weight=-p["g_i"] * p["J_i"],  # Negative = inhibitory
        )

        # Combine into block matrix for full connectivity
        # [w_ee  w_ie ]   [E_targets × E_sources  E_targets × I_sources]
        # [w_ei  w_ii ]   [I_targets × E_sources  I_targets × I_sources]

        top = torch.cat([w_ee, w_ie], dim=1)  # E_targets × (E_sources + I_sources)
        bottom = torch.cat([w_ei, w_ii], dim=1)  # I_targets × (E_sources + I_sources)
        w_full = torch.cat([top, bottom], dim=0)  # (E+I)_targets × (E+I)_sources

        # Convert to scipy sparse (COO format) for SparseConn
        w_sparse = scipy.sparse.coo_array(w_full.numpy())

        # Create sparse connection
        self.conn = SparseConn(conn=w_sparse, bias=None)

        # Store individual weights for analysis
        self.w_ee = w_ee
        self.w_ei = w_ei
        self.w_ie = w_ie
        self.w_ii = w_ii

    def _create_sparse_weights(
        self,
        n_pre: int,
        n_post: int,
        n_conn: int,
        weight: float,
    ) -> torch.Tensor:
        """Create sparse weight matrix with fixed number of connections per neuron.

        Args:
            n_pre: Number of presynaptic neurons
            n_post: Number of postsynaptic neurons
            n_conn: Number of connections per postsynaptic neuron
            weight: Synaptic weight value

        Returns:
            Weight matrix of shape (n_post, n_pre)
        """
        w = torch.zeros(n_post, n_pre)

        # For each postsynaptic neuron, choose n_conn random presynaptic sources
        for i in range(n_post):
            # Random presynaptic indices (without replacement)
            indices = torch.randperm(n_pre)[:n_conn]
            w[i, indices] = weight

        return w

    def _create_synapses(self):
        """Create synaptic processing modules."""
        # For delta-function PSCs, we use a simple current-based synapse
        # that just passes through the weighted spikes

        # The synapse simply computes: I_syn = conn(spikes)
        # This is handled by the RecurrentNN wrapper
        pass

    def _create_rnn(self):
        """Create RNN wrapper for time-stepped simulation."""
        # Custom forward that handles delay and external input
        self.rnn = _BrunelRNN(
            neuron=self.neuron,
            conn=self.conn,
            n_exc=self.n_exc,
            n_inh=self.n_inh,
            delay_steps=self.delay_steps,
        )

    def forward(
        self,
        external_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Forward pass through the network.

        Args:
            external_input: External input current [T, batch, n_neurons]
                or [T, n_neurons] for single batch

        Returns:
            spikes: Spike tensor [T, batch, n_neurons]
            states: Dictionary of neuron states
        """
        # Handle single batch case
        if external_input.ndim == 2:
            external_input = external_input.unsqueeze(1)

        # Run RNN
        spikes, states = self.rnn(external_input)

        return spikes, states

    def generate_external_input(
        self,
        duration_ms: Optional[float] = None,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Generate external Poisson input.

        Args:
            duration_ms: Simulation duration in ms (default: cfg.sim.duration)
            batch_size: Batch size

        Returns:
            External input current [T, batch, n_neurons]
        """
        if duration_ms is None:
            duration_ms = self.cfg.sim.duration

        dt = self.cfg.sim.dt
        n_steps = int(duration_ms / dt)

        p = self.p

        # External rate per neuron (Hz)
        nu_ext_e = p["nu_ext_e"]  # Rate to E neurons
        nu_ext_i = p["nu_ext_i"]  # Rate to I neurons

        # Number of external connections per neuron
        c_ext = self.derived["c_ext"]

        # Poisson rate per synapse per time step
        # Each neuron receives C_ext independent Poisson processes with rate nu_ext
        # The sum of C_ext independent Poisson(rate) is Poisson(C_ext * rate)
        total_rate_e = (
            nu_ext_e * c_ext * dt / 1000.0
        )  # Total external spikes per neuron per step
        total_rate_i = nu_ext_i * c_ext * dt / 1000.0

        # Generate external spikes directly (sum of C_ext Poisson processes)
        ext_current = torch.zeros(n_steps, batch_size, self.n_total)
        ext_current[:, :, : self.n_exc] = (
            torch.poisson(torch.full((n_steps, batch_size, self.n_exc), total_rate_e))
            * p["J_e"]
        )
        ext_current[:, :, self.n_exc :] = (
            torch.poisson(torch.full((n_steps, batch_size, self.n_inh), total_rate_i))
            * p["J_i"]
        )

        return ext_current

    def get_population_spikes(
        self,
        spikes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split spikes into E and I populations.

        Args:
            spikes: Spike tensor [T, batch, n_neurons]

        Returns:
            spikes_e: [T, batch, n_exc]
            spikes_i: [T, batch, n_inh]
        """
        spikes_e = spikes[..., : self.n_exc]
        spikes_i = spikes[..., self.n_exc :]
        return spikes_e, spikes_i


class _HeterogeneousNeurons(nn.Module):
    """Wrapper for separate E and I neuron populations.

    Provides unified interface for Model B where E and I have
    different parameters.
    """

    def __init__(self, e_neuron: LIF, i_neuron: LIF):
        super().__init__()
        self.e_neuron = e_neuron
        self.i_neuron = i_neuron
        self.n_neuron = e_neuron.n_neuron + i_neuron.n_neuron

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining E and I populations.

        Args:
            x: Input current [..., n_neurons]

        Returns:
            Spikes [..., n_neurons]
        """
        n_exc = self.e_neuron.n_neuron

        # Split input
        x_e = x[..., :n_exc]
        x_i = x[..., n_exc:]

        # Process each population
        spikes_e = self.e_neuron(x_e)
        spikes_i = self.i_neuron(x_i)

        # Combine
        return torch.cat([spikes_e, spikes_i], dim=-1)

    @property
    def v(self):
        """Combined membrane potential."""
        return torch.cat([self.e_neuron.v, self.i_neuron.v], dim=-1)


class _BrunelRNN(nn.Module):
    """Custom RNN for Brunel network with delay handling.

    Handles:
        - Synaptic delay via circular buffer
        - External input integration
        - Recurrent connectivity
    """

    def __init__(
        self,
        neuron: nn.Module,
        conn: SparseConn,
        n_exc: int,
        n_inh: int,
        delay_steps: int,
    ):
        super().__init__()
        self.neuron = neuron
        self.conn = conn
        self.n_exc = n_exc
        self.n_inh = n_inh
        self.n_total = n_exc + n_inh
        self.delay_steps = delay_steps

        # Delay buffer for recurrent spikes
        # Shape: [delay_steps, batch, n_neurons]
        self.register_buffer(
            "delay_buffer",
            torch.zeros(delay_steps, 1, self.n_total),
            persistent=False,
        )
        self.buffer_idx = 0

    def forward(
        self,
        external_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Forward pass through time.

        Args:
            external_input: [T, batch, n_neurons]

        Returns:
            spikes: [T, batch, n_neurons]
            states: Dictionary with neuron states
        """
        T, batch, n_neurons = external_input.shape

        # Initialize delay buffer if needed
        if self.delay_buffer.shape[1] != batch:
            self.delay_buffer = torch.zeros(
                self.delay_steps,
                batch,
                self.n_total,
                device=external_input.device,
            )

        # Storage for outputs
        spikes_list = []
        v_list = []

        # Time-stepped simulation
        for t in range(T):
            # Get delayed recurrent spikes
            delayed_spikes = self.delay_buffer[self.buffer_idx]

            # Compute recurrent input: I_rec = W @ s_delayed
            i_rec = self.conn(delayed_spikes)

            # Total input: I_total = I_rec + I_ext
            i_total = i_rec + external_input[t]

            # Neuron dynamics
            spikes = self.neuron(i_total)

            # Store outputs
            spikes_list.append(spikes)
            if hasattr(self.neuron, "v"):
                v_list.append(self.neuron.v)

            # Update delay buffer with current spikes
            self.delay_buffer[self.buffer_idx] = spikes
            self.buffer_idx = (self.buffer_idx + 1) % self.delay_steps

        # Stack outputs
        spikes_out = torch.stack(spikes_list, dim=0)

        states = {
            "spikes": spikes_out,
        }
        if v_list:
            states["v"] = torch.stack(v_list, dim=0)

        return spikes_out, states

    def reset_state(self, batch_size: int = 1):
        """Reset delay buffer and neuron states."""
        self.delay_buffer.zero_()
        self.buffer_idx = 0
        functional.reset_net_state(self.neuron, batch_size=batch_size)


def create_model(cfg: BrunelConfig) -> BrunelNetwork:
    """Create and initialize Brunel network model.

    Args:
        cfg: Configuration

    Returns:
        Initialized model
    """
    model = BrunelNetwork(cfg)

    # Initialize with reset values (deterministic)
    functional.init_net_state(model, batch_size=1)
    init.uniform_v_(model.neuron, set_reset_value=True)

    return model


if __name__ == "__main__":
    from brunel_config import load_config

    # Test Model A
    print("Testing Model A...")
    cfg = load_config()
    cfg.network.n_exc = 100
    cfg.network.n_inh = 25
    cfg.sim.duration = 100.0

    model = create_model(cfg)
    print(f"Model created: {cfg.network.n_exc} E, {cfg.network.n_inh} I neurons")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Generate external input
    ext_input = model.generate_external_input(batch_size=1)
    print(f"External input shape: {ext_input.shape}")

    # Run simulation
    functional.reset_net_state(model, batch_size=1)
    init.uniform_v_(model.neuron, set_reset_value=True)

    with environ.context(dt=cfg.sim.dt):
        spikes, states = model(ext_input)

    print(f"Output spikes shape: {spikes.shape}")
    print(f"Total spikes: {spikes.sum().item():.0f}")

    # Get population firing rates
    spikes_e, spikes_i = model.get_population_spikes(spikes)
    rate_e = spikes_e.sum() / (cfg.network.n_exc * cfg.sim.duration / 1000.0)
    rate_i = spikes_i.sum() / (cfg.network.n_inh * cfg.sim.duration / 1000.0)
    print(f"E firing rate: {rate_e.item():.1f} Hz")
    print(f"I firing rate: {rate_i.item():.1f} Hz")
