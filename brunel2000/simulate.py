"""Simulation loop for Brunel (2000) RSNN models."""

from dataclasses import dataclass

import torch

from btorch.datasets.noise import poisson_noise
from btorch.models import environ, functional
from btorch.models.init import uniform_v_
from btorch.models.rnn import RecurrentNN

from brunel2000.config import ModelAConfig, ModelBConfig, BrunelConfig


@dataclass
class SimulationResult:
    spikes: torch.Tensor  # (T, batch, n_neuron)
    voltages: torch.Tensor  # (T, batch, n_neuron)
    psc: torch.Tensor  # (T, batch, n_neuron)
    n_e: int
    dt_ms: float


def generate_poisson_input(
    T: int,
    batch: int,
    n_neuron: int,
    rate_hz: float,
    amplitude: float,
    dt_ms: float,
    device: str,
    rng: torch.Generator,
) -> torch.Tensor:
    """Generate Poisson spike input currents.

    One spike contributes exactly one synaptic efficacy unit (J), matching
    delta-shot current updates in the Brunel (2000) formulation.
    """
    counts = poisson_noise(
        batch,
        n_neuron,
        rate=rate_hz,
        T=T,
        dt=dt_ms / 1000.0,
        device=torch.device(device),
        dtype=torch.float32,
        generator=rng,
    )
    return counts * amplitude


def _simulate_with_input(
    model: RecurrentNN,
    ext_input: torch.Tensor,
    n_e: int,
    dt_ms: float,
    device: str,
) -> SimulationResult:
    batch = ext_input.shape[1]
    functional.init_net_state(model, batch_size=batch, device=device)
    functional.reset_net(model, batch_size=batch)
    uniform_v_(model.neuron, set_reset_value=True)

    with environ.context(dt=dt_ms), torch.no_grad():
        spikes_rec, states = model(ext_input)

    v_key = (
        "neuron.v"
        if "neuron.v" in states
        else next(key for key in states.keys() if key.endswith(".v"))
    )
    psc_key = "synapse.psc" if "synapse.psc" in states else None
    if psc_key is None:
        psc_candidates = [key for key in states.keys() if "psc" in key]
        if psc_candidates:
            psc_key = psc_candidates[0]
    psc_rec = (
        states[psc_key] if psc_key is not None else torch.zeros_like(states[v_key])
    )

    return SimulationResult(
        spikes=spikes_rec,
        voltages=states[v_key],
        psc=psc_rec,
        n_e=n_e,
        dt_ms=dt_ms,
    )


def simulate_model_a(
    model: RecurrentNN, config: BrunelConfig, rng: torch.Generator
) -> SimulationResult:
    mcfg: ModelAConfig = config.model
    dt = config.sim.dt_ms
    T = (
        int(mcfg.duration_ms / dt)
        if hasattr(mcfg, "duration_ms")
        else int(config.sim.duration_ms / dt)
    )
    T = int(config.sim.duration_ms / dt)
    device = config.sim.device
    batch = 1
    n_e = int(mcfg.n_neurons * mcfg.n_e_ratio)

    ext_rate = mcfg.c_ext * mcfg.nu_ext_hz
    ext_input = generate_poisson_input(
        T, batch, mcfg.n_neurons, ext_rate, mcfg.j, dt, device, rng
    )
    return _simulate_with_input(model, ext_input, n_e=n_e, dt_ms=dt, device=device)


def simulate_model_b(
    model: RecurrentNN, config: BrunelConfig, rng: torch.Generator
) -> SimulationResult:
    mcfg: ModelBConfig = config.model
    dt = config.sim.dt_ms
    T = int(config.sim.duration_ms / dt)
    device = config.sim.device
    batch = 1
    n_e = int(mcfg.n_neurons * mcfg.n_e_ratio)

    ext_rate_e = mcfg.c_ext * mcfg.nu_e_ext_hz
    ext_rate_i = mcfg.c_ext * mcfg.nu_i_ext_hz

    ext_input_e = generate_poisson_input(
        T, batch, n_e, ext_rate_e, mcfg.j_e, dt, device, rng
    )
    ext_input_i = generate_poisson_input(
        T, batch, mcfg.n_neurons - n_e, ext_rate_i, mcfg.j_i, dt, device, rng
    )
    ext_input = torch.cat([ext_input_e, ext_input_i], dim=-1)
    return _simulate_with_input(model, ext_input, n_e=n_e, dt_ms=dt, device=device)


def run_simulation(model, config: BrunelConfig, seed: int = 42) -> SimulationResult:
    """Dispatch to the appropriate simulation loop."""
    rng = torch.Generator(device=config.sim.device)
    rng.manual_seed(seed)

    if isinstance(config.model, ModelAConfig):
        return simulate_model_a(model, config, rng)
    elif isinstance(config.model, ModelBConfig):
        return simulate_model_b(model, config, rng)
    else:
        raise ValueError(f"Unknown model config type: {type(config.model)}")
