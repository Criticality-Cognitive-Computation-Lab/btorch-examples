"""Simulation loop for Brunel (2000) RSNN models."""

from dataclasses import dataclass

import torch

from btorch.models import environ, functional

from brunel2000.config import ModelAConfig, ModelBConfig, BrunelConfig
from brunel2000.model import ModelRSNN


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

    Weights are scaled by 1/dt_ms so that a discrete-time spike produces the
    same voltage jump as the paper's delta-function current.
    """
    mean_spikes = rate_hz * dt_ms / 1000.0
    scale = 1.0 / dt_ms
    counts = torch.poisson(
        torch.full((T, batch, n_neuron), mean_spikes, device=device), generator=rng
    )
    return counts.float() * amplitude * scale


def simulate_model_a(
    model: ModelRSNN, config: BrunelConfig, rng: torch.Generator
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

    model.reset_state(batch_size=batch)

    ext_rate = mcfg.c_ext * mcfg.nu_ext_hz
    ext_input = generate_poisson_input(
        T, batch, mcfg.n_neurons, ext_rate, mcfg.j, dt, device, rng
    )

    spikes_rec = torch.zeros(T, batch, mcfg.n_neurons, device=device)
    volts_rec = torch.zeros(T, batch, mcfg.n_neurons, device=device)
    psc_rec = torch.zeros(T, batch, mcfg.n_neurons, device=device)

    with environ.context(dt=dt), torch.no_grad():
        for t in range(T):
            z, _ = model.rnn(ext_input[t])
            spikes_rec[t] = z
            volts_rec[t] = model.neuron.v
            psc_rec[t] = model.synapse.psc

    return SimulationResult(
        spikes=spikes_rec,
        voltages=volts_rec,
        psc=psc_rec,
        n_e=n_e,
        dt_ms=dt,
    )


def simulate_model_b(
    model: ModelRSNN, config: BrunelConfig, rng: torch.Generator
) -> SimulationResult:
    mcfg: ModelBConfig = config.model
    dt = config.sim.dt_ms
    T = int(config.sim.duration_ms / dt)
    device = config.sim.device
    batch = 1
    n_e = int(mcfg.n_neurons * mcfg.n_e_ratio)

    model.reset_state(batch_size=batch)

    ext_rate_e = mcfg.c_ext * mcfg.nu_e_ext_hz
    ext_rate_i = mcfg.c_ext * mcfg.nu_i_ext_hz

    ext_input_e = generate_poisson_input(
        T, batch, n_e, ext_rate_e, mcfg.j_e, dt, device, rng
    )
    ext_input_i = generate_poisson_input(
        T, batch, mcfg.n_neurons - n_e, ext_rate_i, mcfg.j_i, dt, device, rng
    )
    ext_input = torch.cat([ext_input_e, ext_input_i], dim=-1)

    spikes_rec = torch.zeros(T, batch, mcfg.n_neurons, device=device)
    volts_rec = torch.zeros(T, batch, mcfg.n_neurons, device=device)
    psc_rec = torch.zeros(T, batch, mcfg.n_neurons, device=device)

    with environ.context(dt=dt), torch.no_grad():
        for t in range(T):
            z, _ = model.rnn(ext_input[t])
            spikes_rec[t] = z
            volts_rec[t] = model.neuron.v
            psc_rec[t] = model.synapse.psc

    return SimulationResult(
        spikes=spikes_rec,
        voltages=volts_rec,
        psc=psc_rec,
        n_e=n_e,
        dt_ms=dt,
    )


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
