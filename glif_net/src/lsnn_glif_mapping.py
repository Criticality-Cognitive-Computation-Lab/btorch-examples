"""LSNN threshold adaptation <-> GLIF ASC mapping and validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MappingResult:
    tau_a: float
    k: float
    beta: float
    B0: float
    dt: float
    rho: float
    tau_m: float
    c_m: float
    asc_amps_continuous: float
    asc_amps_bernoulli: float
    spike_scaling: str


def calibrate_asc_amps(
    *, beta: float, tau_a: float, tau_m: float, c_m: float, dt: float
) -> MappingResult:
    if tau_a <= 0 or tau_m <= 0 or c_m <= 0 or dt <= 0:
        raise ValueError("tau_a, tau_m, c_m, dt must be positive")
    rho = float(np.exp(-dt / tau_a))
    asc_cont = -(c_m / tau_m) * beta
    asc_bern = asc_cont * (1.0 - rho)
    return MappingResult(
        tau_a=tau_a,
        k=1.0 / tau_a,
        beta=beta,
        B0=0.0,
        dt=dt,
        rho=rho,
        tau_m=tau_m,
        c_m=c_m,
        asc_amps_continuous=asc_cont,
        asc_amps_bernoulli=asc_bern,
        spike_scaling="continuous_rate_or_bernoulli",
    )


def lsnn_to_glif_symbol_table(
    *, beta: float, tau_a: float, B0: float, dt: float, tau_m: float, c_m: float
) -> dict[str, str | float]:
    mapped = calibrate_asc_amps(beta=beta, tau_a=tau_a, tau_m=tau_m, c_m=c_m, dt=dt)
    return {
        "tau_a": tau_a,
        "k": mapped.k,
        "beta": beta,
        "asc_amps_continuous": mapped.asc_amps_continuous,
        "asc_amps_bernoulli": mapped.asc_amps_bernoulli,
        "B0": B0,
        "v_threshold": B0,
        "dt": dt,
        "environ.dt": dt,
        "z_convention": "align 0/1 vs 0/1/dt with asc_amps scaling",
    }


def _lsnn_b_from_spikes(spikes: np.ndarray, rho: float) -> np.ndarray:
    b = np.zeros_like(spikes, dtype=np.float64)
    for t in range(1, spikes.shape[0]):
        b[t] = rho * b[t - 1] + (1.0 - rho) * spikes[t - 1]
    return b


def _glif_iasc_from_spikes(
    spikes: np.ndarray, rho: float, asc_amp: float
) -> np.ndarray:
    iasc = np.zeros_like(spikes, dtype=np.float64)
    for t in range(1, spikes.shape[0]):
        iasc[t] = rho * iasc[t - 1] + asc_amp * spikes[t - 1]
    return iasc


def validate_equivalence(
    *,
    beta: float,
    tau_a: float,
    tau_m: float,
    c_m: float,
    dt: float,
    impulse_steps: int = 200,
    periodic_steps: int = 1200,
    period: int = 25,
) -> dict[str, float]:
    mapped = calibrate_asc_amps(beta=beta, tau_a=tau_a, tau_m=tau_m, c_m=c_m, dt=dt)
    rho = mapped.rho

    spikes = np.zeros((impulse_steps, 1), dtype=np.float64)
    spikes[0, 0] = 1.0
    b = _lsnn_b_from_spikes(spikes, rho=rho)
    iasc = _glif_iasc_from_spikes(spikes, rho=rho, asc_amp=mapped.asc_amps_bernoulli)
    dv_eq = -(tau_m / c_m) * iasc
    imp_err = np.abs(dv_eq - beta * b)

    periodic_spikes = np.zeros((periodic_steps, 1), dtype=np.float64)
    periodic_spikes[:: max(period, 1), 0] = 1.0
    b_p = _lsnn_b_from_spikes(periodic_spikes, rho=rho)
    iasc_p = _glif_iasc_from_spikes(
        periodic_spikes, rho=rho, asc_amp=mapped.asc_amps_bernoulli
    )
    dv_eq_p = -(tau_m / c_m) * iasc_p
    p_err = np.abs(dv_eq_p - beta * b_p)

    settle = slice(periodic_steps // 2, periodic_steps)
    return {
        "impulse_max_abs_error": float(imp_err.max()),
        "impulse_mean_abs_error": float(imp_err.mean()),
        "periodic_max_abs_error": float(p_err.max()),
        "periodic_mean_abs_error": float(p_err.mean()),
        "periodic_steady_max_abs_error": float(p_err[settle].max()),
        "periodic_steady_mean_abs_error": float(p_err[settle].mean()),
        "rho": rho,
        "k": mapped.k,
        "asc_amps_bernoulli": mapped.asc_amps_bernoulli,
        "asc_amps_continuous": mapped.asc_amps_continuous,
    }
