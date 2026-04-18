"""Analysis and plotting for Brunel (2000) simulation results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from btorch.analysis import compute_spectrum, firing_rate, isi_cv_population
from btorch.analysis.dynamic_tools.ei_balance import compute_lag_correlation
from btorch.visualisation import plot_raster, plot_traces

from brunel2000.simulate import SimulationResult


def _to_spikes_2d(spikes: torch.Tensor) -> torch.Tensor:
    if spikes.ndim != 3:
        raise ValueError(
            f"Expected spikes shape (T, batch, n_neuron), got {spikes.shape}"
        )
    return spikes[:, 0, :]


def _as_scalar(value: torch.Tensor | np.ndarray | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        return float(np.asarray(value).item())
    return float(value)


def _downsample_raster_spikes(
    spikes: np.ndarray,
    n_e: int,
    max_neurons: int = 800,
) -> tuple[np.ndarray, np.ndarray]:
    n_neuron = spikes.shape[1]
    if n_neuron <= max_neurons:
        idx = np.arange(n_neuron)
        return spikes, idx

    e_count = int(round(max_neurons * (n_e / n_neuron)))
    e_count = min(max(e_count, 1), max_neurons - 1)
    i_count = max_neurons - e_count

    e_idx = np.linspace(0, max(n_e - 1, 0), num=e_count, dtype=int)
    i_idx = np.linspace(n_e, n_neuron - 1, num=i_count, dtype=int)
    keep = np.concatenate([e_idx, i_idx])
    keep = np.unique(keep)
    return spikes[:, keep], keep


def population_rate(
    spikes: torch.Tensor, dt_ms: float, bin_size_ms: float = 1.0
) -> np.ndarray:
    """Compute population firing rate in Hz using btorch smoothing API."""
    spikes_2d = _to_spikes_2d(spikes)
    width_steps = max(bin_size_ms / dt_ms, 1.0)
    rate = firing_rate(spikes_2d, width=width_steps, dt=dt_ms * 1e-3, axis=-1)
    return np.asarray(rate.detach().cpu())


def population_rate_by_pop(
    spikes: torch.Tensor, n_e: int, dt_ms: float, bin_size_ms: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Compute separate E and I population rates in Hz using btorch API."""
    spikes_2d = _to_spikes_2d(spikes)
    width_steps = max(bin_size_ms / dt_ms, 1.0)
    rate_e = firing_rate(
        spikes_2d[:, :n_e], width=width_steps, dt=dt_ms * 1e-3, axis=-1
    )
    rate_i = firing_rate(
        spikes_2d[:, n_e:], width=width_steps, dt=dt_ms * 1e-3, axis=-1
    )
    return np.asarray(rate_e.detach().cpu()), np.asarray(rate_i.detach().cpu())


def mean_firing_rate(spikes: torch.Tensor, dt_ms: float) -> dict[str, float]:
    """Mean firing rate per neuron in Hz using btorch firing-rate utility."""
    spikes_2d = _to_spikes_2d(spikes)
    per_step_rate_hz = firing_rate(spikes_2d, width=0, dt=dt_ms * 1e-3)
    per_neuron_rate_hz = np.asarray(per_step_rate_hz.detach().cpu()).mean(axis=0)
    n_neuron = per_neuron_rate_hz.shape[0]
    n_e = n_neuron // 2 if n_neuron % 2 == 0 else int(n_neuron * 0.8)
    return {
        "overall": float(per_neuron_rate_hz.mean()),
        "e": float(per_neuron_rate_hz[:n_e].mean()),
        "i": float(per_neuron_rate_hz[n_e:].mean()),
    }


def mean_firing_rate_result(result: SimulationResult) -> dict[str, float]:
    """Mean firing rate from SimulationResult in Hz."""
    per_step_rate_hz = firing_rate(
        _to_spikes_2d(result.spikes), width=0, dt=result.dt_ms * 1e-3
    )
    per_neuron_rate_hz = np.asarray(per_step_rate_hz.detach().cpu()).mean(axis=0)
    n_e = result.n_e
    return {
        "overall": float(per_neuron_rate_hz.mean()),
        "e": float(per_neuron_rate_hz[:n_e].mean()),
        "i": float(per_neuron_rate_hz[n_e:].mean()),
    }


def cv_isi(spikes: torch.Tensor, dt_ms: float) -> float:
    """Population ISI CV using btorch's pooled ISI API."""
    spikes_2d = _to_spikes_2d(spikes).detach().cpu().numpy()
    total_isi = 0
    for neuron_id in range(spikes_2d.shape[1]):
        spike_idx = np.flatnonzero(spikes_2d[:, neuron_id] > 0)
        if spike_idx.size >= 2:
            total_isi += spike_idx.size - 1
            if total_isi >= 2:
                break

    if total_isi < 2:
        return float("nan")

    cv_value, _ = isi_cv_population(spikes_2d, dt_ms=dt_ms, stat="cv")
    return _as_scalar(cv_value)


def phase_lag(rate_e: np.ndarray, rate_i: np.ndarray, dt_ms: float) -> dict[str, float]:
    """Compute E/I lag and dominant frequency using btorch analysis APIs."""
    if len(rate_e) < 10 or len(rate_i) < 10:
        return {"lag_ms": 0.0, "lag_deg": 0.0, "dominant_freq_hz": 0.0}

    _, lag_ms, _ = compute_lag_correlation(
        np.asarray(rate_e)[:, None],
        np.asarray(rate_i)[:, None],
        dt=dt_ms,
        max_lag_ms=min(30.0, (len(rate_e) - 1) * dt_ms),
    )
    lag_ms_value = _as_scalar(lag_ms)

    avg_rate = (np.asarray(rate_e) + np.asarray(rate_i)) / 2.0
    freqs, power = compute_spectrum(
        avg_rate,
        dt=dt_ms * 1e-3,
        nperseg=min(256, len(avg_rate)),
    )
    freqs = np.asarray(freqs)
    power = np.asarray(power)
    valid = freqs > 0
    dom_freq = float(freqs[valid][np.argmax(power[valid])]) if np.any(valid) else 0.0

    if dom_freq > 0.0:
        lag_deg = (lag_ms_value / 1000.0) * dom_freq * 360.0
        lag_deg = ((lag_deg + 180.0) % 360.0) - 180.0
    else:
        lag_deg = 0.0

    return {
        "lag_ms": float(lag_ms_value),
        "lag_deg": float(lag_deg),
        "dominant_freq_hz": float(dom_freq),
    }


def dominant_frequency(rate: np.ndarray, dt_ms: float) -> float:
    """Dominant non-zero frequency in Hz from a population rate trace."""
    if len(rate) < 10:
        return 0.0
    if np.allclose(rate, 0.0):
        return 0.0
    rate_arr = (
        np.asarray(rate.detach().cpu())
        if isinstance(rate, torch.Tensor)
        else np.asarray(rate)
    )
    freqs, power = compute_spectrum(
        rate_arr,
        dt=dt_ms * 1e-3,
        nperseg=min(256, len(rate_arr)),
    )
    freqs = np.asarray(freqs)
    power = np.asarray(power)
    valid = freqs > 0
    return float(freqs[valid][np.argmax(power[valid])]) if np.any(valid) else 0.0


def analyze(result: SimulationResult, model_type: str) -> dict:
    """Run full analysis and return summary dict."""
    rates = mean_firing_rate_result(result)
    cv = cv_isi(result.spikes, result.dt_ms)
    out = {
        "mean_rate_overall_hz": rates["overall"],
        "mean_rate_e_hz": rates["e"],
        "mean_rate_i_hz": rates["i"],
        "cv_isi": cv,
    }

    if model_type == "ModelBConfig":
        rate_e, rate_i = population_rate_by_pop(
            result.spikes, result.n_e, result.dt_ms, bin_size_ms=1.0
        )
        pl = phase_lag(rate_e, rate_i, result.dt_ms)
        out["ei_phase_lag_ms"] = pl["lag_ms"]
        out["ei_phase_lag_deg"] = pl["lag_deg"]
        out["dominant_freq_hz"] = pl["dominant_freq_hz"]
    else:
        rate = population_rate(result.spikes, result.dt_ms, bin_size_ms=1.0)
        out["dominant_freq_hz"] = dominant_frequency(rate, result.dt_ms)

    return out


def plot_results(
    result: SimulationResult,
    model_type: str,
    regime: str,
    outdir: Path,
    raster_dpi: int = 110,
    trace_format: str = "pdf",
):
    """Generate and save plots via btorch visualisation APIs."""
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)

    spikes = _to_spikes_2d(result.spikes).detach().cpu().numpy()
    volts = result.voltages[:, 0, :].detach().cpu().numpy()
    n_neuron = spikes.shape[1]
    n_e = result.n_e

    spikes_raster, keep_idx = _downsample_raster_spikes(
        spikes, n_e=n_e, max_neurons=800
    )
    rate_full = population_rate(result.spikes, result.dt_ms, bin_size_ms=1.0)

    raster_axes = plot_raster(
        spikes_raster,
        dt=result.dt_ms,
        rate=rate_full,
        rate_window_ms=1.0,
        marker_size=2.0,
        title=(
            f"{model_type} - {regime.upper()} - Raster + Rate "
            f"(sampled {len(keep_idx)}/{n_neuron} neurons)"
        ),
    )
    raster_fig = (
        raster_axes[0].figure if isinstance(raster_axes, tuple) else raster_axes.figure
    )
    raster_fig.savefig(outdir / "raster.png", dpi=raster_dpi, bbox_inches="tight")
    plt.close(raster_fig)

    sample_ids = [0, n_e, n_e // 2, n_e + n_e // 2]
    sample_ids = [idx for idx in sample_ids if 0 <= idx < n_neuron]
    sample_ids = list(dict.fromkeys(sample_ids))
    volt_ax = plot_traces(
        volts,
        dt=result.dt_ms,
        neurons=sample_ids,
        title=f"{model_type} - {regime.upper()} - Voltage Traces",
        ylabel="Voltage (mV)",
    )
    trace_format = trace_format.lower().lstrip(".")
    volt_ax.figure.savefig(
        outdir / f"voltage.{trace_format}",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(volt_ax.figure)
