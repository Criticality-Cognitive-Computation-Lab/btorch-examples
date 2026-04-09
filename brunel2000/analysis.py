"""Analysis and plotting for Brunel (2000) simulation results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy import signal

from brunel2000.simulate import SimulationResult


def population_rate(
    spikes: torch.Tensor, dt_ms: float, bin_size_ms: float = 1.0
) -> np.ndarray:
    """Compute population firing rate in Hz."""
    T, batch, n_neuron = spikes.shape
    bin_steps = max(int(bin_size_ms / dt_ms), 1)
    n_bins = T // bin_steps
    spikes_binned = spikes[: n_bins * bin_steps].reshape(
        n_bins, bin_steps, batch, n_neuron
    )
    rate = (
        spikes_binned.sum(dim=(1, 3)) / (bin_size_ms / 1000.0) / n_neuron
    )  # (n_bins, batch)
    return rate.squeeze().cpu().numpy()


def population_rate_by_pop(
    spikes: torch.Tensor, n_e: int, dt_ms: float, bin_size_ms: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Compute separate E and I population rates in Hz."""
    T, batch, n_neuron = spikes.shape
    bin_steps = max(int(bin_size_ms / dt_ms), 1)
    n_bins = T // bin_steps
    spikes_binned = spikes[: n_bins * bin_steps].reshape(
        n_bins, bin_steps, batch, n_neuron
    )

    rate_e = spikes_binned[:, :, :, :n_e].sum(dim=(1, 3)) / (bin_size_ms / 1000.0) / n_e
    rate_i = (
        spikes_binned[:, :, :, n_e:].sum(dim=(1, 3))
        / (bin_size_ms / 1000.0)
        / (n_neuron - n_e)
    )
    return rate_e.squeeze().cpu().numpy(), rate_i.squeeze().cpu().numpy()


def mean_firing_rate(spikes: torch.Tensor, dt_ms: float) -> dict[str, float]:
    """Mean firing rate per neuron in Hz."""
    T, batch, n_neuron = spikes.shape
    duration_s = T * dt_ms / 1000.0
    rates = spikes.sum(dim=0).mean(dim=0) / duration_s  # (n_neuron,)
    rates = rates.cpu().numpy()
    n_e = n_neuron // 2 if n_neuron % 2 == 0 else int(n_neuron * 0.8)
    return {
        "overall": float(rates.mean()),
        "e": float(rates[:n_e].mean()),
        "i": float(rates[n_e:].mean()),
    }


def mean_firing_rate_result(result: SimulationResult) -> dict[str, float]:
    """Mean firing rate from SimulationResult."""
    T, batch, n_neuron = result.spikes.shape
    duration_s = T * result.dt_ms / 1000.0
    rates = result.spikes.sum(dim=0).mean(dim=0) / duration_s
    rates = rates.cpu().numpy()
    n_e = result.n_e
    return {
        "overall": float(rates.mean()),
        "e": float(rates[:n_e].mean()),
        "i": float(rates[n_e:].mean()),
    }


def cv_isi(spikes: torch.Tensor, dt_ms: float) -> float:
    """Coefficient of variation of inter-spike intervals."""
    T, batch, n_neuron = spikes.shape
    assert batch == 1
    spikes = spikes[:, 0, :].cpu().numpy()
    cvs = []
    for n in range(n_neuron):
        times = np.where(spikes[:, n] > 0)[0] * dt_ms
        if len(times) < 2:
            continue
        isi = np.diff(times)
        if isi.mean() > 0:
            cvs.append(isi.std() / isi.mean())
    return float(np.mean(cvs)) if cvs else 0.0


def phase_lag(rate_e: np.ndarray, rate_i: np.ndarray, dt_ms: float) -> dict[str, float]:
    """Compute E/I phase lag from cross-correlation."""
    if len(rate_e) < 10 or len(rate_i) < 10:
        return {"lag_ms": 0.0, "lag_deg": 0.0, "dominant_freq_hz": 0.0}

    # Detrend
    rate_e = signal.detrend(rate_e)
    rate_i = signal.detrend(rate_i)

    # Cross-correlation
    xcorr = signal.correlate(rate_e, rate_i, mode="full")
    lags = signal.correlation_lags(len(rate_e), len(rate_i), mode="full")
    peak_idx = np.argmax(np.abs(xcorr))
    lag_bins = lags[peak_idx]
    lag_ms = lag_bins * dt_ms

    # Dominant frequency via FFT of average rate
    avg_rate = (rate_e + rate_i) / 2.0
    freqs = np.fft.rfftfreq(len(avg_rate), d=dt_ms / 1000.0)
    spectrum = np.abs(np.fft.rfft(avg_rate))
    dom_freq = freqs[np.argmax(spectrum[1:]) + 1] if len(spectrum) > 1 else 0.0

    # Phase lag in degrees
    if dom_freq > 0:
        lag_deg = (lag_ms / 1000.0) * dom_freq * 360.0
        lag_deg = ((lag_deg + 180.0) % 360.0) - 180.0
    else:
        lag_deg = 0.0

    return {
        "lag_ms": float(lag_ms),
        "lag_deg": float(lag_deg),
        "dominant_freq_hz": float(dom_freq),
    }


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
        out["dominant_freq_hz"] = 0.0

    return out


def plot_results(result: SimulationResult, model_type: str, regime: str, outdir: Path):
    """Generate and save plots."""
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    spikes = result.spikes[:, 0, :].cpu().numpy()
    n_neuron = spikes.shape[1]
    n_e = result.n_e
    T = spikes.shape[0]
    time = np.arange(T) * result.dt_ms

    # Raster plot (sample)
    fig, ax = plt.subplots(figsize=(10, 4))
    sample_size = min(200, n_neuron)
    indices = np.linspace(0, n_neuron - 1, sample_size, dtype=int)
    for idx, nid in enumerate(indices):
        times = time[spikes[:, nid] > 0]
        color = "tab:blue" if nid < n_e else "tab:red"
        ax.scatter(times, np.full_like(times, idx), c=color, s=1, alpha=0.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron (sample)")
    ax.set_title(f"{model_type} – {regime.upper()} – Raster")
    fig.savefig(outdir / "raster.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Population rate
    fig, ax = plt.subplots(figsize=(10, 3))
    if model_type == "ModelBConfig":
        rate_e, rate_i = population_rate_by_pop(
            result.spikes, n_e, result.dt_ms, bin_size_ms=1.0
        )
        t_rate = np.arange(len(rate_e)) * 1.0
        ax.plot(t_rate, rate_e, label="E rate", color="tab:blue")
        ax.plot(t_rate, rate_i, label="I rate", color="tab:red")
        ax.legend()
    else:
        rate = population_rate(result.spikes, result.dt_ms, bin_size_ms=1.0)
        t_rate = np.arange(len(rate)) * 1.0
        ax.plot(t_rate, rate, color="tab:green")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Rate (Hz)")
    ax.set_title(f"{model_type} – {regime.upper()} – Population Rate")
    fig.savefig(outdir / "rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Voltage traces (sample)
    volts = result.voltages[:, 0, :].cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 3))
    for nid in [0, n_e, n_e // 2, n_e + n_e // 2]:
        if nid < n_neuron:
            ax.plot(time, volts[:, nid], alpha=0.7, label=f"N{nid}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (mV)")
    ax.set_title(f"{model_type} – {regime.upper()} – Voltage Traces")
    ax.legend()
    fig.savefig(outdir / "voltage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
