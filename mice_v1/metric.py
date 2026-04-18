from typing import Any

import numpy as np
import pandas as pd
import torch

from btorch.analysis.spiking import fano, isi_cv, kurtosis


def ensure_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _agg_axes(x):
    return tuple(range(x.ndim - 1))


def _mean_abs(x, axis):
    return np.nanmean(np.abs(x), axis=axis)


def _pct(x, q):
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if x.size else np.nan


def _softplus(z):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)


def _soft_hinge(z, s):
    return s * _softplus(z / s)


def _softplus_torch(z: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.exp(-torch.abs(z))) + torch.clamp(z, min=0.0)


def _soft_hinge_torch(z: torch.Tensor, s: float) -> torch.Tensor:
    if s <= 0:
        return torch.clamp(z, min=0.0)
    s_t = torch.as_tensor(s, device=z.device, dtype=z.dtype)
    return s_t * _softplus_torch(z / s_t)


def _log_ratio(a, b, eps):
    return np.log10(np.maximum(a, eps)) - np.log10(np.maximum(b, eps))


def _hinge_log_bounds(x, lo, hi, eps):
    log_x = np.log10(np.maximum(x, eps))
    log_lo = np.log10(np.maximum(lo, eps))
    log_hi = np.log10(np.maximum(hi, eps))
    return np.maximum(0.0, log_lo - log_x) + np.maximum(0.0, log_x - log_hi)


def _safe_max_arg(x, ids):
    if x.size == 0:
        return np.nan, None
    idx = int(np.argmax(x))
    return float(x[idx]), int(ids[idx])


def _log_abs_summary(x, eps):
    x_log = np.log10(np.maximum(x, eps))
    return {
        "q05": _pct(x_log, 5),
        "q50": _pct(x_log, 50),
        "q95": _pct(x_log, 95),
    }


def compute_dynamics_metrics(spikes, dt=1.0):
    """Compute scalar metrics for all neurons."""
    spikes = ensure_numpy(spikes)
    #breakpoint()
    T, B, N = spikes.shape

    # Firing Rate (Hz)
    rate = spikes.mean(axis=(0, 1)) * 1000.0 / dt

    # CV (per-neuron; keep behavior aligned with previous mean-over-batch logic)
    cv, _ = isi_cv(spikes, dt_ms=dt)
    # Manual nan-safe mean to avoid warnings on all-NaN slices
    cv = np.where(np.isfinite(cv), cv, np.nan)
    finite_mask = np.isfinite(cv)
    counts = finite_mask.sum(axis=0)
    sum_cv = np.nansum(cv, axis=0)
    cv = np.divide(sum_cv, counts, out=np.zeros_like(sum_cv), where=counts > 0)

    # Fano & Kurtosis
    window = max(10, min(100 if T > 200 else T // 2, T - 1))

    fano_values, _ = fano(spikes, window=window)
    fano_mean = np.nanmean(fano_values, axis=0)

    kurt_values, _ = kurtosis(spikes, window=window)
    kurt_mean = np.nanmean(kurt_values, axis=0)
    return pd.DataFrame(
        {
            "rate_hz": rate,
            "cv": cv,
            "fano": fano_mean,
            "kurtosis": kurt_mean,
        }
    )


def skip_initial_timesteps(x, skip_ms, dt):
    """Utility to skip initial timesteps for metrics that are sensitive to
    startup transients."""
    skip_steps = int(skip_ms / dt)
    return x[skip_steps:, ...]


def compute_magnitude_mismatch_metrics(
    psc: torch.Tensor | np.ndarray,
    asc: torch.Tensor | np.ndarray,
    epsc: torch.Tensor | np.ndarray,
    ipsc: torch.Tensor | np.ndarray,
    *,
    e_index: np.ndarray | None = None,
    c_min: float | None = None,
    c_max: float | None = None,
    ratio_thresh: float = 40.0,
    eps_floor: float = 1e-12,
    percentiles: tuple[float, float] = (5.0, 95.0),
) -> tuple[dict[str, float], dict[str, Any]]:
    """Compute PSC/ASC and EPSC/IPSC mismatch metrics.

    Returns (metrics, info). Inputs are [T, B, N], while ASC may be [T,
    B, N, S] and is reduced over the state dim.
    """

    #breakpoint()

    psc = ensure_numpy(psc)
    asc = ensure_numpy(asc)
    epsc = ensure_numpy(epsc)
    ipsc = ensure_numpy(ipsc)

    psc_abs = _mean_abs(psc, axis=_agg_axes(psc))
    asc_abs = _mean_abs(asc, axis=tuple(range(asc.ndim - 2)))
    asc_abs = _mean_abs(asc_abs, axis=-1)
    epsc_abs = _mean_abs(epsc, axis=_agg_axes(epsc))
    ipsc_abs = _mean_abs(ipsc, axis=_agg_axes(ipsc))

    total_abs = psc_abs + asc_abs
    p_lo, p_hi = percentiles
    if c_min is None:
        c_min = _pct(total_abs, p_lo)
    if c_max is None:
        c_max = _pct(total_abs, p_hi)
    eps = max(float(c_min) * 1e-3, eps_floor)

    log_psc_asc = _log_ratio(psc_abs, asc_abs, eps)
    log_epsc_ipsc = _log_ratio(epsc_abs, ipsc_abs, eps)
    log_ratio_thresh = np.log10(max(ratio_thresh, 1.0 + 1e-12))
    l_balance_psc_asc = float(np.nanmean(np.abs(log_psc_asc)))
    l_balance_epsc_ipsc = float(np.nanmean(np.abs(log_epsc_ipsc)))
    l_magnitude = float(np.nanmean(_hinge_log_bounds(total_abs, c_min, c_max, eps)))

    mask_psc_asc = (psc_abs > c_min) & (asc_abs > c_min)
    mask_epsc_ipsc = (epsc_abs > c_min) & (ipsc_abs > c_min)
    psc_over_asc_masked = psc_abs[mask_psc_asc] / np.maximum(asc_abs[mask_psc_asc], eps)
    epsc_over_ipsc_masked = epsc_abs[mask_epsc_ipsc] / np.maximum(
        ipsc_abs[mask_epsc_ipsc], eps
    )
    log_psc_asc_masked = log_psc_asc[mask_psc_asc]
    log_epsc_ipsc_masked = log_epsc_ipsc[mask_epsc_ipsc]

    local_index = np.arange(psc_abs.shape[0], dtype=int)
    if e_index is None:
        e_index = local_index.copy()
    else:
        e_index = np.asarray(e_index, dtype=int)
        if e_index.shape[0] != psc_abs.shape[0]:
            raise ValueError(
                "e_index must have the same length as the neuron dimension of inputs."
            )

    dead_mask = total_abs < c_min
    exploding_mask = total_abs > c_max

    unbalanced_psc_asc_mask = np.zeros_like(mask_psc_asc, dtype=bool)
    if log_psc_asc_masked.size:
        unbalanced_psc_asc_mask[mask_psc_asc] = (
            np.abs(log_psc_asc_masked) > log_ratio_thresh
        )

    unbalanced_epsc_ipsc_mask = np.zeros_like(mask_epsc_ipsc, dtype=bool)
    if log_epsc_ipsc_masked.size:
        unbalanced_epsc_ipsc_mask[mask_epsc_ipsc] = (
            np.abs(log_epsc_ipsc_masked) > log_ratio_thresh
        )

    error_mask = (
        dead_mask | exploding_mask | unbalanced_psc_asc_mask | unbalanced_epsc_ipsc_mask
    )

    max_abs_log_psc_asc, max_abs_log_psc_asc_sid = _safe_max_arg(
        np.abs(log_psc_asc_masked), e_index[mask_psc_asc]
    )
    max_abs_log_epsc_ipsc, max_abs_log_epsc_ipsc_sid = _safe_max_arg(
        np.abs(log_epsc_ipsc_masked), e_index[mask_epsc_ipsc]
    )

    metrics = {
        "l_balance_psc_asc": l_balance_psc_asc,
        "l_balance_epsc_ipsc": l_balance_epsc_ipsc,
        "l_magnitude": l_magnitude,
    }

    info = {
        "thresholds": {
            "c_min": float(c_min),
            "c_max": float(c_max),
            "ratio_thresh": float(ratio_thresh),
            "log10_ratio_thresh": float(log_ratio_thresh),
        },
        "frac_dead": float(np.mean(total_abs < c_min)),
        "frac_exploding": float(np.mean(total_abs > c_max)),
        "frac_unbalanced_psc_asc": float(
            np.mean(np.abs(log_psc_asc_masked) > log_ratio_thresh)
        )
        if psc_over_asc_masked.size
        else np.nan,
        "frac_unbalanced_epsc_ipsc": float(
            np.mean(np.abs(log_epsc_ipsc_masked) > log_ratio_thresh)
        )
        if epsc_over_ipsc_masked.size
        else np.nan,
        "max_abs_log_psc_asc": max_abs_log_psc_asc,
        "max_abs_log_psc_asc_sid": max_abs_log_psc_asc_sid,
        "max_abs_log_epsc_ipsc": max_abs_log_epsc_ipsc,
        "max_abs_log_epsc_ipsc_sid": max_abs_log_epsc_ipsc_sid,
        "dead_local_indices": local_index[dead_mask].tolist(),
        "dead_sids": e_index[dead_mask].tolist(),
        "exploding_local_indices": local_index[exploding_mask].tolist(),
        "exploding_sids": e_index[exploding_mask].tolist(),
        "unbalanced_psc_asc_local_indices": local_index[
            unbalanced_psc_asc_mask
        ].tolist(),
        "unbalanced_psc_asc_sids": e_index[unbalanced_psc_asc_mask].tolist(),
        "unbalanced_epsc_ipsc_local_indices": local_index[
            unbalanced_epsc_ipsc_mask
        ].tolist(),
        "unbalanced_epsc_ipsc_sids": e_index[unbalanced_epsc_ipsc_mask].tolist(),
        "error_local_indices": local_index[error_mask].tolist(),
        "error_sids": e_index[error_mask].tolist(),
        "log_abs_summary": {
            "psc": _log_abs_summary(psc_abs, eps),
            "asc": _log_abs_summary(asc_abs, eps),
            "epsc": _log_abs_summary(epsc_abs, eps),
            "ipsc": _log_abs_summary(ipsc_abs, eps),
        },
    }

    return metrics, info


def compute_ei_balance(
    I_e: torch.Tensor | np.ndarray,
    I_i: torch.Tensor | np.ndarray,
    *,
    I_ext: torch.Tensor | np.ndarray | None = None,
    dt: float = 1.0,
    max_lag_ms: float = 30.0,
    debug: bool = False,
) -> tuple[dict[str, float], dict[str, Any]]:
    eps = 1e-8

    #breakpoint()

    if dt <= 0:
        raise ValueError("dt must be positive.")

    device = (
        I_e.device
        if isinstance(I_e, torch.Tensor)
        else I_i.device
        if isinstance(I_i, torch.Tensor)
        else I_ext.device
        if isinstance(I_ext, torch.Tensor)
        else torch.device("cpu")
    )

    def _to_3d_tensor(x: torch.Tensor | np.ndarray) -> torch.Tensor:
        x_t = x if isinstance(x, torch.Tensor) else torch.from_numpy(np.asarray(x))
        x_t = x_t.to(device=device, dtype=torch.float32)
        if x_t.ndim == 2:
            x_t = x_t.unsqueeze(1)
        if x_t.ndim != 3:
            raise ValueError("Inputs must have shape [T, N] or [T, B, N].")
        return x_t

    Ie = _to_3d_tensor(I_e)
    Ii = _to_3d_tensor(I_i)
    if Ie.shape != Ii.shape:
        raise ValueError("I_e and I_i must have the same shape.")

    Iext = None if I_ext is None else _to_3d_tensor(I_ext)
    if Iext is not None and Iext.shape != Ie.shape:
        raise ValueError("I_ext must have the same shape as I_e and I_i.")

    T, B, N = Ie.shape
    lag_bins = int(max_lag_ms / dt)

    with torch.no_grad():
        # 1. Exact balance (ECI)
        I_rec = Ie + Ii
        imbalance_signal = I_rec if Iext is None else (I_rec + Iext)
        numer = torch.abs(imbalance_signal.mean(dim=(0, 1)))
        denom = (torch.abs(Ie) + torch.abs(Ii)).mean(dim=(0, 1)) + eps
        eci_per_neuron = numer / denom

        # 2. Delay-tolerant tracking correlation
        peak_corr = torch.full((N,), -1.0, device=device, dtype=torch.float32)
        best_lag_bins = torch.zeros((N,), device=device, dtype=torch.int64)

        # Collect all lags and their correlations (only in debug mode)
        valid_lags: list[int] = []
        all_corrs: list[torch.Tensor] = []

        for lag in range(-lag_bins, lag_bins + 1):
            shift = abs(lag)
            if shift >= T:
                continue

            if debug:
                valid_lags.append(lag)

            if lag >= 0:
                x = Ie[: T - lag]
                y = -Ii[lag:]
            else:
                x = Ie[shift:]
                y = -Ii[: T - shift]

            x_mean = x.mean(dim=(0, 1))
            y_mean = y.mean(dim=(0, 1))
            x_centered = x - x_mean
            y_centered = y - y_mean

            cov = (x_centered * y_centered).mean(dim=(0, 1))
            x_std = torch.sqrt(torch.clamp((x_centered**2).mean(dim=(0, 1)), min=eps))
            y_std = torch.sqrt(torch.clamp((y_centered**2).mean(dim=(0, 1)), min=eps))
            corr = cov / (x_std * y_std + eps)

            if debug:
                all_corrs.append(corr)

            improved = corr > peak_corr
            peak_corr = torch.where(improved, corr, peak_corr)
            best_lag_bins = torch.where(
                improved,
                torch.full_like(best_lag_bins, lag),
                best_lag_bins,
            )

        # Stack all correlations into [n_lags, N] array (only in debug mode)
        if debug:
            corr_over_lags = torch.stack(all_corrs, dim=0)  # [n_lags, N]
            lag_values = torch.tensor(
                valid_lags, device=device, dtype=torch.float32
            ) * float(dt)

        delay_ms_per_neuron = best_lag_bins.to(torch.float32) * float(dt)

        # 3. Aggregate metrics
        eci_mean = float(eci_per_neuron.mean())
        eci_median = float(torch.quantile(eci_per_neuron, 0.5))
        eci_p90 = float(torch.quantile(eci_per_neuron, 0.9))

        track_corr_peak_mean = float(peak_corr.mean())
        track_corr_peak_median = float(torch.quantile(peak_corr, 0.5))
        track_corr_peak_p90 = float(torch.quantile(peak_corr, 0.9))

        delay_ms_mean = float(delay_ms_per_neuron.mean())
        delay_ms_median = float(torch.quantile(delay_ms_per_neuron, 0.5))
        delay_ms_abs_mean = float(torch.abs(delay_ms_per_neuron).mean())

    metrics = {
        "eci_mean": eci_mean,
        "eci_median": eci_median,
        "eci_p90": eci_p90,
        "track_corr_peak_mean": track_corr_peak_mean,
        "track_corr_peak_median": track_corr_peak_median,
        "track_corr_peak_p90": track_corr_peak_p90,
        "delay_ms_mean": delay_ms_mean,
        "delay_ms_median": delay_ms_median,
        "delay_ms_abs_mean": delay_ms_abs_mean,
    }

    info: dict[str, Any] = {
        "eci_per_neuron": eci_per_neuron.cpu().numpy(),
        "track_corr_peak_per_neuron": peak_corr.cpu().numpy(),
        "delay_ms_per_neuron": delay_ms_per_neuron.cpu().numpy(),
        "best_lag_bins_per_neuron": best_lag_bins.cpu().numpy(),
        "lag_bins": float(lag_bins),
        "dt_ms": float(dt),
    }

    if debug:
        info["corr_over_lags"] = corr_over_lags.cpu().numpy()
        info["lag_values_ms"] = lag_values.cpu().numpy()

    return metrics, info


def compute_voltage_clamp(
    v: torch.Tensor | np.ndarray,
    *,
    v_min: float | np.ndarray | None = None,
    v_max: float | np.ndarray | None = None,
    softness: float = 1.0,
    percentiles: tuple[float, float] = (5.0, 95.0),
) -> tuple[dict[str, float], dict[str, np.ndarray | float | list[int]]]:
    # 1. Stay in PyTorch for the heavy lifting
    if isinstance(v, np.ndarray):
        v = torch.from_numpy(v)

    if v.ndim == 2:
        v = v.unsqueeze(1)  # [T, 1, N]

    device = v.device
    #breakpoint()
    T, B, N = v.shape

    # 2. Handle bounds using PyTorch's faster reductions
    if v_min is None or v_max is None:
        # We compute min/max across time and batch
        v_min_per = torch.amin(v, dim=(0, 1))
        v_max_per = torch.amax(v, dim=(0, 1))
        p_lo, p_hi = percentiles

        if v_min is None:
            v_min = torch.quantile(v_min_per, p_lo / 100.0).item()
        if v_max is None:
            v_max = torch.quantile(v_max_per, p_hi / 100.0).item()

    v_min_t = torch.as_tensor(v_min, device=device)
    v_max_t = torch.as_tensor(v_max, device=device)
    v_range = torch.clamp(v_max_t - v_min_t, min=1e-6)

    # 3. Memory-efficient violations with soft-hinge penalties
    with torch.no_grad():
        # Calculate undershoot components
        diff_low = (v_min_t - v).clamp_(min=0.0).div_(v_range)
        v_low_mask = diff_low > 0

        # softness controls smoothness/magnitude of the hinge transition.
        undershoot_pen_t = _soft_hinge_torch(diff_low, softness)

        # Immediate reduction to [N]
        v_pen_undershoot = undershoot_pen_t.mean(dim=(0, 1))
        v_undershoot_frac = v_low_mask.float().mean(dim=(0, 1))
        v_undershoot_any = v_low_mask.any(dim=(0, 1))

        # Free up memory before next big allocation
        del diff_low, v_low_mask, undershoot_pen_t

        # Repeat for overshoot
        diff_high = (v - v_max_t).clamp_(min=0.0).div_(v_range)
        v_high_mask = diff_high > 0
        overshoot_pen_t = _soft_hinge_torch(diff_high, softness)

        v_pen_overshoot = overshoot_pen_t.mean(dim=(0, 1))
        v_overshoot_frac = v_high_mask.float().mean(dim=(0, 1))
        v_overshoot_any = v_high_mask.any(dim=(0, 1))

        # v_out_frac needs the combined mask
        v_out_frac = (v_high_mask | (v < v_min_t)).float().mean(dim=(0, 1))

        v_pen = v_pen_undershoot + v_pen_overshoot
        v_any_violating = v_undershoot_any | v_overshoot_any

    # 4. Package outputs (Matching your original format exactly)
    metrics = {
        "v_penalty_mean": float(v_pen.mean()),
        "v_penalty_undershoot_mean": float(v_pen_undershoot.mean()),
        "v_penalty_overshoot_mean": float(v_pen_overshoot.mean()),
        "v_out_frac_mean": float(v_out_frac.mean()),
        "v_undershoot_frac_mean": float(v_undershoot_frac.mean()),
        "v_overshoot_frac_mean": float(v_overshoot_frac.mean()),
        "v_violating_neurons_frac": float(v_any_violating.float().mean()),
    }

    info = {
        "v_min_mean": float(v_min_t.mean()),
        "v_max_mean": float(v_max_t.mean()),
        "v_penalty_per_neuron": v_pen.cpu().numpy(),
        "v_penalty_undershoot_per_neuron": v_pen_undershoot.cpu().numpy(),
        "v_penalty_overshoot_per_neuron": v_pen_overshoot.cpu().numpy(),
        "v_out_frac_per_neuron": v_out_frac.cpu().numpy(),
        "v_undershoot_frac_per_neuron": v_undershoot_frac.cpu().numpy(),
        "v_overshoot_frac_per_neuron": v_overshoot_frac.cpu().numpy(),
        "v_undershoot_indices": torch.where(v_undershoot_any)[0].cpu().tolist(),
        "v_overshoot_indices": torch.where(v_overshoot_any)[0].cpu().tolist(),
    }

    return metrics, info


def compute_rate_clamp(
    spikes: torch.Tensor | np.ndarray,
    *,
    dt: float = 1.0,
    rate_min: float | None = None,
    rate_max: float | None = None,
    softness: float = 1.0,
    percentiles: tuple[float, float] = (1.0, 99.0),
) -> tuple[dict[str, float], dict[str, np.ndarray | float]]:
    """Soft-clamp penalties for firing rate only.

    Returns (metrics, info). Inputs are [T, B, N] or [T, N].
    """

    spikes = ensure_numpy(spikes)
    if spikes.ndim == 2:
        spikes = spikes[:, None, :]

    rate_hz = spikes.mean(axis=(0, 1)) * 1000.0 / dt
    rate_hz_mean = float(np.nanmean(rate_hz))

    p_lo, p_hi = percentiles
    if rate_min is None:
        rate_min = _pct(rate_hz, p_lo)
    if rate_max is None:
        rate_max = _pct(rate_hz, p_hi)

    r_pen_undershoot = _soft_hinge(rate_min - rate_hz_mean, softness)
    r_pen_overshoot = _soft_hinge(rate_hz_mean - rate_max, softness)
    r_pen = r_pen_undershoot + r_pen_overshoot

    rate_low_mask = rate_hz < rate_min
    rate_high_mask = rate_hz > rate_max

    metrics = {
        "rate_penalty_mean": float(r_pen),
        "rate_penalty_undershoot": float(r_pen_undershoot),
        "rate_penalty_overshoot": float(r_pen_overshoot),
        "rate_out_frac": float((rate_hz_mean < rate_min) or (rate_hz_mean > rate_max)),
        "rate_undershoot_frac": float(np.mean(rate_low_mask)),
        "rate_overshoot_frac": float(np.mean(rate_high_mask)),
    }

    info = {
        "rate_min": float(rate_min),
        "rate_max": float(rate_max),
        "rate_hz_per_neuron": rate_hz,
        "rate_undershoot_indices": np.where(rate_low_mask)[0].tolist(),
        "rate_overshoot_indices": np.where(rate_high_mask)[0].tolist(),
    }

    return metrics, info


def sample_neurons_by_group(
    neurons_df: pd.DataFrame,
    sample_size: int,
    *,
    group_col: str = "group",
    id_col: str = "simple_id",
    random_state: int = 42,
) -> np.ndarray:
    """Sample neuron indices approximately balanced across groups.

    Falls back to random sampling when no groups are available.
    Returns a numpy array of ids from `id_col`.
    """
    if id_col not in neurons_df.columns:
        raise ValueError(f"Column '{id_col}' not found in neurons_df.")
    if group_col not in neurons_df.columns:
        raise ValueError(
            f"Column '{group_col}' not found in neurons_df for subgrouping."
        )

    rng = np.random.default_rng(random_state)
    n_neurons_total = len(neurons_df)
    sample_size = int(min(sample_size, n_neurons_total))

    neurons_meta_full = neurons_df.set_index(id_col)
    sub_labels = neurons_meta_full[group_col].fillna("Unknown")
    unique_subs = list(dict.fromkeys(sub_labels.tolist()))

    if not unique_subs:
        return rng.choice(n_neurons_total, size=sample_size, replace=False)

    per_sub = sample_size // len(unique_subs)
    remainder = sample_size % len(unique_subs)
    selected: list[int] = []

    for i, sub in enumerate(unique_subs):
        sub_pool = neurons_meta_full[sub_labels == sub].index.to_numpy()
        target = per_sub + (1 if i < remainder else 0)
        take = min(target, len(sub_pool))
        if take > 0:
            chosen = rng.choice(sub_pool, size=take, replace=False)
            selected.extend(chosen.tolist())

    selected = list(dict.fromkeys(selected))
    if len(selected) < sample_size:
        remaining_pool = np.setdiff1d(
            neurons_meta_full.index.to_numpy(), np.array(selected)
        )
        need = sample_size - len(selected)
        if len(remaining_pool) > 0 and need > 0:
            extra = rng.choice(
                remaining_pool, size=min(need, len(remaining_pool)), replace=False
            )
            selected.extend(extra.tolist())

    sample_indices = np.array(selected, dtype=int)
    return sample_indices
