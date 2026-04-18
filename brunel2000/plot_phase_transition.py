"""Plot Brunel phase-transition maps from sweep CSV outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from omegaconf import OmegaConf


@dataclass
class PlotConfig:
    input_dir: str = "outputs/brunel2000_phase_sweep"
    output_dir: str = "outputs/brunel2000_phase_sweep/plots"


def load_config() -> PlotConfig:
    defaults = OmegaConf.structured(PlotConfig())
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.unsafe_merge(defaults, cli_cfg)
    return OmegaConf.to_object(cfg)


def classify_regime(row: pd.Series) -> str:
    rate = float(row.get("mean_rate_overall_hz", 0.0))
    cv = float(row.get("cv_isi", np.nan))
    freq = float(row.get("dominant_freq_hz", 0.0))

    if rate < 0.5:
        return "silent"
    if np.isfinite(cv) and cv < 0.7 and freq < 15.0:
        return "SR"
    if freq >= 40.0:
        return "SI-fast"
    if freq >= 10.0:
        return "SI-slow"
    return "AI"


def load_sweep_df(input_dir: Path) -> pd.DataFrame:
    raw_dir = input_dir / "raw"
    files = sorted(raw_dir.glob("sweep_chunk_*.csv"))
    if not files:
        single = raw_dir / "sweep_results.csv"
        if single.exists():
            files = [single]
    if not files:
        raise FileNotFoundError(
            f"No sweep CSV files found in {raw_dir} (expected sweep_results.csv "
            "or sweep_chunk_*.csv)."
        )
    frames = [pd.read_csv(path) for path in files]
    df = pd.concat(frames, ignore_index=True)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    if df.empty:
        raise RuntimeError("All sweep rows failed; no data to plot.")
    return df


def plot_model_maps(df_model: pd.DataFrame, model: str, outdir: Path, label: str = ""):
    df_model = df_model.copy()
    df_model["regime"] = df_model.apply(classify_regime, axis=1)

    def _extent_from_pivot(piv: pd.DataFrame) -> list[float] | None:
        if piv.empty:
            return None
        x_vals = np.asarray(piv.columns, dtype=float)
        y_vals = np.asarray(piv.index, dtype=float)
        if not np.isfinite(x_vals).any() or not np.isfinite(y_vals).any():
            return None
        x_min = float(np.nanmin(x_vals))
        x_max = float(np.nanmax(x_vals))
        y_min = float(np.nanmin(y_vals))
        y_max = float(np.nanmax(y_vals))
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5
        if y_min == y_max:
            y_min -= 0.5
            y_max += 0.5
        return [x_min, x_max, y_min, y_max]

    metrics = [
        ("mean_rate_overall_hz", "Mean Rate (Hz)"),
        ("cv_isi", "CV(ISI)"),
        ("dominant_freq_hz", "Dominant Freq (Hz)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    ax_regime = axes[0, 0]

    regime_order = ["silent", "SR", "AI", "SI-slow", "SI-fast"]
    regime_to_code = {name: idx for idx, name in enumerate(regime_order)}
    cmap = ListedColormap(["#f0f0f0", "#4c78a8", "#54a24b", "#f58518", "#b279a2"])

    piv_reg = (
        df_model.assign(regime_code=df_model["regime"].map(regime_to_code))
        .pivot_table(index="eta", columns="g", values="regime_code", aggfunc="first")
        .sort_index()
    )
    reg_extent = _extent_from_pivot(piv_reg)
    if reg_extent is not None:
        im = ax_regime.imshow(
            piv_reg.values,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            extent=reg_extent,
            vmin=0,
            vmax=len(regime_order) - 1,
        )
        del im
    else:
        ax_regime.text(0.5, 0.5, "No data", ha="center", va="center")
    title_prefix = f"Model {model.upper()}"
    if label:
        title_prefix = f"{title_prefix} ({label})"
    ax_regime.set_title(f"{title_prefix} - Regime Map")
    ax_regime.set_xlabel("g")
    ax_regime.set_ylabel("eta")
    legend_handles = [
        plt.Line2D(
            [0], [0], marker="s", color="w", markerfacecolor=cmap(i), markersize=10
        )
        for i in range(len(regime_order))
    ]
    ax_regime.legend(legend_handles, regime_order, loc="upper right", framealpha=0.95)

    target_axes = [axes[0, 1], axes[1, 0], axes[1, 1]]
    for ax, (metric, title) in zip(target_axes, metrics):
        if metric not in df_model.columns:
            ax.text(0.5, 0.5, f"Missing metric: {metric}", ha="center", va="center")
            ax.set_title(title)
            ax.set_xlabel("g")
            ax.set_ylabel("eta")
            continue
        piv = df_model.pivot_table(
            index="eta", columns="g", values=metric, aggfunc="mean"
        ).sort_index()
        extent = _extent_from_pivot(piv)
        if extent is None:
            ax.text(0.5, 0.5, "No finite data", ha="center", va="center")
        else:
            mesh = ax.imshow(
                piv.values,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="viridis",
            )
            fig.colorbar(mesh, ax=ax, shrink=0.9)
        ax.set_title(title)
        ax.set_xlabel("g")
        ax.set_ylabel("eta")

    suffix = f"_{label}" if label else ""
    out_path = outdir / f"phase_transition_model_{model}{suffix}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    cfg = load_config()
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_sweep_df(input_dir)
    df.to_csv(output_dir / "sweep_summary_all.csv", index=False)

    for model in sorted(df["model"].dropna().unique()):
        df_model = df[df["model"] == model]
        if "j_i" in df_model.columns:
            ji_vals = sorted(v for v in df_model["j_i"].dropna().unique())
        else:
            ji_vals = []

        if model == "b" and len(ji_vals) > 1:
            for ji in ji_vals:
                tag = f"j_i_{ji:.3f}".replace(".", "p")
                plot_model_maps(
                    df_model[df_model["j_i"] == ji],
                    model=model,
                    outdir=output_dir,
                    label=tag,
                )
        else:
            plot_model_maps(df_model, model=model, outdir=output_dir)


if __name__ == "__main__":
    main()
