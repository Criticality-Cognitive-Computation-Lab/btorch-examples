"""Parameter sweep for Brunel (2000) phase transitions.

This script sweeps the (g, eta) plane for Model A and/or Model B, where
eta = nu_ext / nu_thr. It includes paper-oriented presets to reproduce the
key phase maps discussed around Fig. 2 (Model A) and Fig. 6 (Model B).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from btorch.models import environ

from brunel2000.analysis import analyze, plot_results
from brunel2000.config import BrunelConfig, ModelAConfig, ModelBConfig, SimConfig
from brunel2000.simulate import run_simulation


@dataclass
class SweepGridConfig:
    g_min: float = 3.0
    g_max: float = 7.0
    g_steps: int = 9
    eta_min: float = 0.5
    eta_max: float = 4.0
    eta_steps: int = 15


@dataclass
class SweepSimConfig:
    duration_ms: float = 1000.0
    dt_ms: float = 0.1
    seed: int = 42
    device: str = "cpu"
    n_neurons: int = 12500
    n_e_ratio: float = 0.8
    c_e: int = 1000
    c_ext: int = 1000
    n_trials: int = 4


@dataclass
class SweepExecConfig:
    output_dir: str = "outputs/brunel2000_phase_sweep"
    num_workers: int = 4
    save_trial_plots: bool = True
    raster_dpi: int = 90
    trace_format: str = "pdf"


@dataclass
class SweepPaperConfig:
    # Presets based on Brunel (2000):
    # - model_a_fig2: phase diagram in g vs nu_ext/nu_thr for model A.
    # - model_b_fig6: phase diagrams for model B with multiple J_I values.
    # - both_key: run both presets in one sweep output.
    profile: Literal["none", "model_a_fig2", "model_b_fig6", "both_key"] = "both_key"
    model_b_j_i_values: list[float] = field(default_factory=lambda: [0.05, 0.1, 0.2])


@dataclass
class SweepConfig:
    model: Literal["a", "b", "both"] = "both"
    grid: SweepGridConfig = field(default_factory=SweepGridConfig)
    sim: SweepSimConfig = field(default_factory=SweepSimConfig)
    exec: SweepExecConfig = field(default_factory=SweepExecConfig)
    paper: SweepPaperConfig = field(default_factory=SweepPaperConfig)


def load_config() -> SweepConfig:
    defaults = OmegaConf.structured(SweepConfig())
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.unsafe_merge(defaults, cli_cfg)
    return OmegaConf.to_object(cfg)


def linspace_values(vmin: float, vmax: float, steps: int) -> list[float]:
    if steps <= 1:
        return [float(vmin)]
    step = (vmax - vmin) / float(steps - 1)
    return [float(vmin + i * step) for i in range(steps)]


def apply_paper_profile(cfg: SweepConfig) -> SweepConfig:
    profile = cfg.paper.profile
    if profile == "none":
        return cfg

    # Axes in the paper are shown as g vs nu_ext / nu_thr.
    cfg.grid.eta_min = 0.0
    cfg.grid.eta_max = 4.0
    cfg.grid.eta_steps = 6

    # Keep one trial by default for broad phase-map reproduction.
    if cfg.sim.n_trials <= 0:
        cfg.sim.n_trials = 1

    if profile == "model_a_fig2":
        cfg.model = "a"
        cfg.grid.g_min = 1.0
        cfg.grid.g_max = 8.0
        cfg.grid.g_steps = 12
    elif profile == "model_b_fig6":
        cfg.model = "b"
        cfg.grid.g_min = 1.0
        cfg.grid.g_max = 8.0
        cfg.grid.g_steps = 12
    else:  # both_key
        cfg.model = "both"
        cfg.grid.g_min = 1.0
        cfg.grid.g_max = 8.0
        cfg.grid.g_steps = 12

    return cfg


def nu_thr_hz(theta: float, j: float, c_e: int, tau_ms: float) -> float:
    # Brunel 2000 threshold input rate (converted to Hz)
    return (theta / (j * c_e * tau_ms)) * 1000.0


def build_model_a_config(sim: SweepSimConfig, g: float, eta: float) -> ModelAConfig:
    base = ModelAConfig(
        n_neurons=sim.n_neurons,
        n_e_ratio=sim.n_e_ratio,
        c_e=sim.c_e,
        c_ext=sim.c_ext,
    )
    nuth = nu_thr_hz(theta=base.theta, j=base.j, c_e=base.c_e, tau_ms=base.tau_ms)
    return ModelAConfig(
        n_neurons=base.n_neurons,
        n_e_ratio=base.n_e_ratio,
        c_e=base.c_e,
        c_ext=base.c_ext,
        j=base.j,
        g=g,
        d_ms=base.d_ms,
        tau_ms=base.tau_ms,
        tau_ref_ms=base.tau_ref_ms,
        theta=base.theta,
        v_reset=base.v_reset,
        tau_syn_ms=base.tau_syn_ms,
        nu_ext_hz=eta * nuth,
    )


def build_model_b_config(
    sim: SweepSimConfig,
    g: float,
    eta: float,
    j_i_override: float | None = None,
) -> ModelBConfig:
    base = ModelBConfig(
        n_neurons=sim.n_neurons,
        n_e_ratio=sim.n_e_ratio,
        c_e=sim.c_e,
        c_ext=sim.c_ext,
    )
    nuth = nu_thr_hz(theta=base.theta, j=base.j_e, c_e=base.c_e, tau_ms=base.tau_e_ms)
    nu_ext = eta * nuth
    j_i = base.j_i if j_i_override is None else float(j_i_override)
    return ModelBConfig(
        n_neurons=base.n_neurons,
        n_e_ratio=base.n_e_ratio,
        c_e=base.c_e,
        c_ext=base.c_ext,
        j_e=base.j_e,
        j_i=j_i,
        g_e=g,
        g_i=g,
        tau_e_ms=base.tau_e_ms,
        tau_i_ms=base.tau_i_ms,
        tau_ref_ms=base.tau_ref_ms,
        theta=base.theta,
        v_reset=base.v_reset,
        d_ee_ms=base.d_ee_ms,
        d_ei_ms=base.d_ei_ms,
        d_ie_ms=base.d_ie_ms,
        d_ii_ms=base.d_ii_ms,
        tau_syn_e_ms=base.tau_syn_e_ms,
        tau_syn_i_ms=base.tau_syn_i_ms,
        nu_e_ext_hz=nu_ext,
        nu_i_ext_hz=nu_ext,
    )


def iter_points(cfg: SweepConfig):
    gs = linspace_values(cfg.grid.g_min, cfg.grid.g_max, cfg.grid.g_steps)
    etas = linspace_values(cfg.grid.eta_min, cfg.grid.eta_max, cfg.grid.eta_steps)
    models = [cfg.model] if cfg.model in ("a", "b") else ["a", "b"]
    ji_values = [None]
    if cfg.paper.profile in ("model_b_fig6", "both_key"):
        ji_values = [float(v) for v in cfg.paper.model_b_j_i_values]
    for model_kind in models:
        for eta in etas:
            for g in gs:
                if model_kind == "b":
                    for ji in ji_values:
                        for trial_idx in range(max(1, int(cfg.sim.n_trials))):
                            yield model_kind, g, eta, trial_idx, ji
                else:
                    for trial_idx in range(max(1, int(cfg.sim.n_trials))):
                        yield model_kind, g, eta, trial_idx, None


def run_one(
    model_kind: str,
    g: float,
    eta: float,
    trial_idx: int,
    j_i_override: float | None,
    point_idx: int,
    cfg: SweepConfig,
) -> dict:
    sim_cfg = cfg.sim
    if model_kind == "a":
        model_cfg = build_model_a_config(sim_cfg, g=g, eta=eta)
        model_name = "ModelAConfig"
    else:
        model_cfg = build_model_b_config(
            sim_cfg,
            g=g,
            eta=eta,
            j_i_override=j_i_override,
        )
        model_name = "ModelBConfig"

    device = sim_cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    point_seed = int(sim_cfg.seed + point_idx)

    brunel_cfg = BrunelConfig(
        model=model_cfg,
        sim=SimConfig(
            duration_ms=sim_cfg.duration_ms,
            dt_ms=sim_cfg.dt_ms,
            seed=point_seed,
            regime="ai",
            device=device,
        ),
    )

    with environ.context(dt=brunel_cfg.sim.dt_ms):
        model = brunel_cfg.model.build_model(
            dt_ms=brunel_cfg.sim.dt_ms,
            device=brunel_cfg.sim.device,
        )

    result = run_simulation(model, brunel_cfg, seed=point_seed)
    stats = analyze(result, model_name)

    if cfg.exec.save_trial_plots:
        g_tag = f"{g:.3f}".replace(".", "p")
        eta_tag = f"{eta:.3f}".replace(".", "p")
        if model_kind == "b":
            ji_val = model_cfg.j_i if hasattr(model_cfg, "j_i") else float("nan")
            ji_tag = f"{ji_val:.3f}".replace(".", "p")
            trial_dir = (
                Path(cfg.exec.output_dir)
                / "trial_plots"
                / f"model_{model_kind}"
                / f"j_i_{ji_tag}"
                / f"eta_{eta_tag}"
                / f"g_{g_tag}"
                / f"trial_{trial_idx:03d}"
            )
        else:
            trial_dir = (
                Path(cfg.exec.output_dir)
                / "trial_plots"
                / f"model_{model_kind}"
                / f"eta_{eta_tag}"
                / f"g_{g_tag}"
                / f"trial_{trial_idx:03d}"
            )
        plot_results(
            result,
            model_name,
            "ai",
            trial_dir,
            raster_dpi=int(cfg.exec.raster_dpi),
            trace_format=cfg.exec.trace_format,
        )

    row = {
        "model": model_kind,
        "model_type": model_name,
        "g": float(g),
        "eta": float(eta),
        "trial": int(trial_idx),
        "point_idx": int(point_idx),
        "seed": int(point_seed),
        "j_i": float(model_cfg.j_i) if hasattr(model_cfg, "j_i") else np.nan,
        "paper_profile": cfg.paper.profile,
        **stats,
    }
    return row


def run_one_task(
    task: tuple[int, str, float, float, int, float | None],
    cfg: SweepConfig,
) -> dict:
    point_idx, model_kind, g, eta, trial_idx, j_i_override = task
    return run_one(
        model_kind=model_kind,
        g=g,
        eta=eta,
        trial_idx=trial_idx,
        j_i_override=j_i_override,
        point_idx=point_idx,
        cfg=cfg,
    )


def main():
    cfg = apply_paper_profile(load_config())

    out_root = Path(cfg.exec.output_dir)
    raw_dir = out_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(idx, *task) for idx, task in enumerate(iter_points(cfg))]
    total_points = len(tasks)
    max_workers = max(1, int(cfg.exec.num_workers))
    max_workers = min(max_workers, total_points) if total_points > 0 else 1
    print(
        f"Sweep setup: total_points={total_points}, trials={cfg.sim.n_trials}, "
        f"num_workers={max_workers}, cpu_count={os.cpu_count()}"
    )

    rows: list[dict] = []
    if max_workers == 1:
        for point_idx, model_kind, g, eta, trial_idx, j_i_override in tasks:
            print(
                f"Running idx={point_idx} model={model_kind} g={g:.3f} eta={eta:.3f} "
                f"trial={trial_idx} j_i={j_i_override}"
            )
            try:
                row = run_one_task(
                    (point_idx, model_kind, g, eta, trial_idx, j_i_override),
                    cfg,
                )
                row["status"] = "ok"
            except Exception as exc:
                row = {
                    "model": model_kind,
                    "g": float(g),
                    "eta": float(eta),
                    "trial": int(trial_idx),
                    "seed": int(cfg.sim.seed + point_idx),
                    "point_idx": int(point_idx),
                    "j_i": float(j_i_override) if j_i_override is not None else np.nan,
                    "paper_profile": cfg.paper.profile,
                    "status": "error",
                    "error": repr(exc),
                }
            rows.append(row)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(run_one_task, task, cfg): task for task in tasks
            }
            for future in as_completed(future_to_task):
                point_idx, model_kind, g, eta, trial_idx, j_i_override = future_to_task[
                    future
                ]
                try:
                    row = future.result()
                    row["status"] = "ok"
                except Exception as exc:
                    row = {
                        "model": model_kind,
                        "g": float(g),
                        "eta": float(eta),
                        "trial": int(trial_idx),
                        "seed": int(cfg.sim.seed + point_idx),
                        "point_idx": int(point_idx),
                        "j_i": float(j_i_override)
                        if j_i_override is not None
                        else np.nan,
                        "paper_profile": cfg.paper.profile,
                        "status": "error",
                        "error": repr(exc),
                    }
                rows.append(row)

    df = pd.DataFrame(rows).sort_values(by=["model", "j_i", "eta", "g", "trial"])
    out_path = raw_dir / "sweep_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
