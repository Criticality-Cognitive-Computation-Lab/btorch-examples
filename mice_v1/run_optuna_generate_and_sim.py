from __future__ import annotations

import argparse
import gc
import json
import logging
import multiprocessing as mp
import os
import pickle
import re
import sys
import traceback
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch
import matplotlib.pyplot as plt
from hydra import compose, initialize
from omegaconf import OmegaConf
from optuna.trial import TrialState

from btorch.models import environ
from btorch.utils.conf import load_config
from btorch.visualisation.timeseries import plot_neuron_traces

from src.models.base import BaseRSNN
from src.runner import ExperimentRunner
from src.utils.device import detect_device
from src.utils.vis_utils import prepare_data_from_dict, visualize_results, select_neurons_by_class
from src.utils.other import set_seed
from src.utils.dataloader import create_dataloaders

import ipdb

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent.parent
NETWORK_PKG_PATH = PARENT_DIR / "network_generator"
if str(NETWORK_PKG_PATH) not in sys.path:
    sys.path.append(str(NETWORK_PKG_PATH))

#breakpoint()

from build_real_bluebrain_caller import build_brain_net
from network_generator.algorithms.allocator import HybridWeightAllocator

from metric import (
    compute_dynamics_metrics,
    compute_ei_balance,
    compute_magnitude_mismatch_metrics,
    compute_rate_clamp,
    compute_voltage_clamp,
)

from param_tune.optuna_metrics import (
    ObjectiveConfig,
    _make_low_activity_evaluation,
    _make_non_finite_evaluation,
    _objective_spec,
    evaluate_analysis_out,
)


LOGGER = logging.getLogger(__name__)


def _get_process_memory_gb() -> float:
    """Return current process RSS memory in GB (best effort)."""
    # Linux: ru_maxrss is in KiB.
    try:
        import resource

        rss_kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return rss_kb / (1024.0 * 1024.0)
    except Exception:
        return float("nan")


def _log_trial_memory(tag: str, trial_no: int) -> None:
    cpu_gb = _get_process_memory_gb()
    if torch.cuda.is_available():
        try:
            device = torch.cuda.current_device()
            alloc_gb = torch.cuda.memory_allocated(device) / (1024.0**3)
            reserved_gb = torch.cuda.memory_reserved(device) / (1024.0**3)
            max_alloc_gb = torch.cuda.max_memory_allocated(device) / (1024.0**3)
            max_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024.0**3)
            LOGGER.info(
                "trial %s [%s] mem | CPU(maxRSS)=%.3f GB | GPU alloc=%.3f GB reserved=%.3f GB peak_alloc=%.3f GB peak_reserved=%.3f GB",
                trial_no,
                tag,
                cpu_gb,
                alloc_gb,
                reserved_gb,
                max_alloc_gb,
                max_reserved_gb,
            )
        except Exception as exc:
            LOGGER.info(
                "trial %s [%s] mem | CPU(maxRSS)=%.3f GB | GPU mem unavailable (%s)",
                trial_no,
                tag,
                cpu_gb,
                exc,
            )
    else:
        LOGGER.info(
            "trial %s [%s] mem | CPU(maxRSS)=%.3f GB | GPU unavailable",
            trial_no,
            tag,
            cpu_gb,
        )


@dataclass
class FloatRange:
    low: float
    high: float
    log: bool = False


@dataclass
class SearchSpace:
    ee_weight_mean: FloatRange = field(default_factory=lambda: FloatRange(70, 110.0))
    ee_weight_sigma: FloatRange = field(default_factory=lambda: FloatRange(5, 20))
    ei_weight_mean: FloatRange = field(default_factory=lambda: FloatRange(110, 150.0))
    ei_weight_sigma: FloatRange = field(default_factory=lambda: FloatRange(5, 20))
    ii_weight_mean: FloatRange = field(default_factory=lambda: FloatRange(170, 230.0))
    ii_weight_sigma: FloatRange = field(default_factory=lambda: FloatRange(5, 20))
    target_ie_ratio: FloatRange = field(default_factory=lambda: FloatRange(0.1, 3.0))


@dataclass
class OptunaConfig:
    sampler: str = "bayesian" #bayesian
    n_trials: int = 16384
    n_jobs: int = 1
    timeout_s: int | None = None
    seed: int = 8
    study_name: str = "rsnn-generate-sim-optuna"
    storage: str | None = None
    load_if_exists: bool = True
    show_progress_bar: bool = True
    gc_after_trial: bool = True
    # Run each trial in a dedicated subprocess to avoid memory accumulation.
    isolate_trial_process: bool = True
    # Grid search controls (used when sampler == "grid")
    grid_n_points: int = 4
    # If True, grid uses logspace for all positive ranges; otherwise uses each
    # parameter's FloatRange.log flag.
    grid_force_logspace: bool = True


@dataclass
class ExperimentConfig:
    name: str = "exp_default"
    save_trial_json: bool = True
    rerun_best_for_plots: bool = True
    rerun_all_non_nan_for_plots: bool = True


@dataclass
class FastPlotConfig:
    enabled: bool = True
    plot_during_trial: bool = True
    figure_dpi: int = 140
    max_failed_neurons: int = 24
    failed_trace_figure_name: str = "failed_metrics_traces.png"
    max_good_neurons: int = 24
    good_trace_figure_name: str = "good_metrics_traces.png"
    random_trace_count: int = 40
    random_trace_seed: int = 123

    # Detailed visualization (same pipeline as run_generate_and_sim / runner.run_visualize).
    # This is expensive, so keep disabled unless a metric condition is met.
    detailed_enabled: bool = True
    detailed_use_or: bool = True
    detailed_if_track_corr_peak_mean_gt: float | None = 0.8
    detailed_if_eci_mean_lt: float | None = 0.5
    detailed_if_total_firing_rate_hz_min: float | None = 2.0
    detailed_if_total_firing_rate_hz_max: float | None = 15.0


@dataclass
class OutputConfig:
    root_dir: str = "outputs/optuna_0328_input_e_only-bayesian"


@dataclass
class NetworkReuseConfig:
    enabled: bool = True
    network_dir: str = ""


@dataclass
class MetricsConfig:
    # Start metric computation after this many milliseconds.
    start_after_ms: float = 150.0
    # Scope for EI metrics (eci/track_corr_peak_mean):
    # - "e_only": only excitatory neurons
    # - "all": all neurons
    ei_scope: str = "e_only"


@dataclass
class TuneConfig:
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    search_space: SearchSpace = field(default_factory=SearchSpace)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    fast_plot: FastPlotConfig = field(default_factory=FastPlotConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    reuse: NetworkReuseConfig = field(default_factory=NetworkReuseConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    # Hydra input group override forwarded to composed simulation config.
    # Allowed examples: "all", "e_only", "partial".
    input: str = "e_only"


def _parse_cli_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description=(
            "Optuna search + optional rerun for selected trial IDs with plots. "
            "Default mode runs study; rerun mode re-simulates saved trials."
        )
    )
    p.add_argument(
        "--mode",
        choices=["study", "rerun"],
        default="study",
        help="Execution mode: study (default) or rerun selected trials.",
    )
    p.add_argument(
        "--trial-id",
        type=int,
        action="append",
        default=None,
        help="Trial id to rerun (repeatable), e.g., --trial-id 12 --trial-id 37.",
    )
    p.add_argument(
        "--trial-id-file",
        type=Path,
        default=None,
        help="Optional JSON/TXT file containing trial ids to rerun.",
    )
    p.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Override study name when loading from Optuna storage in rerun mode.",
    )
    p.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Override Optuna storage URL/path in rerun mode.",
    )
    p.add_argument(
        "--rerun-out-subdir",
        type=str,
        default="selected_trials",
        help="Subdirectory under output.reruns for rerun mode outputs.",
    )
    args, unknown = p.parse_known_args(argv)
    return args, unknown


def _load_trial_ids_from_file(path: Path) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(f"trial id file not found: {path}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if isinstance(data.get("matched_trial_ids"), list):
                values = data["matched_trial_ids"]
            elif isinstance(data.get("trial_ids"), list):
                values = data["trial_ids"]
            else:
                raise ValueError(
                    "JSON must contain list field 'matched_trial_ids' or 'trial_ids'."
                )
        elif isinstance(data, list):
            values = data
        else:
            raise ValueError("Unsupported JSON format for trial ids.")
        return [int(v) for v in values]

    text = path.read_text(encoding="utf-8")
    ids = [int(x) for x in re.findall(r"\d+", text)]
    if not ids:
        raise ValueError(f"No integer trial id found in {path}")
    return ids


def _collect_requested_trial_ids(args: argparse.Namespace) -> list[int]:
    ids: list[int] = []
    if args.trial_id:
        ids.extend(int(x) for x in args.trial_id)
    if args.trial_id_file is not None:
        ids.extend(_load_trial_ids_from_file(Path(args.trial_id_file)))
    ids = sorted(set(ids))
    return ids


def _resolve_storage(cfg: TuneConfig, root: Path) -> str:
    if cfg.optuna.storage:
        storage = str(cfg.optuna.storage).strip()
        if "://" in storage:
            return storage
        storage_path = Path(storage).expanduser().resolve()
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{storage_path}"
    root.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(root / 'study.sqlite3').resolve()}"


def _create_dirs(cfg: TuneConfig) -> dict[str, Path]:
    root = Path(cfg.output.root_dir)
    trials = root / "trials"
    plots = root / "plots"
    reruns = root / "reruns"
    base_network = root / "base_network"
    for p in (root, trials, plots, reruns, base_network):
        p.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "trials": trials,
        "plots": plots,
        "reruns": reruns,
        "base_network": base_network,
    }


def _count_non_finite(*arrays: np.ndarray) -> tuple[int, float]:
    total = 0
    non_finite = 0
    for arr in arrays:
        x = np.asarray(arr)
        finite = np.isfinite(x)
        total += int(finite.size)
        non_finite += int(finite.size - finite.sum())
    ratio = float(non_finite) / float(total) if total else 0.0
    return int(non_finite), float(ratio)


def _sample_ids(ids: list[int], max_keep: int, seed: int) -> list[int]:
    if len(ids) <= max_keep:
        return ids
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.asarray(ids, dtype=int), size=max_keep, replace=False)
    return sorted(chosen.astype(int).tolist())


def _format_root_ids(ids: list[int], per_line: int = 5) -> str:
    vals = [str(int(x)) for x in ids]
    if not vals:
        return ""
    lines = [", ".join(vals[i : i + per_line]) for i in range(0, len(vals), per_line)]
    return "\n".join(lines)


def _should_plot_detailed_visualization(metrics: dict[str, Any], cfg: TuneConfig) -> tuple[bool, dict[str, bool]]:
    if not cfg.fast_plot.detailed_enabled:
        return False, {}

    cond: dict[str, bool] = {}

    thr_corr = cfg.fast_plot.detailed_if_track_corr_peak_mean_gt
    if thr_corr is not None:
        v = metrics.get("track_corr_peak_mean", float("nan"))
        cond["track_corr_peak_mean_gt"] = bool(np.isfinite(v) and float(v) > float(thr_corr))

    thr_eci = cfg.fast_plot.detailed_if_eci_mean_lt
    if thr_eci is not None:
        v = metrics.get("eci_mean", float("nan"))
        cond["eci_mean_lt"] = bool(np.isfinite(v) and float(v) < float(thr_eci))

    rate_min = cfg.fast_plot.detailed_if_total_firing_rate_hz_min
    rate_max = cfg.fast_plot.detailed_if_total_firing_rate_hz_max
    if rate_min is not None or rate_max is not None:
        v = metrics.get("total_firing_rate_hz", float("nan"))
        ok = bool(np.isfinite(v))
        if ok and rate_min is not None:
            ok = ok and (float(v) >= float(rate_min))
        if ok and rate_max is not None:
            ok = ok and (float(v) <= float(rate_max))
        cond["total_firing_rate_hz_in_range"] = ok

    if not cond:
        return False, {}

    should_plot = any(cond.values()) if cfg.fast_plot.detailed_use_or else all(cond.values())
    return bool(should_plot), cond


def _plot_detailed_visualization(
    *,
    states: dict[str, Any],
    model: BaseRSNN,
    cfg_trial: Any,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_indices, _ = select_neurons_by_class(model.connectome_data, num_per_class=4)
    data = prepare_data_from_dict(states, model.params, cfg_trial, model.connectome_data)
    try:
        visualize_results(**data, selected_neurons=selected_indices, epoch_figure_dir=out_dir)
    finally:
        # Explicitly release potentially large intermediate dicts/figures.
        del data
        plt.close("all")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _save_trial_payload(trial_dir: Path, payload: dict[str, Any]) -> None:
    (trial_dir / "trial_payload.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _trial_plot_dir(trial_dir: Path) -> Path:
    plot_dir = trial_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_neurons_connections(network_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    neurons_path = network_dir / "neurons.csv.gz"
    connections_path = network_dir / "connections.csv.gz"
    if neurons_path.exists() and connections_path.exists():
        neurons = pd.read_csv(neurons_path)
        connections = pd.read_csv(connections_path)
        return neurons, connections

    bundle_path = network_dir / "network_bundle.pkl"
    if bundle_path.exists():
        with open(bundle_path, "rb") as handle:
            bundle = pickle.load(handle)
        net = bundle.get("net")
        if net is None:
            raise FileNotFoundError(
                f"network_bundle.pkl in {network_dir} missing net data"
            )
        neurons = net.neurons.copy()
        adj = net.adj_matrix.tocoo()
        connections = pd.DataFrame(
            {
                "pre_simple_id": adj.row.astype(np.int64),
                "post_simple_id": adj.col.astype(np.int64),
            }
        )
        connections["syn_count"] = 1
        return neurons, connections

    raise FileNotFoundError(
        "Missing neurons.csv.gz or connections.csv.gz in "
        f"{network_dir}. Generate the base network first."
    )


def _apply_weight_distributions(
    connections: pd.DataFrame,
    neurons: pd.DataFrame,
    params: dict[str, float],
) -> pd.DataFrame:
    """Re-assign weights on a fixed topology using network_generator strategy.

    This mirrors `network_generator` logic but runs on an existing `connections`
    DataFrame (same edges). Strategy: `ee_ei_ii_inverse_pool`.

    Assumptions:
    - `connections` has columns: pre_simple_id, post_simple_id, EI, syn_count
    - `neurons` has columns: simple_id, EI
    """
    from btorch.connectome.connection import make_hetersynapse_conn
    from network_generator.network_generator.algorithms.weight_gen import (
        NeuronInDegreeWeightGenParam,
        compute_deg,
        generate_hetersyn_ee,
        generate_hetersyn_weights,
    )

    if "syn_count" not in connections.columns:
        raise ValueError("connections must contain syn_count for weight assignment")

    # ensure neurons has EI
    if "EI" not in neurons.columns:
        if "cell_class" in neurons.columns:
            e_mask = neurons["cell_class"].astype(str).str.contains(
                "pyr|et|it", case=False, na=False
            )
            neurons = neurons.copy()
            neurons["EI"] = np.where(e_mask, "E", "I")
        else:
            raise ValueError("neurons must contain EI or cell_class")

    n_neurons = int(neurons.shape[0])

    # Build sparse conn mats for each receptor pair from the fixed topology.
    # make_hetersynapse_conn expects neuron-mode EI on neurons, and uses
    # pre/post ids from connections.
    conn_mats, receptor_df = make_hetersynapse_conn(
        neurons=neurons,
        connections=connections,
        receptor_type_col="EI",
        receptor_type_mode="neuron",
        return_dict=True,
    )

    rng_seed = int(params.get("seed", 0))
    rng = np.random.default_rng(rng_seed)

    # Map params -> rule fields
    rule = NeuronInDegreeWeightGenParam(
        ie_ratio=float(params["target_ie_ratio"]),
        ie_std=float(params.get("ie_weight_sigma", params["ee_weight_sigma"])),
        ei_mean=float(params.get("ei_weight_mean", params["ee_weight_mean"])),
        ei_std=float(params.get("ei_weight_sigma", params["ee_weight_sigma"])),
        ii_ei_ratio=float(params.get("ii_ei_ratio", 1.0)),
        ii_mean=float(params.get("ii_weight_mean", params["ee_weight_mean"])),
        ii_std=float(params.get("ii_weight_sigma", params["ee_weight_sigma"])),
        strategy="ee_ei_ii_inverse_pool",
    )

    #breakpoint()

    # Generate EE pools (per-post)
    in_deg_ee = compute_deg(conn_mats[("E", "E")])
    weight_ee = generate_hetersyn_ee(
        in_deg_ee,
        mean=float(params["ee_weight_mean"]),
        std=float(params["ee_weight_sigma"]),
        rng=rng,
    )

    conn_mats = generate_hetersyn_weights(conn_mats, rule=rule, weight_ee=weight_ee, rng=rng)

    # Now map weights back onto the existing edge list.
    # Build lookup arrays from sparse matrices to edge weights.
    # Each conn_mat is (pre, post) with weights in data. We create a dict mapping
    # (pre, post) -> weight by iterating non-zeros once per channel.
    weights_lookup: dict[tuple[int, int], float] = {}
    for key, mat in conn_mats.items():
        coo = mat.tocoo()
        for pre, post, w in zip(coo.row, coo.col, coo.data, strict=False):
            weights_lookup[(int(pre), int(post))] = float(w)

    out = connections.copy()
    # Fill using lookup; preserve ordering
    out["weight"] = [
        weights_lookup.get((int(p), int(q)), 0.0)
        for p, q in zip(out["pre_simple_id"].to_numpy(), out["post_simple_id"].to_numpy())
    ]
    out["weight"] = out["weight"].astype(np.float32)
    return out


def _write_weighted_network(
    base_dir: Path,
    trial_dir: Path,
    params: dict[str, float],
    cfg_trial: Any,
) -> Path:
    trial_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = base_dir / "network_bundle.pkl"
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Missing network_bundle.pkl in {base_dir}. Generate base network first."
        )

    with open(bundle_path, "rb") as handle:
        bundle = pickle.load(handle)

    net = deepcopy(bundle["net"])
    registry = bundle["registry"]
    ei_registry = bundle["ei_registry"]

    # breakpoint()

    allocator = HybridWeightAllocator(
        param_registry=registry,
        ei_registry=ei_registry,
        ie_ratio=params["target_ie_ratio"],
        weight_rules={"strategy": "ee_ei_ii_inverse_pool"},
        weight_distributions={
            "ee_weight_mean": params["ee_weight_mean"],
            "ee_weight_sigma": params["ee_weight_sigma"],
            "ei_weight_mean": params.get("ei_weight_mean"),
            "ei_weight_sigma": params.get("ei_weight_sigma"),
            "ii_weight_mean": params.get("ii_weight_mean"),
            "ii_weight_sigma": params.get("ii_weight_sigma"),
            #"ii_ei_ratio": params["target_ie_ratio"],
        },
        seed=int(cfg_trial.seed),
    )

    allocator.assign(net)

    save_dir = _ensure_dir(trial_dir)
    neurons = net.neurons.copy()
    #breakpoint()
    if "EI" not in neurons.columns:
        if "type" in neurons.columns:
            e_mask = neurons["type"].astype(str).str.contains("pyr|et|it", case=False, na=False)
            neurons["EI"] = np.where(e_mask, "E", "I")
        elif "cell_class" in neurons.columns:
            e_mask = neurons["cell_class"].astype(str).str.contains("pyr|et|it", case=False, na=False)
            neurons["EI"] = np.where(e_mask, "E", "I")
    neurons.to_csv(save_dir / "neurons.csv.gz", index=False, compression="gzip")

    adj = net.adj_matrix.tocoo()
    w = net.weight_matrix.tocoo()
    edges_df = pd.DataFrame(
        {
            "pre_simple_id": adj.row.astype(np.int64),
            "post_simple_id": adj.col.astype(np.int64),
        }
    )
    weights_df = pd.DataFrame(
        {
            "pre_simple_id": w.row.astype(np.int64),
            "post_simple_id": w.col.astype(np.int64),
            "weight": w.data.astype(np.float32),
        }
    )
    edges_df = edges_df.merge(
        weights_df,
        on=["pre_simple_id", "post_simple_id"],
        how="left",
        validate="one_to_one",
    )
    edges_df["weight"] = edges_df["weight"].fillna(0.0)
    if "type" in neurons.columns:
        type_by_id = neurons["type"].astype(str).to_numpy()
        edges_df["pre_type"] = type_by_id[edges_df["pre_simple_id"].to_numpy()]
        edges_df["post_type"] = type_by_id[edges_df["post_simple_id"].to_numpy()]
        edges_df["EI"] = np.where(
            edges_df["pre_type"].str.contains("pyr|et|it", case=False),
            "E",
            "I",
        )
    edges_df["syn_count"] = 1

    edges_df.to_csv(save_dir / "connections.csv.gz", index=False, compression="gzip")

    return save_dir


def _ensure_base_network(
    cfg_trial: Any,
    dirs: dict[str, Path],
) -> tuple[Path, dict[str, Any]]:
    reuse_cfg = OmegaConf.select(cfg_trial, "reuse", default=None)
    reuse_enabled = False
    reuse_dir = ""
    if reuse_cfg is not None:
        reuse_enabled = bool(getattr(reuse_cfg, "enabled", False))
        reuse_dir = str(getattr(reuse_cfg, "network_dir", ""))

    if reuse_enabled and reuse_dir:
        base_dir = Path(reuse_dir).expanduser()
        if not base_dir.exists():
            raise FileNotFoundError(f"Base network dir not found: {base_dir}")
        return base_dir, {}

    base_dir = dirs["base_network"]
    marker = base_dir / "network_bundle.pkl"
    if marker.exists():
        return base_dir, {}

    generated = build_brain_net(cfg_trial.generator)
    generated_path = Path(generated["save_path"])
    bundle_path = generated_path / "network_bundle.pkl"
    if not bundle_path.exists():
        bundle_path = base_dir / "network_bundle.pkl"
        base_dir.mkdir(parents=True, exist_ok=True)
        with open(bundle_path, "wb") as handle:
            pickle.dump(
                {
                    "net": generated["net"],
                    "registry": generated["registry"],
                    "ei_registry": generated["ei_registry"],
                    "dist_rule": generated["dist_rule"],
                    "save_path": generated["save_path"],
                    "n_neurons": generated["n_neurons"],
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    else:
        base_dir.mkdir(parents=True, exist_ok=True)
        (base_dir / "network_bundle.pkl").write_bytes(bundle_path.read_bytes())

    return base_dir, generated


def _prepare_trace_arrays(
    v: np.ndarray,
    asc: np.ndarray,
    psc: np.ndarray,
    epsc: np.ndarray,
    ipsc: np.ndarray,
    spikes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def _ensure_time_major(x: np.ndarray) -> np.ndarray:
        if x.ndim >= 2 and x.shape[0] < x.shape[-1]:
            return x
        return np.transpose(x, (1, 0) + tuple(range(2, x.ndim)))

    v_t = _ensure_time_major(v)
    asc_t = _ensure_time_major(asc)
    psc_t = _ensure_time_major(psc)
    epsc_t = _ensure_time_major(epsc)
    ipsc_t = _ensure_time_major(ipsc)
    spikes_t = _ensure_time_major(spikes)
    return v_t, asc_t, psc_t, epsc_t, ipsc_t, spikes_t


def plot_metrics_selected_traces(
    *,
    states: dict[str, Any],
    dt: float,
    out_dir: Path,
    simple_ids: list[int],
    root_ids: list[int] | None,
    failure_tags_by_simple_id: dict[int, str] | None,
    figure_name: str,
    max_neurons: int,
) -> None:
    if not simple_ids:
        return

    selected = np.asarray(simple_ids, dtype=int)
    selected = np.unique(selected)
    if selected.size > max_neurons:
        selected = selected[:max_neurons]

    neuron = states.get("neuron", {})
    synapse = states.get("synapse", {})

    def _to_cpu_np(x: Any) -> np.ndarray:
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    v = _to_cpu_np(neuron["v"])[..., selected]
    spikes = _to_cpu_np(neuron["spike"])[..., selected]
    psc = _to_cpu_np(synapse["psc"])[..., selected]
    psc_e = _to_cpu_np(synapse.get("psc_e"))[..., selected]
    psc_i = _to_cpu_np(synapse.get("psc_i"))[..., selected]

    # Iasc shape is typically (T, B, N, n_asc_components).
    # Select neurons on the neuron axis (-2), not on the last ASC-component axis.
    iasc = _to_cpu_np(neuron.get("Iasc"))
    asc = iasc[..., selected, :]

    v_plot, asc_plot, psc_plot, epsc_plot, ipsc_plot, spike_plot = _prepare_trace_arrays(
        v, asc, psc, psc_e, psc_i, spikes
    )
    # v_plot: (1, 1000, 24)
    # asc_plot: (1, 1000, 24, 1)
    # psc_plot: (1, 1000, 24)
    # epsc_plot: (1, 1000, 24)
    # ipsc_plot: (1, 1000, 24)
    # spike_plot: (1, 1000, 24)

    if v_plot.ndim > 2:
        v_plot = v_plot[0, :, :]
        asc_plot = asc_plot[0, :, :, 0]
        psc_plot = psc_plot[0, :, :]
        epsc_plot = epsc_plot[0, :, :]
        ipsc_plot = ipsc_plot[0, :, :]
        spike_plot = spike_plot[0, :, :]

    root_id_map: dict[int, int] = {}
    if root_ids is not None and len(root_ids) == len(simple_ids):
        root_id_map = {int(s): int(r) for s, r in zip(simple_ids, root_ids)}

    fig = plot_neuron_traces(
        voltage=v_plot,
        asc=asc_plot,
        psc=psc_plot,
        epsc=epsc_plot,
        ipsc=ipsc_plot,
        spikes=spike_plot,
        dt=dt,
        neuron_indices=list(range(len(selected))),
        neuron_labels=lambda i: (
            f"{root_id_map.get(int(selected[int(i)]), int(selected[int(i)]))}"
            f" [{failure_tags_by_simple_id[int(selected[int(i)])]}]"
            if (
                failure_tags_by_simple_id is not None
                and int(selected[int(i)]) in failure_tags_by_simple_id
            )
            else str(root_id_map.get(int(selected[int(i)]), int(selected[int(i)])))
        ),
        neurons_per_row=2,
        neuron_label_position="top",
        show_voltage=True,
        show_asc=True,
        show_psc=True,
    )

    out_path = out_dir / figure_name
    out_dir.mkdir(parents=True, exist_ok=True)
    #breakpoint()
    if isinstance(fig, dict):
        for subfig in fig.values():
            subfig.savefig(out_path, dpi=140)
            plt.close(subfig)
    else:
        fig.savefig(out_path, dpi=140)
        plt.close(fig)

    # Release plotting/intermediate CPU memory promptly.
    del v, spikes, psc, psc_e, psc_i, asc, v_plot, asc_plot, psc_plot, epsc_plot, ipsc_plot, spike_plot
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _choose_priority_trial(study: optuna.study.Study):
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not complete_trials:
        return None
    return min(complete_trials, key=lambda t: tuple(t.values or []))


def _is_non_nan_complete_trial(trial: optuna.trial.FrozenTrial) -> bool:
    if trial.state != TrialState.COMPLETE:
        return False
    non_finite_count = trial.user_attrs.get("non_finite_count")
    if non_finite_count is not None:
        return int(non_finite_count) == 0
    values = trial.values
    if not values:
        return False
    return float(values[0]) <= 0.0


def rerun_trial_and_store(
    *,
    trial: optuna.trial.FrozenTrial,
    cfg: TuneConfig,
    out_dir: Path,
) -> dict[str, Any]:
    return rerun_params_and_store(
        trial_number=int(trial.number),
        params={k: float(v) for k, v in trial.params.items()},
        cfg=cfg,
        out_dir=out_dir,
        trial_values=list(trial.values or []),
        trial_params_raw=dict(trial.params),
    )


def rerun_params_and_store(
    *,
    trial_number: int,
    params: dict[str, float],
    cfg: TuneConfig,
    out_dir: Path,
    trial_values: list[float] | None = None,
    trial_params_raw: dict[str, Any] | None = None,
    rerun_overrides: list[str] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    trial_id = f"trial_{trial_number:05d}"

    cfg_trial = _build_cfg_from_trial(
        params,
        trial_id,
        out_dir,
        input_override=cfg.input,
        extra_overrides=rerun_overrides,
    )
    #breakpoint()
    if cfg_trial.device == "auto":
        cfg_trial.device, _ = detect_device()

    set_seed(cfg_trial.seed)
    environ.set(dt=cfg_trial.simulation.dt)

    # Reuse the same topology generated in study mode: outputs/.../base_network
    # (do not create a per-rerun topology under out_dir).
    rerun_dirs = _create_dirs(cfg)
    base_dir, generated_data = _ensure_base_network(
        cfg_trial,
        {
            "base_network": rerun_dirs["base_network"],
        },
    )
    params_with_seed = dict(params)
    params_with_seed["seed"] = cfg.optuna.seed + trial_number
    trial_network_dir = out_dir / "network"
    weighted_dir = _write_weighted_network(
        base_dir=base_dir,
        trial_dir=trial_network_dir,
        params=params_with_seed,
        cfg_trial=cfg_trial,
    )

    neuron_num = generated_data.get("n_neurons") if generated_data else None
    if neuron_num is None:
        neurons_df, _ = _load_neurons_connections(weighted_dir)
        neuron_num = int(neurons_df.shape[0])

    OmegaConf.set_struct(cfg_trial, False)
    cfg_trial.network.conn_path = str(weighted_dir)
    cfg_trial.network.n_neuron = neuron_num
    cfg_trial.simulation.n_neuron = neuron_num
    OmegaConf.set_struct(cfg_trial, True)

    model = BaseRSNN(cfg_trial)
    model.to(cfg_trial.device)

    i_thr = model.neuron_params.get("I_thr", None)
    dataloaders = create_dataloaders(cfg_trial, i_thr)
    runner = ExperimentRunner(cfg_trial, model, dataloaders)

    _, states = runner.run_simulation_return_states()
    if states is None:
        raise RuntimeError("Simulation did not return states.")

    neurons_df = model.connectome_data[0]
    e_mask = (neurons_df["EI"].to_numpy() == "E") if "EI" in neurons_df.columns else None
    analysis_out = _prepare_analysis_out(
        states,
        cfg,
        cfg_trial.simulation.dt,
        e_mask=e_mask,
    )
    evaluation = evaluate_analysis_out(analysis_out=analysis_out, cfg=cfg.objective)

    plot_metrics_selected_traces(
        states=states,
        dt=cfg_trial.simulation.dt,
        out_dir=out_dir,
        simple_ids=evaluation.failed_metrics_simple_ids,
        root_ids=evaluation.failed_metrics_root_ids,
        failure_tags_by_simple_id=evaluation.failed_tags_by_simple_id,
        figure_name=cfg.fast_plot.failed_trace_figure_name,
        max_neurons=cfg.fast_plot.max_failed_neurons,
    )
    plot_metrics_selected_traces(
        states=states,
        dt=cfg_trial.simulation.dt,
        out_dir=out_dir,
        simple_ids=evaluation.good_metrics_simple_ids,
        root_ids=evaluation.good_metrics_root_ids,
        failure_tags_by_simple_id=None,
        figure_name=cfg.fast_plot.good_trace_figure_name,
        max_neurons=cfg.fast_plot.max_good_neurons,
    )

    # Rerun mode: always produce full detailed visualization
    # (PSC/voltage/raster pipeline from run_generate_and_sim).
    detailed_dir = out_dir / "detailed_visualization"
    _plot_detailed_visualization(
        states=states,
        model=model,
        cfg_trial=cfg_trial,
        out_dir=detailed_dir,
    )

    summary = {
        "trial_number": trial_number,
        "trial_values": list(trial_values or []),
        "trial_params": trial_params_raw if trial_params_raw is not None else params,
        "rerun_overrides": list(rerun_overrides or []),
        "rerun_objective_values": list(evaluation.objective_values),
        "rerun_metrics": evaluation.metrics,
        "failed_metrics_root_ids": _sample_ids(
            evaluation.failed_metrics_root_ids,
            max_keep=200,
            seed=cfg.optuna.seed + trial_number,
        ),
        "failed_metrics_root_ids_total": len(evaluation.failed_metrics_root_ids),
        "good_metrics_root_ids": evaluation.good_metrics_root_ids,
    }
    summary["failed_metrics_root_ids_text"] = _format_root_ids(
        summary["failed_metrics_root_ids"]
    )

    (out_dir / "rerun_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def rerun_all_non_nan_trials_and_plot(
    study: optuna.study.Study,
    cfg: TuneConfig,
    dirs: dict[str, Path],
) -> list[dict[str, Any]]:
    summaries = []
    for trial in study.trials:
        if not _is_non_nan_complete_trial(trial):
            continue
        out_dir = dirs["reruns"] / f"trial_{trial.number:05d}"
        summaries.append(rerun_trial_and_store(trial=trial, cfg=cfg, out_dir=out_dir))
    return summaries


def rerun_best_and_plot(
    study: optuna.study.Study,
    cfg: TuneConfig,
    dirs: dict[str, Path],
) -> dict[str, Any] | None:
    trial = _choose_priority_trial(study)
    if trial is None:
        return None
    out_dir = dirs["reruns"] / "best"
    return rerun_trial_and_store(trial=trial, cfg=cfg, out_dir=out_dir)


def _suggest_params(trial: optuna.trial.Trial, cfg: TuneConfig) -> dict[str, float]:
    return {
        "ee_weight_mean": float(
            trial.suggest_float(
                "ee_weight_mean",
                cfg.search_space.ee_weight_mean.low,
                cfg.search_space.ee_weight_mean.high,
                log=cfg.search_space.ee_weight_mean.log,
            )
        ),
        "ee_weight_sigma": float(
            trial.suggest_float(
                "ee_weight_sigma",
                cfg.search_space.ee_weight_sigma.low,
                cfg.search_space.ee_weight_sigma.high,
                log=cfg.search_space.ee_weight_sigma.log,
            )
        ),
        "ei_weight_mean": float(
            trial.suggest_float(
                "ei_weight_mean",
                cfg.search_space.ei_weight_mean.low,
                cfg.search_space.ei_weight_mean.high,
                log=cfg.search_space.ei_weight_mean.log,
            )
        ),
        "ei_weight_sigma": float(
            trial.suggest_float(
                "ei_weight_sigma",
                cfg.search_space.ei_weight_sigma.low,
                cfg.search_space.ei_weight_sigma.high,
                log=cfg.search_space.ei_weight_sigma.log,
            )
        ),
        "ii_weight_mean": float(
            trial.suggest_float(
                "ii_weight_mean",
                cfg.search_space.ii_weight_mean.low,
                cfg.search_space.ii_weight_mean.high,
                log=cfg.search_space.ii_weight_mean.log,
            )
        ),
        "ii_weight_sigma": float(
            trial.suggest_float(
                "ii_weight_sigma",
                cfg.search_space.ii_weight_sigma.low,
                cfg.search_space.ii_weight_sigma.high,
                log=cfg.search_space.ii_weight_sigma.log,
            )
        ),
        "target_ie_ratio": float(
            trial.suggest_float(
                "target_ie_ratio",
                cfg.search_space.target_ie_ratio.low,
                cfg.search_space.target_ie_ratio.high,
                log=cfg.search_space.target_ie_ratio.log,
            )
        ),
    }


def _build_cfg_from_trial(
    params: dict[str, float],
    trial_id: str,
    out_dir: Path,
    *,
    input_override: str,
    extra_overrides: list[str] | None = None,
):
    with initialize(version_base=None, config_path="../conf", job_name="optuna_generate_and_sim"):
        overrides = [
            f"generator.weight_distributions.ee_weight_mean={params['ee_weight_mean']}",
            f"generator.weight_distributions.ee_weight_sigma={params['ee_weight_sigma']}",
            f"generator.weight_distributions.ei_weight_mean={params.get('ei_weight_mean', params['ee_weight_mean'])}",
            f"generator.weight_distributions.ei_weight_sigma={params.get('ei_weight_sigma', params['ee_weight_sigma'])}",
            f"generator.weight_distributions.ii_weight_mean={params.get('ii_weight_mean', params['ee_weight_mean'])}",
            f"generator.weight_distributions.ii_weight_sigma={params.get('ii_weight_sigma', params['ee_weight_sigma'])}",
            f"generator.target_ie_ratio={params['target_ie_ratio']}",
            f"input={input_override}",
            f"+generator.trial_id={trial_id}",
            f"generator.save_dir={out_dir.resolve()}",
            f"out_dir={out_dir.resolve()}",
            "bridge.use_generated_data=true",
        ]
        if extra_overrides:
            overrides.extend(extra_overrides)
        cfg = compose(config_name="config", overrides=overrides)
    return cfg


def _prepare_analysis_out(
    states: dict[str, Any],
    cfg: TuneConfig,
    dt: float,
    *,
    e_mask: np.ndarray | None = None,
):
    neuron = states.get("neuron", {})
    synapse = states.get("synapse", {})

    def _to_numpy(x: Any) -> np.ndarray | None:
        if x is None:
            return None
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    spikes = _to_numpy(neuron.get("spike"))
    v = _to_numpy(neuron.get("v"))
    # Keep ASC components as-is (T, B, N, n_asc_components),
    # metric.py handles component reduction internally.
    asc = _to_numpy(neuron.get("Iasc"))
    psc = _to_numpy(synapse.get("psc"))
    psc_e = _to_numpy(synapse.get("psc_e"))
    psc_i = _to_numpy(synapse.get("psc_i"))
    i_ext = _to_numpy(synapse.get("I_ext"))

    # Keep only time after start_after_ms for all metrics.
    skip_t = int(max(0, round(float(cfg.metrics.start_after_ms) / float(dt))))

    def _slice_time(x: np.ndarray | None):
        if x is None:
            return None
        if x.shape[0] <= skip_t:
            return x[0:0]
        return x[skip_t:]

    spikes = _slice_time(spikes)
    v = _slice_time(v)
    asc = _slice_time(asc)
    psc = _slice_time(psc)
    psc_e = _slice_time(psc_e)
    psc_i = _slice_time(psc_i)
    i_ext = _slice_time(i_ext)

    if spikes is None or v is None or psc is None or psc_e is None or psc_i is None:
        raise ValueError("Missing required synapse/neuron states for analysis.")

    #asc = np.zeros((*psc.shape, 1), dtype=psc.dtype)

    non_finite_count, non_finite_ratio = _count_non_finite(
        np.asarray(spikes, dtype=float),
        np.asarray(v, dtype=float),
        np.asarray(asc, dtype=float),
        np.asarray(psc, dtype=float),
        np.asarray(psc_e, dtype=float),
        np.asarray(psc_i, dtype=float),
    )
    analysis_out: dict[str, Any] = {
        "integrity": {
            "non_finite_count": int(non_finite_count),
            "non_finite_ratio": float(non_finite_ratio),
        },
        "metrics_time_window": {
            "start_after_ms": float(cfg.metrics.start_after_ms),
            "start_timestep": int(skip_t),
            "dt": float(dt),
        },
        "ei_scope": str(cfg.metrics.ei_scope),
    }

    if non_finite_count > 0:
        return analysis_out

    if spikes.shape[0] == 0:
        raise ValueError(
            f"No timesteps left after metrics start_after_ms={cfg.metrics.start_after_ms}ms (dt={dt})."
        )

    # EI metrics can be computed on E-only neurons (default) or all neurons.
    if cfg.metrics.ei_scope not in {"e_only", "all"}:
        raise ValueError(f"Unsupported cfg.metrics.ei_scope={cfg.metrics.ei_scope}")

    if cfg.metrics.ei_scope == "e_only":
        if e_mask is None:
            raise ValueError("e_mask is required when cfg.metrics.ei_scope='e_only'.")
        e_mask_arr = np.asarray(e_mask, dtype=bool)
        if e_mask_arr.shape[0] != spikes.shape[-1]:
            raise ValueError(
                f"e_mask length {e_mask_arr.shape[0]} != neuron dim {spikes.shape[-1]}"
            )
        if not np.any(e_mask_arr):
            raise ValueError("No excitatory neuron selected by e_mask for EI metrics.")

        psc_e_ei = psc_e[..., e_mask_arr]
        psc_i_ei = psc_i[..., e_mask_arr]
        i_ext_ei = i_ext[..., e_mask_arr] if i_ext is not None else None
    else:
        psc_e_ei = psc_e
        psc_i_ei = psc_i
        i_ext_ei = i_ext

    ei_metrics, ei_info = compute_ei_balance(
        I_e=psc_e_ei,
        I_i=psc_i_ei,
        I_ext=i_ext_ei,
        dt=dt,
        max_lag_ms=cfg.objective.ei.max_lag_ms,
    )
    dynamics_metrics = compute_dynamics_metrics(spikes, dt=dt)
    mismatch_metrics, mismatch_info = compute_magnitude_mismatch_metrics(
        psc,
        asc,
        psc_e,
        psc_i,
        ratio_thresh=cfg.objective.stability.mismatch_ratio_thresh,
        c_min=cfg.objective.stability.mismatch_c_min,
        c_max=cfg.objective.stability.mismatch_c_max,
        eps_floor=cfg.objective.stability.mismatch_eps_floor,
    )
    rate_metrics, rate_info = compute_rate_clamp(
        spikes,
        dt=dt,
        rate_min=cfg.objective.stability.rate_min,
        rate_max=cfg.objective.stability.rate_max,
        softness=cfg.objective.stability.rate_softness,
    )
    voltage_metrics, voltage_info = compute_voltage_clamp(
        v,
        v_min=cfg.objective.stability.v_min,
        v_max=cfg.objective.stability.v_max,
        softness=cfg.objective.stability.v_softness,
    )

    dynamics_metrics_global = {
        "rate_hz": {"mean": float(np.nanmean(dynamics_metrics["rate_hz"]))}
        if not dynamics_metrics.empty
        else {"mean": float("nan")}
    }

    analysis_out.update(
        {
            "ei_balance_metrics": ei_metrics,
            "ei_balance_info": ei_info,
            "dynamics_metrics": dynamics_metrics,
            "dynamics_metrics_global": dynamics_metrics_global,
            "mismatch_metrics": mismatch_metrics,
            "mismatch_info": mismatch_info,
            "rate_metrics": rate_metrics,
            "rate_info": rate_info,
            "voltage_metrics": voltage_metrics,
            "voltage_info": voltage_info,
            "neuron_ids": list(range(spikes.shape[-1])),
            "neuron_root_ids": list(range(spikes.shape[-1])),
            "ei_metrics_neuron_count": int(psc_e_ei.shape[-1]),
        }
    )
    return analysis_out


def _build_grid_search_space(cfg: TuneConfig) -> dict[str, list[float]]:
    def _grid_values(r: FloatRange) -> list[float]:
        n = max(2, int(cfg.optuna.grid_n_points))
        low = float(r.low)
        high = float(r.high)
        use_log = bool(cfg.optuna.grid_force_logspace or r.log)
        if use_log:
            if low <= 0 or high <= 0:
                raise ValueError(
                    f"log-scale grid requires positive range, got low={low}, high={high}"
                )
            vals = np.logspace(np.log10(low), np.log10(high), num=n)
        else:
            vals = np.linspace(low, high, num=n)
        return [float(v) for v in vals]

    return {
        "ee_weight_mean": _grid_values(cfg.search_space.ee_weight_mean),
        "ee_weight_sigma": _grid_values(cfg.search_space.ee_weight_sigma),
        "ei_weight_mean": _grid_values(cfg.search_space.ei_weight_mean),
        "ei_weight_sigma": _grid_values(cfg.search_space.ei_weight_sigma),
        "ii_weight_mean": _grid_values(cfg.search_space.ii_weight_mean),
        "ii_weight_sigma": _grid_values(cfg.search_space.ii_weight_sigma),
        "target_ie_ratio": _grid_values(cfg.search_space.target_ie_ratio),
    }


def _select_sampler(cfg: TuneConfig):
    if cfg.optuna.sampler == "random":
        return optuna.samplers.RandomSampler(seed=cfg.optuna.seed)
    if cfg.optuna.sampler == "grid":
        return optuna.samplers.GridSampler(_build_grid_search_space(cfg))
    return optuna.samplers.TPESampler(
        seed=cfg.optuna.seed,
        multivariate=True,
        group=True,
        constant_liar=True,
    )


def _run_trial_core(
    *,
    cfg: TuneConfig,
    dirs: dict[str, Path],
    trial_number: int,
    params: dict[str, float],
    trial_overrides: list[str] | None = None,
) -> dict[str, Any]:
    trial_id = f"trial_{trial_number:05d}"
    trial_dir = dirs["trials"] / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)

    cfg_trial = _build_cfg_from_trial(
        params,
        trial_id,
        trial_dir,
        input_override=cfg.input,
        extra_overrides=trial_overrides,
    )
    if cfg_trial.device == "auto":
        cfg_trial.device, _ = detect_device()

    set_seed(cfg_trial.seed)
    environ.set(dt=cfg_trial.simulation.dt)

    model = None
    runner = None
    dataloaders = None
    try:
        base_dir, generated_data = _ensure_base_network(cfg_trial, dirs)
        weighted_dir = _write_weighted_network(
            base_dir=base_dir,
            trial_dir=trial_dir,
            params=params,
            cfg_trial=cfg_trial,
        )

        OmegaConf.set_struct(cfg_trial, False)
        cfg_trial.network.conn_path = str(weighted_dir)
        if generated_data:
            neuron_num = generated_data["n_neurons"]
        else:
            neurons_df, _ = _load_neurons_connections(base_dir)
            neuron_num = len(neurons_df)
        cfg_trial.network.n_neuron = neuron_num
        cfg_trial.simulation.n_neuron = neuron_num
        OmegaConf.set_struct(cfg_trial, True)

        model = BaseRSNN(cfg_trial)
        model.to(cfg_trial.device)

        i_thr = model.neuron_params.get("I_thr", None)
        dataloaders = create_dataloaders(cfg_trial, i_thr)
        runner = ExperimentRunner(cfg_trial, model, dataloaders)

        _, states = runner.run_simulation_return_states()
        if states is None:
            raise RuntimeError("Simulation did not return states.")

        neurons_df = model.connectome_data[0]
        e_mask = (neurons_df["EI"].to_numpy() == "E") if "EI" in neurons_df.columns else None
        analysis_out = _prepare_analysis_out(states, cfg, cfg_trial.simulation.dt, e_mask=e_mask)
        integrity = analysis_out.get("integrity", {})
        non_finite_count = int(integrity.get("non_finite_count", 0))
        non_finite_ratio = float(integrity.get("non_finite_ratio", 0.0))

        if non_finite_count > 0 and cfg.objective.nan.early_stop_on_non_finite:
            evaluation = _make_non_finite_evaluation(
                cfg=cfg,
                non_finite_count=non_finite_count,
                non_finite_ratio=non_finite_ratio,
            )
        else:
            spike_np = states["neuron"]["spike"]
            if hasattr(spike_np, "detach"):
                spike_np = spike_np.detach().cpu().numpy()
            total_rate = float(np.nanmean(spike_np)) * (1000.0 / float(cfg_trial.simulation.dt))
            if (
                cfg.objective.nan.early_stop_on_low_activity
                and np.isfinite(total_rate)
                and total_rate < cfg.objective.nan.low_activity_rate_threshold_hz
            ):
                evaluation = _make_low_activity_evaluation(
                    cfg=cfg,
                    total_firing_rate_hz=total_rate,
                    non_finite_count=non_finite_count,
                    non_finite_ratio=non_finite_ratio,
                )
            else:
                evaluation = evaluate_analysis_out(analysis_out=analysis_out, cfg=cfg.objective)

        payload = {
            "trial": trial_number,
            "params": params,
            "objective_values": list(evaluation.objective_values),
            "metrics": evaluation.metrics,
            "non_finite_ratio": evaluation.non_finite_ratio,
            "failed_metrics_root_ids": evaluation.failed_metrics_root_ids,
            "good_metrics_root_ids": evaluation.good_metrics_root_ids,
            "failed_tags_by_simple_id": evaluation.failed_tags_by_simple_id,
        }
        payload["failed_metrics_root_ids_text"] = _format_root_ids(payload["failed_metrics_root_ids"])
        if cfg.experiment.save_trial_json:
            _save_trial_payload(trial_dir, payload)
        (trial_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

        if cfg.fast_plot.enabled and cfg.fast_plot.plot_during_trial and evaluation.non_finite_count == 0:
            plot_dir = _trial_plot_dir(trial_dir)
            plot_metrics_selected_traces(
                states=states,
                dt=cfg_trial.simulation.dt,
                out_dir=plot_dir,
                simple_ids=evaluation.failed_metrics_simple_ids,
                root_ids=evaluation.failed_metrics_root_ids,
                failure_tags_by_simple_id=evaluation.failed_tags_by_simple_id,
                figure_name=cfg.fast_plot.failed_trace_figure_name,
                max_neurons=cfg.fast_plot.max_failed_neurons,
            )
            plot_metrics_selected_traces(
                states=states,
                dt=cfg_trial.simulation.dt,
                out_dir=plot_dir,
                simple_ids=evaluation.good_metrics_simple_ids,
                root_ids=evaluation.good_metrics_root_ids,
                failure_tags_by_simple_id=None,
                figure_name=cfg.fast_plot.good_trace_figure_name,
                max_neurons=cfg.fast_plot.max_good_neurons,
            )

        should_plot_detailed, cond_flags = _should_plot_detailed_visualization(evaluation.metrics, cfg)
        if should_plot_detailed and evaluation.non_finite_count == 0:
            detailed_dir = trial_dir / "detailed_visualization"
            _plot_detailed_visualization(states=states, model=model, cfg_trial=cfg_trial, out_dir=detailed_dir)

        return {
            "objective_values": [float(v) for v in evaluation.objective_values],
            "metrics": evaluation.metrics,
            "non_finite_count": int(evaluation.non_finite_count),
            "detailed_plot_triggered": bool(should_plot_detailed),
            "detailed_plot_conditions": cond_flags,
        }
    finally:
        if model is not None:
            del model
        if runner is not None:
            del runner
        if dataloaders is not None:
            del dataloaders
        plt.close("all")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _trial_worker_main(
    queue: mp.Queue,
    cfg: TuneConfig,
    dirs: dict[str, Path],
    trial_number: int,
    params: dict[str, float],
    trial_overrides: list[str] | None,
) -> None:
    try:
        result = _run_trial_core(
            cfg=cfg,
            dirs=dirs,
            trial_number=trial_number,
            params=params,
            trial_overrides=trial_overrides,
        )
        queue.put({"ok": True, "result": result})
    except Exception:
        queue.put({"ok": False, "error": traceback.format_exc()})


def _make_objective(cfg: TuneConfig, dirs: dict[str, Path], trial_overrides: list[str] | None = None):
    def _objective(trial: optuna.trial.Trial) -> tuple[float, ...]:
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        _log_trial_memory("start", trial.number)

        params = _suggest_params(trial, cfg)

        if cfg.optuna.isolate_trial_process:
            ctx = mp.get_context("spawn")
            q: mp.Queue = ctx.Queue(maxsize=1)
            p = ctx.Process(
                target=_trial_worker_main,
                args=(q, cfg, dirs, int(trial.number), params, trial_overrides),
                daemon=False,
            )
            p.start()
            msg = q.get()
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Trial subprocess exited with code {p.exitcode}")
            if not msg.get("ok", False):
                raise RuntimeError(msg.get("error", "Unknown trial subprocess failure"))
            result = msg["result"]
        else:
            result = _run_trial_core(
                cfg=cfg,
                dirs=dirs,
                trial_number=int(trial.number),
                params=params,
                trial_overrides=trial_overrides,
            )

        for key, value in result.get("metrics", {}).items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("non_finite_count", int(result.get("non_finite_count", 0)))
        trial.set_user_attr("detailed_plot_triggered", bool(result.get("detailed_plot_triggered", False)))
        trial.set_user_attr("detailed_plot_conditions", result.get("detailed_plot_conditions", {}))

        _log_trial_memory("end", trial.number)
        return tuple(float(v) for v in result["objective_values"])

    return _objective


def run_study(cfg: TuneConfig, trial_overrides: list[str] | None = None):
    dirs = _create_dirs(cfg)
    storage_url = _resolve_storage(cfg, dirs["root"])
    sampler = _select_sampler(cfg)

    metric_names, directions = _objective_spec(cfg.objective)
    study = optuna.create_study(
        directions=directions,
        sampler=sampler,
        study_name=cfg.optuna.study_name,
        storage=storage_url,
        load_if_exists=cfg.optuna.load_if_exists,
    )
    study.set_metric_names(metric_names)

    study.optimize(
        _make_objective(cfg, dirs, trial_overrides=trial_overrides),
        n_trials=cfg.optuna.n_trials,
        timeout=cfg.optuna.timeout_s,
        n_jobs=cfg.optuna.n_jobs,
        show_progress_bar=cfg.optuna.show_progress_bar,
        gc_after_trial=cfg.optuna.gc_after_trial,
    )

    summary = {
        "n_trials_total": len(study.trials),
        "n_trials_complete": sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials),
        "best_trials": [
            {
                "number": t.number,
                "values": list(t.values or []),
                "params": t.params,
                "user_attrs": t.user_attrs,
            }
            for t in study.best_trials
        ],
    }
    (dirs["root"] / "study_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    if cfg.experiment.rerun_best_for_plots:
        rerun_best_and_plot(study, cfg, dirs)
    if cfg.experiment.rerun_all_non_nan_for_plots:
        rerun_all_non_nan_trials_and_plot(study, cfg, dirs)

    LOGGER.info("Experiment directory: %s", dirs["root"])


def _load_study_for_rerun(
    cfg: TuneConfig,
    *,
    study_name_override: str | None,
    storage_override: str | None,
) -> optuna.study.Study:
    dirs = _create_dirs(cfg)
    storage_url = _resolve_storage(cfg, dirs["root"])
    if storage_override:
        storage_url = _resolve_storage(
            TuneConfig(
                objective=cfg.objective,
                search_space=cfg.search_space,
                optuna=OptunaConfig(
                    **{
                        **asdict(cfg.optuna),
                        "storage": storage_override,
                    }
                ),
                experiment=cfg.experiment,
                fast_plot=cfg.fast_plot,
                output=cfg.output,
                reuse=cfg.reuse,
            ),
            dirs["root"],
        )

    study_name = study_name_override or cfg.optuna.study_name
    metric_names, directions = _objective_spec(cfg.objective)

    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage_url,
            sampler=_select_sampler(cfg),
        )
        try:
            study.set_metric_names(metric_names)
        except Exception:
            pass
        _ = directions
        return study
    except KeyError as exc:
        LOGGER.warning(
            "Study '%s' not found in storage '%s'; falling back to trial summaries rerun.",
            study_name,
            storage_url,
        )
        raise exc


def _load_trial_params_from_summary(trials_dir: Path, trial_id: int) -> dict[str, float] | None:
    summary_path = trials_dir / f"trial_{trial_id:05d}" / "summary.json"
    if not summary_path.exists():
        return None
    try:
        obj = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("Failed to read summary for trial %s: %s", trial_id, exc)
        return None

    params = obj.get("params") if isinstance(obj, dict) else None
    if not isinstance(params, dict):
        return None

    out: dict[str, float] = {}
    for k, v in params.items():
        if isinstance(v, (int, float)):
            out[str(k)] = float(v)
    return out if out else None


def rerun_selected_trials(
    cfg: TuneConfig,
    *,
    trial_ids: list[int],
    study_name_override: str | None,
    storage_override: str | None,
    rerun_out_subdir: str,
    rerun_overrides: list[str] | None = None,
) -> dict[str, Any]:
    if not trial_ids:
        raise ValueError("No trial ids provided. Use --trial-id and/or --trial-id-file.")

    dirs = _create_dirs(cfg)
    root = dirs["reruns"] / rerun_out_subdir
    root.mkdir(parents=True, exist_ok=True)

    rerun_summaries: list[dict[str, Any]] = []
    found_ids: list[int] = []
    missing_ids: list[int] = []
    source = "study"

    try:
        study = _load_study_for_rerun(
            cfg,
            study_name_override=study_name_override,
            storage_override=storage_override,
        )
        trial_map = {int(t.number): t for t in study.trials}
        found_ids = [tid for tid in trial_ids if tid in trial_map]
        missing_ids = [tid for tid in trial_ids if tid not in trial_map]

        for tid in found_ids:
            trial = trial_map[tid]
            if trial.state != TrialState.COMPLETE:
                LOGGER.warning("Skip trial %s because state is %s", tid, trial.state)
                continue
            out_dir = root / f"trial_{tid:05d}"
            rerun_summaries.append(
                rerun_params_and_store(
                    trial_number=int(trial.number),
                    params={k: float(v) for k, v in trial.params.items()},
                    cfg=cfg,
                    out_dir=out_dir,
                    trial_values=list(trial.values or []),
                    trial_params_raw=dict(trial.params),
                    rerun_overrides=rerun_overrides,
                )
            )
    except KeyError:
        source = "summary_json"
        trials_dir = dirs["trials"]
        for tid in trial_ids:
            params = _load_trial_params_from_summary(trials_dir, tid)
            if params is None:
                missing_ids.append(tid)
                continue
            found_ids.append(tid)
            out_dir = root / f"trial_{tid:05d}"
            rerun_summaries.append(
                rerun_params_and_store(
                    trial_number=tid,
                    params=params,
                    cfg=cfg,
                    out_dir=out_dir,
                    trial_values=None,
                    trial_params_raw=params,
                    rerun_overrides=rerun_overrides,
                )
            )

    if not found_ids:
        raise ValueError("None of requested trial ids found in study or trial summaries.")

    report = {
        "requested_trial_ids": trial_ids,
        "found_trial_ids": sorted(found_ids),
        "missing_trial_ids": sorted(missing_ids),
        "rerun_count": len(rerun_summaries),
        "rerun_output_dir": str(root.resolve()),
        "param_source": source,
        "rerun_overrides": list(rerun_overrides or []),
    }
    (root / "rerun_selected_summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    LOGGER.info("Rerun selected summary saved to %s", (root / "rerun_selected_summary.json"))
    return report


def main():
    args, unknown_cfg_args = _parse_cli_args()

    # Split unknown Hydra-style overrides into:
    # 1) TuneConfig-known overrides (fed to load_config), and
    # 2) pass-through rerun overrides (e.g., dataset=..., network=...).
    known_top_keys = {
        "objective",
        "search_space",
        "optuna",
        "experiment",
        "fast_plot",
        "output",
        "reuse",
        "metrics",
        "input",
    }
    cfg_overrides: list[str] = []
    passthrough_overrides: list[str] = []
    for item in unknown_cfg_args:
        if "=" not in item:
            cfg_overrides.append(item)
            continue
        k = item.split("=", 1)[0].lstrip("+")
        top = k.split(".", 1)[0]
        if top in known_top_keys:
            cfg_overrides.append(item)
        else:
            passthrough_overrides.append(item)

    old_argv = sys.argv
    try:
        # Keep compatibility with btorch/OmegaConf CLI overrides by removing
        # our custom flags before load_config parses argv.
        sys.argv = [old_argv[0], *cfg_overrides]
        cfg = load_config(TuneConfig)
    finally:
        sys.argv = old_argv

    np.random.seed(cfg.optuna.seed)
    torch.manual_seed(cfg.optuna.seed)

    if args.mode == "study":
        run_study(cfg, trial_overrides=passthrough_overrides)
        return

    trial_ids = _collect_requested_trial_ids(args)
    report = rerun_selected_trials(
        cfg,
        trial_ids=trial_ids,
        study_name_override=args.study_name,
        storage_override=args.storage,
        rerun_out_subdir=args.rerun_out_subdir,
        rerun_overrides=passthrough_overrides,
    )
    LOGGER.info(
        "Requested=%s Found=%s Missing=%s",
        report["requested_trial_ids"],
        report["found_trial_ids"],
        report["missing_trial_ids"],
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
