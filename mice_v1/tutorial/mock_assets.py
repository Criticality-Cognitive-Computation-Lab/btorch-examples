from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf


LOGGER = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_neurons(cfg, rng: np.random.Generator) -> pd.DataFrame:
    n_exc = int(cfg.network.n_exc)
    n_inh = int(cfg.network.n_inh)
    n_total = n_exc + n_inh

    span = float(cfg.network.spatial_span_um)
    coords = rng.uniform(0.0, span, size=(n_total, 3)).astype(np.float32)

    rows = []
    for simple_id in range(n_total):
        is_exc = simple_id < n_exc
        cell_class = "tutorial_pyr" if is_exc else "tutorial_pv"
        rows.append(
            {
                "root_id": 1000 + simple_id,
                "simple_id": simple_id,
                "type": "Excitatory" if is_exc else "Inhibitory",
                "layer": "L4",
                "cell_class": cell_class,
                "EI": "E" if is_exc else "I",
                "x_position": coords[simple_id, 0],
                "y_position": coords[simple_id, 1],
                "z_position": coords[simple_id, 2],
            }
        )

    return pd.DataFrame(rows)


def _pair_key(pre_ei: str, post_ei: str) -> str:
    return f"{pre_ei}{post_ei}"


def _sample_weight(cfg, rng: np.random.Generator, pair_type: str) -> float:
    low = float(cfg.network.weight_ranges[pair_type][0])
    high = float(cfg.network.weight_ranges[pair_type][1])
    return float(rng.uniform(low, high))


def _build_connections(cfg, neurons: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    simple_ids = neurons["simple_id"].to_numpy(dtype=int)
    root_ids = neurons["root_id"].to_numpy(dtype=int)
    cell_classes = neurons["cell_class"].to_numpy(dtype=object)
    ei_types = neurons["EI"].to_numpy(dtype=object)

    for pre in simple_ids:
        for post in simple_ids:
            if pre == post:
                continue

            pair_type = _pair_key(str(ei_types[pre]), str(ei_types[post]))
            prob = float(cfg.network.connection_probabilities[pair_type])
            if rng.random() > prob:
                continue

            rows.append(
                {
                    "pre_id": int(root_ids[pre]),
                    "post_id": int(root_ids[post]),
                    "pre_simple_id": int(pre),
                    "post_simple_id": int(post),
                    "pre_type": str(cell_classes[pre]),
                    "post_type": str(cell_classes[post]),
                    "EI": str(ei_types[pre]),
                    "syn_count": 1,
                    "weight": _sample_weight(cfg, rng, pair_type),
                }
            )

    if not rows:
        rows.append(
            {
                "pre_id": int(root_ids[0]),
                "post_id": int(root_ids[1]),
                "pre_simple_id": 0,
                "post_simple_id": 1,
                "pre_type": str(cell_classes[0]),
                "post_type": str(cell_classes[1]),
                "EI": str(ei_types[0]),
                "syn_count": 1,
                "weight": _sample_weight(cfg, rng, "EE"),
            }
        )

    # Guarantee at least one outgoing inhibitory edge in the tiny demo network.
    inh_indices = simple_ids[ei_types == "I"]
    exc_indices = simple_ids[ei_types == "E"]
    if inh_indices.size and exc_indices.size:
        existing_ie = any(row["EI"] == "I" and neurons.iloc[row["post_simple_id"]]["EI"] == "E" for row in rows)
        if not existing_ie:
            pre = int(inh_indices[0])
            post = int(exc_indices[0])
            rows.append(
                {
                    "pre_id": int(root_ids[pre]),
                    "post_id": int(root_ids[post]),
                    "pre_simple_id": pre,
                    "post_simple_id": post,
                    "pre_type": str(cell_classes[pre]),
                    "post_type": str(cell_classes[post]),
                    "EI": str(ei_types[pre]),
                    "syn_count": 1,
                    "weight": _sample_weight(cfg, rng, "IE"),
                }
            )

    connections = pd.DataFrame(rows)
    connections = connections.drop_duplicates(subset=["pre_simple_id", "post_simple_id"], keep="first")
    connections = connections.sort_values(["pre_simple_id", "post_simple_id"]).reset_index(drop=True)
    return connections


def _write_glif_templates(cfg, glif_dir: Path) -> None:
    for class_name, params in OmegaConf.to_container(cfg.glif_templates, resolve=True).items():
        class_dir = _ensure_dir(glif_dir / class_name)
        payload = {
            "coeffs": {
                "C": 1.0,
                "th_inf": 1.0,
                "asc_amp_array": [1.0 for _ in params["asc_amp_array"]],
            },
            "C": params["C"],
            "R_input": params["R_input"],
            "spike_cut_length": params["spike_cut_length"],
            "dt": params["dt"],
            "dt_multiplier": params["dt_multiplier"],
            "El_reference": params["El_reference"],
            "El": params["El"],
            "th_inf": params["th_inf"],
            "asc_tau_array": params["asc_tau_array"],
            "asc_amp_array": params["asc_amp_array"],
        }
        with open(class_dir / "template.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


def build_tutorial_assets(cfg, output_dir: Path) -> dict[str, str | int]:
    tutorial_cfg = cfg.tutorial
    rng = np.random.default_rng(int(cfg.seed))

    asset_root = _ensure_dir(output_dir / "tutorial_assets")
    conn_dir = _ensure_dir(asset_root / "connectome")
    glif_dir = _ensure_dir(asset_root / "glif_models")

    neurons = _build_neurons(tutorial_cfg, rng)
    connections = _build_connections(tutorial_cfg, neurons, rng)

    neurons.to_csv(conn_dir / "neurons.csv.gz", index=False, compression="gzip")
    connections.to_csv(conn_dir / "connections.csv.gz", index=False, compression="gzip")
    _write_glif_templates(tutorial_cfg, glif_dir)

    metadata = {
        "n_neurons": int(len(neurons)),
        "n_connections": int(len(connections)),
        "conn_path": str(conn_dir),
        "glif_dir": str(glif_dir),
    }
    with open(asset_root / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    LOGGER.info(
        "Tutorial mock assets prepared: n_neurons=%d n_connections=%d path=%s",
        metadata["n_neurons"],
        metadata["n_connections"],
        conn_dir,
    )
    return metadata
