import json
import logging
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from btorch.models import environ
from src.models.base import BaseRSNN
from src.runner import ExperimentRunner
from src.utils.dataloader import create_dataloaders
from src.utils.device import detect_device
from src.utils.other import set_seed, setup_logging

import ipdb


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def initialize_tutorial_context(cfg: DictConfig) -> dict[str, Any]:
    """
    Prepare the tutorial workspace, logger, random seed, and output folders.

    Notebook usage:
        context = initialize_tutorial_context(cfg)
    """
    output_dir = Path(cfg.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    set_seed(cfg.seed)

    if cfg.device == "auto":
        cfg.device, _ = detect_device()

    rng = np.random.default_rng(int(cfg.seed))
    asset_root = _ensure_dir(output_dir / "tutorial_assets")
    connectome_dir = _ensure_dir(asset_root / "connectome")
    glif_dir = _ensure_dir(asset_root / "glif_models")

    logger.info("Tutorial Configuration:\n%s", OmegaConf.to_yaml(cfg))

    return {
        "cfg": cfg,
        "logger": logger,
        "output_dir": output_dir,
        "asset_root": asset_root,
        "connectome_dir": connectome_dir,
        "glif_dir": glif_dir,
        "tutorial_cfg": cfg.tutorial,
        "rng": rng,
    }


def tutorial_step_1_configure_network(context: dict[str, Any]) -> dict[str, Any]:
    """
    Summarize the tiny E/I tutorial network before generating any data.

    Notebook usage:
        step1 = tutorial_step_1_configure_network(context)
        print(step1["summary"])
    """
    tutorial_cfg = context["tutorial_cfg"]
    logger = context["logger"]

    summary = {
        "seed": int(context["cfg"].seed),
        "n_exc": int(tutorial_cfg.network.n_exc),
        "n_inh": int(tutorial_cfg.network.n_inh),
        "spatial_span_um": float(tutorial_cfg.network.spatial_span_um),
        "connection_probabilities": dict(tutorial_cfg.network.connection_probabilities),
        "weight_ranges": dict(tutorial_cfg.network.weight_ranges),
    }

    logger.info("============== Tutorial Step 1: Configure The Mock Network ==============")
    logger.info(
        "Network summary | n_exc=%d n_inh=%d spatial_span_um=%.1f",
        summary["n_exc"],
        summary["n_inh"],
        summary["spatial_span_um"],
    )
    logger.info("Connection probabilities: %s", summary["connection_probabilities"])
    logger.info("Weight ranges: %s", summary["weight_ranges"])

    return {"summary": summary}


def _build_neurons_table(tutorial_cfg, rng: np.random.Generator) -> pd.DataFrame:
    n_exc = int(tutorial_cfg.network.n_exc)
    n_inh = int(tutorial_cfg.network.n_inh)
    n_total = n_exc + n_inh

    span = float(tutorial_cfg.network.spatial_span_um)
    coords = rng.uniform(0.0, span, size=(n_total, 3)).astype(np.float32)

    rows = []
    for simple_id in range(n_total):
        is_exc = simple_id < n_exc
        rows.append(
            {
                "root_id": 1000 + simple_id,
                "simple_id": simple_id,
                "type": "Excitatory" if is_exc else "Inhibitory",
                "layer": "L4",
                "cell_class": "tutorial_pyr" if is_exc else "tutorial_pv",
                "EI": "E" if is_exc else "I",
                "x_position": coords[simple_id, 0],
                "y_position": coords[simple_id, 1],
                "z_position": coords[simple_id, 2],
            }
        )
    return pd.DataFrame(rows)


def tutorial_step_2_generate_neurons(context: dict[str, Any]) -> dict[str, Any]:
    """
    Generate and save the neuron table used by the recurrent network.

    Notebook usage:
        step2 = tutorial_step_2_generate_neurons(context)
        step2["neurons"].head()
    """
    tutorial_cfg = context["tutorial_cfg"]
    rng = context["rng"]
    logger = context["logger"]
    connectome_dir = context["connectome_dir"]

    neurons = _build_neurons_table(tutorial_cfg, rng)
    neurons_path = connectome_dir / "neurons.csv.gz"
    neurons.to_csv(neurons_path, index=False, compression="gzip")

    logger.info("============== Tutorial Step 2: Generate The Neuron Table ==============")
    logger.info("Neuron table head:\n%s", neurons.head().to_string(index=False))
    logger.info("Neuron counts by EI type:\n%s", neurons["EI"].value_counts().to_string())

    return {
        "neurons": neurons,
        "neurons_path": neurons_path,
        "neuron_counts_by_ei": neurons["EI"].value_counts(),
    }


def _pair_key(pre_ei: str, post_ei: str) -> str:
    return f"{pre_ei}{post_ei}"


def _sample_weight(tutorial_cfg, rng: np.random.Generator, pair_type: str) -> float:
    low = float(tutorial_cfg.network.weight_ranges[pair_type][0])
    high = float(tutorial_cfg.network.weight_ranges[pair_type][1])
    return float(rng.uniform(low, high))


def _build_connections_table(
    tutorial_cfg, neurons: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
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
            prob = float(tutorial_cfg.network.connection_probabilities[pair_type])
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
                    "weight": _sample_weight(tutorial_cfg, rng, pair_type),
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
                "weight": _sample_weight(tutorial_cfg, rng, "EE"),
            }
        )

    inh_indices = simple_ids[ei_types == "I"]
    exc_indices = simple_ids[ei_types == "E"]
    if inh_indices.size and exc_indices.size:
        existing_ie = any(
            row["EI"] == "I" and neurons.iloc[int(row["post_simple_id"])]["EI"] == "E"
            for row in rows
        )
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
                    "weight": _sample_weight(tutorial_cfg, rng, "IE"),
                }
            )

    connections = pd.DataFrame(rows)
    connections = connections.drop_duplicates(subset=["pre_simple_id", "post_simple_id"], keep="first")
    connections = connections.sort_values(["pre_simple_id", "post_simple_id"]).reset_index(drop=True)
    return connections


def tutorial_step_3_generate_connections(
    context: dict[str, Any], neurons: pd.DataFrame
) -> dict[str, Any]:
    """
    Generate and save the random connection table.

    Notebook usage:
        step3 = tutorial_step_3_generate_connections(context, step2["neurons"])
        step3["connections"].head()
    """
    tutorial_cfg = context["tutorial_cfg"]
    rng = context["rng"]
    logger = context["logger"]
    connectome_dir = context["connectome_dir"]

    connections = _build_connections_table(tutorial_cfg, neurons, rng)
    connections_path = connectome_dir / "connections.csv.gz"
    connections.to_csv(connections_path, index=False, compression="gzip")

    pair_labels = (
        connections["EI"]
        + connections["post_simple_id"].map(neurons.set_index("simple_id")["EI"]).astype(str)
    )
    pair_counts = pair_labels.value_counts().sort_index()

    logger.info("============== Tutorial Step 3: Generate The Connection Table ==============")
    logger.info("Connection table head:\n%s", connections.head().to_string(index=False))
    logger.info("Connection counts by pair type:\n%s", pair_counts.to_string())
    logger.info("Total generated connections: %d", len(connections))

    return {
        "connections": connections,
        "connections_path": connections_path,
        "pair_labels": pair_labels,
        "pair_counts": pair_counts,
        "n_connections": int(len(connections)),
    }


def _write_glif_templates(tutorial_cfg, glif_dir: Path) -> list[str]:
    glif_templates = OmegaConf.to_container(tutorial_cfg.glif_templates, resolve=True)
    class_names = []
    for class_name, params in glif_templates.items():
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
        class_names.append(class_name)
    return class_names


def tutorial_step_4_write_glif_templates(context: dict[str, Any]) -> dict[str, Any]:
    """
    Write one excitatory and one inhibitory GLIF template directory.

    Notebook usage:
        step4 = tutorial_step_4_write_glif_templates(context)
        print(step4["class_names"])
    """
    tutorial_cfg = context["tutorial_cfg"]
    logger = context["logger"]
    glif_dir = context["glif_dir"]

    class_names = _write_glif_templates(tutorial_cfg, glif_dir)

    logger.info("============== Tutorial Step 4: Write Mock GLIF Templates ==============")
    logger.info("GLIF template classes: %s", class_names)

    return {
        "glif_dir": glif_dir,
        "class_names": class_names,
    }


def tutorial_step_5_bridge_assets(
    context: dict[str, Any], neurons: pd.DataFrame, connections: pd.DataFrame
) -> dict[str, Any]:
    """
    Update the Hydra config so the generated tutorial assets become model inputs.

    Notebook usage:
        step5 = tutorial_step_5_bridge_assets(context, step2["neurons"], step3["connections"])
        print(step5["metadata"])
    """
    cfg = context["cfg"]
    logger = context["logger"]
    asset_root = context["asset_root"]
    connectome_dir = context["connectome_dir"]
    glif_dir = context["glif_dir"]

    metadata = {
        "n_neurons": int(len(neurons)),
        "n_connections": int(len(connections)),
        "conn_path": str(connectome_dir),
        "glif_dir": str(glif_dir),
    }
    with open(asset_root / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    OmegaConf.set_struct(cfg, False)
    cfg.network.conn_path = metadata["conn_path"]
    cfg.network.glif_dir = metadata["glif_dir"]
    cfg.network.n_neuron = metadata["n_neurons"]
    cfg.simulation.n_neuron = metadata["n_neurons"]
    OmegaConf.set_struct(cfg, True)

    logger.info("============== Tutorial Step 5: Bridge Assets Into The Simulation Config ==============")
    logger.info("Tutorial connectome path: %s", cfg.network.conn_path)
    logger.info("Tutorial glif path: %s", cfg.network.glif_dir)
    logger.info("Tutorial neuron count: %s", cfg.network.n_neuron)

    return {"metadata": metadata, "cfg": cfg}


def tutorial_step_6_build_model(context: dict[str, Any]) -> dict[str, Any]:
    """
    Build the spiking model with the current config and generated asset paths.

    Notebook usage:
        step6 = tutorial_step_6_build_model(context)
        step6["model"]
    """
    cfg = context["cfg"]
    logger = context["logger"]

    logger.info("============== Tutorial Step 6: Build The Spiking Model ==============")
    environ.set(dt=cfg.simulation.dt)
    model = BaseRSNN(cfg)
    model.to(cfg.device)

    logger.info("Model built on device: %s", cfg.device)
    logger.info("Loaded neuron parameter keys: %s", sorted(model.neuron_params.keys()))

    return {
        "model": model,
        "neuron_params": model.neuron_params,
        "connectome_data": model.connectome_data,
    }


def tutorial_step_7_generate_input(
    context: dict[str, Any], model: BaseRSNN
) -> dict[str, Any]:
    """
    Build the dataloaders and preview one input batch before simulation.

    Notebook usage:
        step7 = tutorial_step_7_generate_input(context, step6["model"])
        step7["example_batch"].shape
    """
    cfg = context["cfg"]
    logger = context["logger"]

    logger.info("============== Tutorial Step 7: Generate The Input Batch ==============")
    i_thr = model.neuron_params.get("I_thr", None)
    dataloaders = create_dataloaders(cfg, i_thr)
    _, test_loader = dataloaders

    example_batch = next(iter(test_loader))
    if isinstance(example_batch, (list, tuple)):
        example_batch = example_batch[0]

    logger.info(
        "Example input batch | shape=%s mean=%.4f std=%.4f min=%.4f max=%.4f",
        tuple(example_batch.shape),
        float(example_batch.mean().item()),
        float(example_batch.std().item()),
        float(example_batch.min().item()),
        float(example_batch.max().item()),
    )

    return {
        "dataloaders": dataloaders,
        "example_batch": example_batch,
        "i_thr": i_thr,
    }


def tutorial_step_8_run_simulation(
    context: dict[str, Any], model: BaseRSNN, dataloaders
) -> dict[str, Any]:
    """
    Run the recurrent simulation through the shared ExperimentRunner.

    Notebook usage:
        step8 = tutorial_step_8_run_simulation(context, step6["model"], step7["dataloaders"])
    """
    cfg = context["cfg"]
    logger = context["logger"]

    logger.info("============== Tutorial Step 8: Run The Simulation ==============")
    runner = ExperimentRunner(cfg, model, dataloaders)
    runner.run()

    return {"runner": runner}


def run_all_tutorial_steps(cfg: DictConfig) -> dict[str, Any]:
    """
    Convenience wrapper for script usage. In a notebook, call the step functions
    individually instead.
    """
    contexts = initialize_tutorial_context(cfg)
    breakpoint()
    step1 = tutorial_step_1_configure_network(contexts)
    step2 = tutorial_step_2_generate_neurons(contexts)
    step3 = tutorial_step_3_generate_connections(contexts, step2["neurons"])
    step4 = tutorial_step_4_write_glif_templates(contexts)
    step5 = tutorial_step_5_bridge_assets(contexts, step2["neurons"], step3["connections"])
    step6 = tutorial_step_6_build_model(contexts)
    step7 = tutorial_step_7_generate_input(contexts, step6["model"])
    step8 = tutorial_step_8_run_simulation(contexts, step6["model"], step7["dataloaders"])

    return {
        "context": contexts,
        "step1": step1,
        "step2": step2,
        "step3": step3,
        "step4": step4,
        "step5": step5,
        "step6": step6,
        "step7": step7,
        "step8": step8,
    }


@hydra.main(version_base=None, config_path=".", config_name="tutorial_generate_and_sim")
def main(cfg: DictConfig):
    run_all_tutorial_steps(cfg)


if __name__ == "__main__":
    main()
