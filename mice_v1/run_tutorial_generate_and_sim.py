import json
import logging
import threading
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

try:
    import pynvml  # type: ignore
except Exception:
    pynvml = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from btorch.models import environ
from src.models.base import BaseRSNN
from src.runner import ExperimentRunner
from src.utils.dataloader import create_dataloaders
from src.utils.device import detect_device
from src.utils.other import set_seed, setup_logging


class GPUMemoryMonitor:
    """Periodically sample GPU memory and mark key pipeline steps."""

    def __init__(self, device: str, output_dir: Path, logger: logging.Logger, interval_sec: float = 0.2):
        self.device = str(device)
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.interval_sec = interval_sec
        self.samples = []
        self.key_steps = []
        self._running = False
        self._thread = None
        self._start_time = None
        self._nvml_ready = False
        self._nvml_handle = None

    def _get_gpu_index(self) -> int:
        if ":" in self.device:
            try:
                return int(self.device.split(":")[-1])
            except ValueError:
                return 0
        return 0

    def _init_nvml(self) -> None:
        if pynvml is None:
            return
        try:
            pynvml.nvmlInit()
            idx = self._get_gpu_index()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            self._nvml_ready = True
            self.logger.info("GPU monitor uses NVML on cuda:%d", idx)
        except Exception as exc:
            self.logger.warning("NVML init failed, fallback to torch.cuda stats: %s", exc)
            self._nvml_ready = False
            self._nvml_handle = None

    def _sample_once(self) -> None:
        if self._start_time is None:
            return

        elapsed = time.perf_counter() - self._start_time
        if self._nvml_ready and self._nvml_handle is not None:
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            used_mb = mem.used / (1024 ** 2)
            total_mb = mem.total / (1024 ** 2)
            source = "nvml"
        elif torch.cuda.is_available():
            idx = self._get_gpu_index()
            used_mb = torch.cuda.memory_allocated(idx) / (1024 ** 2)
            total_mb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 2)
            source = "torch.cuda"
        else:
            return

        self.samples.append(
            {
                "time_s": elapsed,
                "used_mb": used_mb,
                "total_mb": total_mb,
                "source": source,
            }
        )

    def _run(self) -> None:
        while self._running:
            self._sample_once()
            time.sleep(self.interval_sec)

    def start(self) -> None:
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available, GPU memory monitor is disabled.")
            return

        self._start_time = time.perf_counter()
        self._init_nvml()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.mark_step("monitor_started")

    def mark_step(self, name: str) -> None:
        if self._start_time is None:
            return
        self._sample_once()
        if not self.samples:
            return

        elapsed = time.perf_counter() - self._start_time
        last_used_mb = self.samples[-1]["used_mb"]
        self.key_steps.append({"name": name, "time_s": elapsed, "used_mb": last_used_mb})
        self.logger.info("[GPU-MONITOR] Step '%s' at %.2fs, used=%.2f MB", name, elapsed, last_used_mb)

    def stop(self) -> None:
        if not self._running:
            return

        self.mark_step("monitor_stopping")
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._sample_once()

        if self._nvml_ready:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def summarize_and_save(self):
        if not self.samples:
            self.logger.warning("No GPU samples were collected.")
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        max_sample = max(self.samples, key=lambda item: item["used_mb"])
        max_used = max_sample["used_mb"]
        max_t = max_sample["time_s"]

        nearest_step = None
        if self.key_steps:
            nearest_step = min(self.key_steps, key=lambda step: abs(step["time_s"] - max_t))

        csv_path = self.output_dir / "gpu_memory_usage.csv"
        with open(csv_path, "w", encoding="utf-8") as handle:
            handle.write("time_s,used_mb,total_mb,source\n")
            for sample in self.samples:
                handle.write(
                    f"{sample['time_s']:.6f},{sample['used_mb']:.6f},{sample['total_mb']:.6f},{sample['source']}\n"
                )

        step_labels = []
        for idx, step in enumerate(self.key_steps):
            label = chr(ord("A") + idx) if idx < 26 else f"P{idx + 1}"
            step_labels.append({**step, "label": label})

        step_path = self.output_dir / "gpu_key_steps.txt"
        with open(step_path, "w", encoding="utf-8") as handle:
            handle.write("label\ttime_s\tused_mb\tstep_name\n")
            for step in step_labels:
                handle.write(f"{step['label']}\t{step['time_s']:.6f}\t{step['used_mb']:.3f}\t{step['name']}\n")

        plot_path = None
        if plt is None:
            self.logger.warning("matplotlib unavailable, skip GPU curve plot.")
        else:
            ts = [sample["time_s"] for sample in self.samples]
            ys = [sample["used_mb"] for sample in self.samples]
            plt.figure(figsize=(13, 7.5))
            plt.plot(ts, ys, label="GPU used memory (MB)", linewidth=1.8)
            plt.scatter([max_t], [max_used], color="red", zorder=5, label=f"Max {max_used:.1f} MB")

            ymin = min(ys) if ys else 0.0
            ymax = max(ys) if ys else 1.0
            yrange = max(ymax - ymin, 1.0)

            for step in step_labels:
                plt.axvline(step["time_s"], linestyle="--", alpha=0.22)
                marker_y = ymin + 0.03 * yrange
                plt.scatter([step["time_s"]], [marker_y], s=22, color="black", zorder=6)
                plt.annotate(
                    step["label"],
                    xy=(step["time_s"], marker_y),
                    xytext=(0, -12),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                )

            plt.xlabel("Time (s)")
            plt.ylabel("GPU memory used (MB)")
            plt.title("GPU Memory Usage Over Time")
            plt.grid(alpha=0.25)
            plt.legend(loc="upper right")
            plt.tight_layout()

            plot_path = self.output_dir / "gpu_memory_usage_curve.png"
            plt.savefig(plot_path, dpi=180)
            plt.close()

        self.logger.info(
            "[GPU-MONITOR] Max GPU memory used: %.2f MB at %.2fs%s",
            max_used,
            max_t,
            f" (nearest step: {nearest_step['name']} @ {nearest_step['time_s']:.2f}s)" if nearest_step else "",
        )
        self.logger.info("[GPU-MONITOR] Samples saved to: %s", csv_path)
        self.logger.info("[GPU-MONITOR] Steps saved to: %s", step_path)
        if plot_path is not None:
            self.logger.info("[GPU-MONITOR] Plot saved to: %s", plot_path)

        return {
            "max_used_mb": max_used,
            "max_time_s": max_t,
            "nearest_step": nearest_step,
            "csv_path": str(csv_path),
            "step_path": str(step_path),
            "plot_path": str(plot_path) if plot_path else None,
        }


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def _pair_key(pre_ei: str, post_ei: str) -> str:
    return f"{pre_ei}{post_ei}"


def _sample_weight(tutorial_cfg, rng: np.random.Generator, pair_type: str) -> float:
    low = float(tutorial_cfg.network.weight_ranges[pair_type][0])
    high = float(tutorial_cfg.network.weight_ranges[pair_type][1])
    return float(rng.uniform(low, high))


def _build_connections_table(tutorial_cfg, neurons: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
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


def _write_glif_templates(tutorial_cfg, glif_dir: Path) -> None:
    glif_templates = OmegaConf.to_container(tutorial_cfg.glif_templates, resolve=True)
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


@hydra.main(version_base=None, config_path=".", config_name="tutorial_generate_and_sim")
def main(cfg: DictConfig):
    output_dir = Path(cfg.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    set_seed(cfg.seed)

    if cfg.device == "auto":
        cfg.device, _ = detect_device()

    logger.info("Tutorial Configuration:\n%s", OmegaConf.to_yaml(cfg))

    gpu_monitor = GPUMemoryMonitor(cfg.device, output_dir, logger, interval_sec=0.2)
    gpu_monitor.start()

    try:
        rng = np.random.default_rng(int(cfg.seed))
        tutorial_cfg = cfg.tutorial

        asset_root = _ensure_dir(output_dir / "tutorial_assets")
        connectome_dir = _ensure_dir(asset_root / "connectome")
        glif_dir = _ensure_dir(asset_root / "glif_models")

        # ============================================================================
        # Tutorial Step 1: Configure a tiny E/I network that is easy to inspect.
        # In the notebook, this is a good place to print the seed, neuron counts,
        # connection probabilities, and weight ranges before anything is generated.
        # ============================================================================
        gpu_monitor.mark_step("tutorial_step_1_configure_network")
        logger.info("============== Tutorial Step 1: Configure The Mock Network ==============")
        logger.info(
            "Network summary | n_exc=%d n_inh=%d spatial_span_um=%.1f",
            int(tutorial_cfg.network.n_exc),
            int(tutorial_cfg.network.n_inh),
            float(tutorial_cfg.network.spatial_span_um),
        )
        logger.info("Connection probabilities: %s", dict(tutorial_cfg.network.connection_probabilities))
        logger.info("Weight ranges: %s", dict(tutorial_cfg.network.weight_ranges))

        # ============================================================================
        # Tutorial Step 2: Generate the neuron table.
        # Each row is one neuron in the future recurrent network. This table is the
        # first object students will usually want to print and inspect.
        # ============================================================================
        gpu_monitor.mark_step("tutorial_step_2_generate_neurons")
        logger.info("============== Tutorial Step 2: Generate The Neuron Table ==============")
        neurons = _build_neurons_table(tutorial_cfg, rng)
        neurons.to_csv(connectome_dir / "neurons.csv.gz", index=False, compression="gzip")
        logger.info("Neuron table head:\n%s", neurons.head().to_string(index=False))
        logger.info("Neuron counts by EI type:\n%s", neurons["EI"].value_counts().to_string())

        # ============================================================================
        # Tutorial Step 3: Generate the connection table.
        # This is the explicit random network-building stage. In the notebook, this
        # is where edge histograms, adjacency previews, or degree plots can be added.
        # ============================================================================
        gpu_monitor.mark_step("tutorial_step_3_generate_connections")
        logger.info("============== Tutorial Step 3: Generate The Connection Table ==============")
        connections = _build_connections_table(tutorial_cfg, neurons, rng)
        connections.to_csv(connectome_dir / "connections.csv.gz", index=False, compression="gzip")

        pair_labels = (
            connections["EI"]
            + connections["post_simple_id"].map(neurons.set_index("simple_id")["EI"]).astype(str)
        )
        logger.info("Connection table head:\n%s", connections.head().to_string(index=False))
        logger.info("Connection counts by pair type:\n%s", pair_labels.value_counts().sort_index().to_string())
        logger.info("Total generated connections: %d", len(connections))

        # ============================================================================
        # Tutorial Step 4: Write two mock GLIF parameter templates.
        # We keep only one excitatory template and one inhibitory template so the
        # neuron-parameter loading path stays realistic without overwhelming detail.
        # ============================================================================
        gpu_monitor.mark_step("tutorial_step_4_write_glif_templates")
        logger.info("============== Tutorial Step 4: Write Mock GLIF Templates ==============")
        _write_glif_templates(tutorial_cfg, glif_dir)
        logger.info("GLIF template classes: %s", list(OmegaConf.to_container(tutorial_cfg.glif_templates, resolve=True).keys()))

        metadata = {
            "n_neurons": int(len(neurons)),
            "n_connections": int(len(connections)),
            "conn_path": str(connectome_dir),
            "glif_dir": str(glif_dir),
        }
        with open(asset_root / "metadata.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        # ============================================================================
        # Tutorial Step 5: Bridge the generated assets into the runtime config.
        # This is the point where the tutorial data becomes a normal simulation
        # input for BaseRSNN.
        # ============================================================================
        gpu_monitor.mark_step("tutorial_step_5_bridge_config")
        logger.info("============== Tutorial Step 5: Bridge Assets Into The Simulation Config ==============")
        OmegaConf.set_struct(cfg, False)
        cfg.network.conn_path = metadata["conn_path"]
        cfg.network.glif_dir = metadata["glif_dir"]
        cfg.network.n_neuron = metadata["n_neurons"]
        cfg.simulation.n_neuron = metadata["n_neurons"]
        OmegaConf.set_struct(cfg, True)
        logger.info("Tutorial connectome path: %s", cfg.network.conn_path)
        logger.info("Tutorial glif path: %s", cfg.network.glif_dir)
        logger.info("Tutorial neuron count: %s", cfg.network.n_neuron)

        # ============================================================================
        # Tutorial Step 6: Initialize the btorch simulation environment and build
        # the recurrent model.
        # In the notebook, this is where students can inspect the built model and
        # the loaded neuron/synapse parameter tensors.
        # ============================================================================
        gpu_monitor.mark_step("tutorial_step_6_build_model")
        logger.info("============== Tutorial Step 6: Build The Spiking Model ==============")
        environ.set(dt=cfg.simulation.dt)
        model = BaseRSNN(cfg)
        model.to(cfg.device)
        logger.info("Model built on device: %s", cfg.device)
        logger.info("Loaded neuron parameter keys: %s", sorted(model.neuron_params.keys()))

        # ============================================================================
        # Tutorial Step 7: Generate the external input batch.
        # The dataloader factory still handles the actual dataset creation, but we
        # explicitly preview one batch here so the tutorial can print or visualize
        # the injected input before the simulation starts.
        # ============================================================================
        gpu_monitor.mark_step("tutorial_step_7_generate_input")
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

        # ============================================================================
        # Tutorial Step 8: Run the actual recurrent simulation through the shared
        # ExperimentRunner.
        # This keeps the tutorial consistent with the main project while making the
        # earlier data-building steps much easier to explain cell by cell.
        # ============================================================================
        gpu_monitor.mark_step("tutorial_step_8_run_simulation")
        logger.info("============== Tutorial Step 8: Run The Simulation ==============")
        runner = ExperimentRunner(cfg, model, dataloaders)
        runner.run()
    finally:
        gpu_monitor.stop()
        gpu_monitor.summarize_and_save()


if __name__ == "__main__":
    main()
