import logging
import time
import threading
from pathlib import Path

import hydra
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
from tutorial.mock_assets import build_tutorial_assets


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

            if step_labels:
                mapping_lines = [f"{step['label']}: {step['name']} ({step['time_s']:.2f}s)" for step in step_labels]
                max_lines_per_col = 12
                n_cols = max(1, (len(mapping_lines) + max_lines_per_col - 1) // max_lines_per_col)
                blocks = []
                for col in range(n_cols):
                    start = col * max_lines_per_col
                    end = min((col + 1) * max_lines_per_col, len(mapping_lines))
                    blocks.append("\n".join(mapping_lines[start:end]))

                for col, block in enumerate(blocks):
                    x_pos = 0.01 + col * (0.98 / n_cols)
                    plt.gcf().text(x_pos, 0.01, block, ha="left", va="bottom", fontsize=8, family="monospace")

                bottom = 0.18 + 0.08 * (n_cols - 1)
                plt.tight_layout(rect=[0, bottom, 1, 1])
            else:
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
        gpu_monitor.mark_step("before_mock_asset_build")
        logger.info("============== Phase 1: Tutorial Mock Asset Build ==============")
        tutorial_assets = build_tutorial_assets(cfg, output_dir)
        gpu_monitor.mark_step("after_mock_asset_build")

        OmegaConf.set_struct(cfg, False)
        cfg.network.conn_path = tutorial_assets["conn_path"]
        cfg.network.glif_dir = tutorial_assets["glif_dir"]
        cfg.network.n_neuron = tutorial_assets["n_neurons"]
        cfg.simulation.n_neuron = tutorial_assets["n_neurons"]
        OmegaConf.set_struct(cfg, True)

        logger.info("Tutorial connectome path set to: %s", cfg.network.conn_path)
        logger.info("Tutorial glif path set to: %s", cfg.network.glif_dir)
        logger.info("Tutorial neuron count: %s", cfg.network.n_neuron)

        logger.info("============== Phase 2: Simulation ==============")
        gpu_monitor.mark_step("before_environ_set")
        environ.set(dt=cfg.simulation.dt)
        gpu_monitor.mark_step("after_environ_set")

        logger.info("🏗️  Building Model...")
        gpu_monitor.mark_step("before_build_model")
        model = BaseRSNN(cfg)
        model.to(cfg.device)
        gpu_monitor.mark_step("after_build_model")

        i_thr = model.neuron_params.get("I_thr", None)

        logger.info("📦 Preparing Data...")
        gpu_monitor.mark_step("before_prepare_data")
        dataloaders = create_dataloaders(cfg, i_thr)
        gpu_monitor.mark_step("after_prepare_data")

        logger.info("🏃 Starting Experiment Runner...")
        runner = ExperimentRunner(cfg, model, dataloaders)
        gpu_monitor.mark_step("before_runner_run")
        runner.run()
        gpu_monitor.mark_step("after_runner_run")
    finally:
        gpu_monitor.stop()
        gpu_monitor.summarize_and_save()


if __name__ == "__main__":
    main()
