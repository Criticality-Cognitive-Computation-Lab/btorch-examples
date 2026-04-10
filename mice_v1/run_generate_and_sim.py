import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
import sys
import time
import threading
from pathlib import Path

import torch

try:
    import pynvml  # type: ignore
except Exception:
    pynvml = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# --- 导入 Generator 模块 ---
# 获取当前脚本文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录路径 (即 network_generator 所在的目录)
parent_dir = os.path.dirname(current_dir)
network_pkg_path = os.path.join(parent_dir, "network_generator")

# 将这个具体路径加入 sys.path
sys.path.append(network_pkg_path)
from build_real_bluebrain_caller import build_brain_net

# --- 导入 Simulator 模块 ---
from src.utils.device import detect_device
from src.utils.other import setup_logging, set_seed
from btorch.models import environ
from src.models.base import BaseRSNN
from src.utils.dataloader import create_dataloaders
from src.runner import ExperimentRunner


class GPUMemoryMonitor:
    """周期采样 GPU 显存使用曲线，并记录关键步骤时间点。"""

    def __init__(self, device: str, output_dir: Path, logger: logging.Logger, interval_sec: float = 0.2):
        self.device = str(device)
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.interval_sec = interval_sec

        self.samples = []  # [{time_s, used_mb, total_mb, source}]
        self.key_steps = []  # [{name, time_s, used_mb}]

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

    def _init_nvml(self):
        if pynvml is None:
            return
        try:
            pynvml.nvmlInit()
            idx = self._get_gpu_index()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            self._nvml_ready = True
            self.logger.info(f"GPU monitor uses NVML on cuda:{idx}")
        except Exception as e:
            self.logger.warning(f"NVML init failed, fallback to torch.cuda stats: {e}")
            self._nvml_ready = False
            self._nvml_handle = None

    def _sample_once(self):
        if self._start_time is None:
            return

        t = time.perf_counter() - self._start_time

        if self._nvml_ready and self._nvml_handle is not None:
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            used_mb = mem.used / (1024 ** 2)
            total_mb = mem.total / (1024 ** 2)
            src = "nvml"
        elif torch.cuda.is_available():
            idx = self._get_gpu_index()
            used_mb = torch.cuda.memory_allocated(idx) / (1024 ** 2)
            total_mb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 2)
            src = "torch.cuda"
        else:
            return

        self.samples.append({
            "time_s": t,
            "used_mb": used_mb,
            "total_mb": total_mb,
            "source": src,
        })

    def _run(self):
        while self._running:
            self._sample_once()
            time.sleep(self.interval_sec)

    def start(self):
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available, GPU memory monitor is disabled.")
            return

        self._start_time = time.perf_counter()
        self._init_nvml()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.mark_step("monitor_started")

    def mark_step(self, name: str):
        if self._start_time is None:
            return

        self._sample_once()
        if not self.samples:
            return

        t = time.perf_counter() - self._start_time
        last_used_mb = self.samples[-1]["used_mb"]
        self.key_steps.append({"name": name, "time_s": t, "used_mb": last_used_mb})
        self.logger.info(f"[GPU-MONITOR] Step '{name}' at {t:.2f}s, used={last_used_mb:.2f} MB")

    def stop(self):
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

        max_sample = max(self.samples, key=lambda x: x["used_mb"])
        max_used = max_sample["used_mb"]
        max_t = max_sample["time_s"]

        nearest_step = None
        if self.key_steps:
            nearest_step = min(self.key_steps, key=lambda s: abs(s["time_s"] - max_t))

        # 保存原始采样
        csv_path = self.output_dir / "gpu_memory_usage.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("time_s,used_mb,total_mb,source\n")
            for s in self.samples:
                f.write(f"{s['time_s']:.6f},{s['used_mb']:.6f},{s['total_mb']:.6f},{s['source']}\n")

        # 给关键步骤分配简短标签（A/B/C...），避免图上文字过长
        step_labels = []
        for idx, s in enumerate(self.key_steps):
            if idx < 26:
                label = chr(ord("A") + idx)
            else:
                label = f"P{idx + 1}"
            step_labels.append({**s, "label": label})

        # 保存关键步骤（含标签）
        step_path = self.output_dir / "gpu_key_steps.txt"
        with open(step_path, "w", encoding="utf-8") as f:
            f.write("label\ttime_s\tused_mb\tstep_name\n")
            for s in step_labels:
                f.write(f"{s['label']}\t{s['time_s']:.6f}\t{s['used_mb']:.3f}\t{s['name']}\n")

        # 绘图
        plot_path = None
        if plt is None:
            self.logger.warning("matplotlib unavailable, skip GPU curve plot.")
        else:
            ts = [s["time_s"] for s in self.samples]
            ys = [s["used_mb"] for s in self.samples]

            plt.figure(figsize=(13, 7.5))
            plt.plot(ts, ys, label="GPU used memory (MB)", linewidth=1.8)
            plt.scatter([max_t], [max_used], color="red", zorder=5, label=f"Max {max_used:.1f} MB")

            ymin = min(ys) if ys else 0.0
            ymax = max(ys) if ys else 1.0
            yrange = max(ymax - ymin, 1.0)

            # 在图上仅标注简短标签，在 x 轴附近放置
            for s in step_labels:
                plt.axvline(s["time_s"], linestyle="--", alpha=0.22)
                marker_y = ymin + 0.03 * yrange
                plt.scatter([s["time_s"]], [marker_y], s=22, color="black", zorder=6)
                plt.annotate(
                    s["label"],
                    xy=(s["time_s"], marker_y),
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

            # 在图下方写标签说明（A: before_build_model ...）
            if step_labels:
                mapping_lines = [f"{s['label']}: {s['name']} ({s['time_s']:.2f}s)" for s in step_labels]
                max_lines_per_col = 12
                n_cols = max(1, (len(mapping_lines) + max_lines_per_col - 1) // max_lines_per_col)

                blocks = []
                for c in range(n_cols):
                    st = c * max_lines_per_col
                    ed = min((c + 1) * max_lines_per_col, len(mapping_lines))
                    blocks.append("\n".join(mapping_lines[st:ed]))

                for c, block in enumerate(blocks):
                    x_pos = 0.01 + c * (0.98 / n_cols)
                    plt.gcf().text(
                        x_pos,
                        0.01,
                        block,
                        ha="left",
                        va="bottom",
                        fontsize=8,
                        family="monospace",
                    )

                # 为底部说明预留空间
                bottom = 0.18 + 0.08 * (n_cols - 1)
                plt.tight_layout(rect=[0, bottom, 1, 1])
            else:
                plt.tight_layout()

            plot_path = self.output_dir / "gpu_memory_usage_curve.png"
            plt.savefig(plot_path, dpi=180)
            plt.close()

        msg = (
            f"[GPU-MONITOR] Max GPU memory used: {max_used:.2f} MB at {max_t:.2f}s"
            + (
                f" (nearest step: {nearest_step['name']} @ {nearest_step['time_s']:.2f}s)"
                if nearest_step is not None
                else ""
            )
        )
        self.logger.info(msg)
        self.logger.info(f"[GPU-MONITOR] Samples saved to: {csv_path}")
        self.logger.info(f"[GPU-MONITOR] Steps saved to: {step_path}")
        if plot_path is not None:
            self.logger.info(f"[GPU-MONITOR] Plot saved to: {plot_path}")

        return {
            "max_used_mb": max_used,
            "max_time_s": max_t,
            "nearest_step": nearest_step,
            "csv_path": str(csv_path),
            "step_path": str(step_path),
            "plot_path": str(plot_path) if plot_path else None,
        }


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # 0. 全局设置
    output_dir = Path(cfg.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    set_seed(cfg.seed)

    logger.info(f"Pipeline Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # 设备尽早确定，便于全流程监控
    if cfg.device == "auto":
        cfg.device, _ = detect_device()

    gpu_monitor = GPUMemoryMonitor(cfg.device, output_dir, logger, interval_sec=0.2)
    gpu_monitor.start()

    try:
        gpu_monitor.mark_step("before_network_generation")

        # ==========================================
        # Phase 1: Network Generation (Generator)
        # ==========================================
        logger.info("============== Phase 1: Network Generation ==============")

        generated_data = build_brain_net(cfg.generator)
        generated_path = generated_data["save_path"]
        neuron_num = generated_data["n_neurons"]
        net = generated_data["net"]
        registry = generated_data["registry"]
        ei_registry = generated_data["ei_registry"]
        dist_rule = generated_data["dist_rule"]

        logger.info(f"Network generated and saved at: {generated_path}")
        logger.info(f"Neuron number: {neuron_num}")
        logger.info(f"Net: {net}")
        logger.info(f"Registry: {registry}")
        logger.info(f"EI Registry: {ei_registry}")
        logger.info(f"Distance rule: {dist_rule}")

        gpu_monitor.mark_step("after_network_generation")

        # ==========================================
        # Phase 2: Configuration Bridging
        # ==========================================
        if cfg.bridge.use_generated_data:
            logger.info("🌉 Bridging: Injecting generated network path into simulation config...")

            OmegaConf.set_struct(cfg, False)  # 允许动态添加/修改字段
            cfg.network.conn_path = generated_path
            cfg.network.n_neuron = neuron_num
            cfg.simulation.n_neuron = neuron_num
            OmegaConf.set_struct(cfg, True)

            logger.info(f"Network conn path set to: {cfg.network.conn_path}")

        print(OmegaConf.to_yaml(cfg))

        # ==========================================
        # Phase 3: Spiking Network Simulation (Simulator)
        # ==========================================
        logger.info("============== Phase 2: Simulation ==============")

        output_dir = Path(cfg.out_dir)
        logger = setup_logging(output_dir)
        set_seed(cfg.seed)
        logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

        # 2. 设置物理环境 dt
        gpu_monitor.mark_step("before_environ_set")
        environ.set(dt=cfg.simulation.dt)
        gpu_monitor.mark_step("after_environ_set")

        # 3. 构建模型
        logger.info("🏗️  Building Model...")
        gpu_monitor.mark_step("before_build_model")
        model = BaseRSNN(cfg)
        model.to(cfg.device)
        gpu_monitor.mark_step("after_build_model")

        #breakpoint()

        i_thr = model.neuron_params.get("I_thr", None)

        # 4. 构建数据
        logger.info("📦 Preparing Data...")
        gpu_monitor.mark_step("before_prepare_data")
        dataloaders = create_dataloaders(cfg, i_thr)
        gpu_monitor.mark_step("after_prepare_data")

        # 5. 运行实验
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
