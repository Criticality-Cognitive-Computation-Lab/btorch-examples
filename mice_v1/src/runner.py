import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from btorch.models import environ, functional, init

from src.utils.other import is_train_mode
from src.utils.vis_utils import (
    prepare_data_from_dict,
    visualize_results,
    select_neurons_by_class,
    save_states_full,
)

import metric as col_metrics

import ipdb

logger = logging.getLogger(__name__)

class ExperimentRunner:
    def __init__(self, cfg, model, dataloaders):
        self.cfg = cfg
        self.model = model
        self.train_loader, self.test_loader = dataloaders
        self.device = cfg.device

    def _resolve_visualize_flag(self):
        if "visualize" in self.cfg:
            return bool(self.cfg.visualize)
        if "generator" in self.cfg and "visualize" in self.cfg.generator:
            return bool(self.cfg.generator.visualize)
        return True

    def run(self):
        """主入口：决定跑什么模式"""
        if is_train_mode(self.cfg):
            self.run_training()
        else:
            #breakpoint()
            self.run_simulation(visualize=self._resolve_visualize_flag())

    def _init_model_states(self, batch_size=None):
        """初始化模型的所有状态变量"""
        # 1. 首先调用 init_state 创建所有状态变量
        functional.init_net_state(self.model, batch_size=batch_size, device=self.device)
        
        # 2. 然后调用 reset_net 将所有状态重置为初始值
        functional.reset_net(self.model, batch_size=batch_size, device=self.device)
        
        # 3. 对神经元进行特定的初始化（如均匀分布）
        neuron_module = self.model.flybrain.brain.neuron
        init.uniform_v_(neuron_module)

    def run_simulation(self, visualize = True):
        """Simulation Mode: 不算梯度，不反传"""
        logger.info("🚀 Starting SIMULATION mode (No Gradient)")
        self.model.eval()
        self.model.to(self.device)
        self._init_model_states(batch_size=self.cfg.simulation.batch_size)
    
        
        # 只需要简单的 Dummy 输入或者 Test Loader 的一个 Batch
        # 这里假设我们想跑个示例
        with torch.no_grad():
            # 构造一个示例输入 (如果没有 dataloader)
            # 实际应从 self.test_loader 取数据
            if self.test_loader:
                #breakpoint()
                inputs = next(iter(self.test_loader))
            else:
                #breakpoint()
                # Fallback: 手动造数据用于测试
                T = self.cfg.simulation.T
                N_in = self.cfg.network.num_input_neuron
                inputs = torch.randn(1, T, N_in).to(self.device) # Dummy
                
            inputs = inputs.to(self.device) #[bs, T, n_neuron]
            #breakpoint()
            output, states = self.model(inputs)
            #breakpoint()
            
            logger.info("✅ Simulation complete.")
            # 保存结果逻辑 (vis_result)
            self._save_results(states, "simulation_output")
            self.run_metrics(states, is_eval=True, epoch_id=None)
            if visualize:
                self.run_visualize(states, is_eval=True, epoch_id=None)
            logger.info("✅ Visualization complete.")

    def _as_numpy(self, value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def compute_metrics(self, states):
        #breakpoint()
        spikes = states["neuron"]["spike"]
        v = states["neuron"]["v"]
        synapse = states.get("synapse", {})
        psc_e = synapse.get("psc_e")
        psc_i = synapse.get("psc_i")
        I_ext = synapse.get("I_ext")

        if spikes is None or v is None:
            raise ValueError("Missing spike or voltage states for metric computation.")

        spikes_t = self._as_numpy(spikes)
        v_t = self._as_numpy(v)
        psc_e_t = None if psc_e is None else torch.as_tensor(psc_e)
        psc_i_t = None if psc_i is None else torch.as_tensor(psc_i)

        #I_ext外界输入电流矩阵
        I_ext = self._as_numpy(I_ext)
        #consider only E neurons' psc
        #breakpoint()
        e_mask = self.model.connectome_data[0]["EI"] == "E"
        #spikes_t = spikes_t[e_mask]
        #v_t = v_t[e_mask]
        psc_e_t = psc_e_t[:, :, e_mask]
        psc_i_t = psc_i_t[:, :, e_mask]
        if I_ext is not None:
            I_ext = I_ext[..., e_mask]

        if np.isnan(spikes_t).any() or np.isnan(v_t).any():
            return {
                "objective_values": (1.0, 1e6, 1e6, 1e6),
                "metrics": {"nan_penalty": 1.0},
            }

        dt = float(self.cfg.simulation.dt)
        ei_metrics, ei_info = col_metrics.compute_ei_balance(
            I_e=psc_e_t,
            I_i=psc_i_t,
            I_ext=I_ext,
            dt=dt,
        )
        ei_obj = max(0.0, ei_metrics["eci_mean"] - 0.15) + max(
            0.0, 0.60 - ei_metrics["track_corr_peak_mean"]
        )

        dyn_df = col_metrics.compute_dynamics_metrics(spikes_t, dt=dt)
        active_mask = dyn_df["rate_hz"] > 0.2
        cv_vals = dyn_df["cv"][active_mask].values
        rate_vals = dyn_df["rate_hz"].values
        fano_vals = dyn_df["fano"].values
        kurt_vals = dyn_df["kurtosis"].values
        if cv_vals.size:
            cv_penalty = float(
                np.mean(
                    np.maximum(0, 0.6 - cv_vals) + np.maximum(0, cv_vals - 1.8)
                )
            )
        else:
            cv_penalty = 10.0
        dynamics_obj = cv_penalty

        v_metrics, v_info = col_metrics.compute_voltage_clamp(
            v_t,
            v_min=-80.0,
            v_max=40.0,
        )
        r_metrics, r_info = col_metrics.compute_rate_clamp(
            spikes_t,
            dt=dt,
            rate_min=0.5,
            rate_max=80.0,
        )
        stability_obj = float(v_metrics["v_penalty_mean"]) + float(
            r_metrics["rate_penalty_mean"]
        )

        return {
            "objective_values": (0.0, float(ei_obj), float(dynamics_obj), float(stability_obj)),
            "metrics": {
                "nan_penalty": 0.0,
                "ei_obj": float(ei_obj),
                "dynamics_obj": float(dynamics_obj),
                "stability_obj": float(stability_obj),
                # EI balance summary
                "eci_mean": float(ei_metrics["eci_mean"]),
                "eci_median": float(ei_metrics.get("eci_median", np.nan)),
                "eci_p90": float(ei_metrics.get("eci_p90", np.nan)),
                "track_corr_peak_mean": float(ei_metrics["track_corr_peak_mean"]),
                "track_corr_peak_median": float(ei_metrics.get("track_corr_peak_median", np.nan)),
                "track_corr_peak_p90": float(ei_metrics.get("track_corr_peak_p90", np.nan)),
                "delay_ms_mean": float(ei_metrics.get("delay_ms_mean", np.nan)),
                "delay_ms_median": float(ei_metrics.get("delay_ms_median", np.nan)),
                "delay_ms_abs_mean": float(ei_metrics.get("delay_ms_abs_mean", np.nan)),
                "lag_bins": float(ei_info.get("lag_bins", np.nan)),
                "dt_ms": float(ei_info.get("dt_ms", dt)),

                # Raw dynamics (all neurons)
                "rate_hz_mean_raw": float(np.nanmean(rate_vals)),
                "rate_hz_median_raw": float(np.nanmedian(rate_vals)),
                "rate_hz_p90_raw": float(np.nanpercentile(rate_vals, 90)),
                "cv_mean_raw": float(np.nanmean(dyn_df["cv"].values)),
                "cv_median_raw": float(np.nanmedian(dyn_df["cv"].values)),
                "cv_mean_active_raw": float(np.nanmean(cv_vals)) if cv_vals.size else float("nan"),
                "fano_mean_raw": float(np.nanmean(fano_vals)),
                "fano_median_raw": float(np.nanmedian(fano_vals)),
                "kurtosis_mean_raw": float(np.nanmean(kurt_vals)),

                # Clamp metrics/info
                "rate_penalty_mean": float(r_metrics.get("rate_penalty_mean", np.nan)),
                "rate_min_used": float(r_info.get("rate_min", np.nan)),
                "rate_max_used": float(r_info.get("rate_max", np.nan)),
                "v_penalty_mean": float(v_metrics.get("v_penalty_mean", np.nan)),
                "v_min_mean_used": float(v_info.get("v_min_mean", np.nan)),
                "v_max_mean_used": float(v_info.get("v_max_mean", np.nan)),
            },
        }

    def run_simulation_return_states(self):
        self.model.eval()
        # 初始化状态 (调用 runner 的方法)
        self._init_model_states(batch_size=self.cfg.simulation.batch_size)
        
        with torch.no_grad():
            # 获取输入 (使用 runner 里的 test_loader)
            if self.test_loader:
                inputs = next(iter(self.test_loader))
                if isinstance(inputs, (list, tuple)):
                    inputs = inputs[0]
            else:
                # Fallback: 手动造数据 (参考 runner.run_simulation)
                T = self.cfg.simulation.T
                N_in = self.cfg.network.num_input_neuron
                inputs = torch.randn(1, T, N_in).abs() # Poisson rate > 0
            
            inputs = inputs.to(self.device)
            
            # 执行前向传播
            output, states = self.model(inputs)

            self.run_metrics(states, is_eval=True, epoch_id=None)

        return output, states

    def run_training(self):
        """Training Mode: 标准训练循环"""
        logger.info("🚀 Starting TRAINING mode")
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.cfg.train.learning_rate
        )
        
        epochs = self.cfg.train.epochs
        self.model.train()
        self.model.to(self.device)

        self._init_model_states(batch_size=self.cfg.train.batch_size)
        
        for epoch in range(epochs):

            # Reset network state
            functional.reset_net(self.model, batch_size=self.cfg.train.batch_size, device=self.device)
            # 与评估阶段保持一致，重置膜电位到合理范围，避免状态遗留导致数值问题
            neuron_module = self.model.flybrain.brain.neuron
            init.uniform_v_(
                neuron_module,
                # set_reset_value=True,
                # batch_size=self.config.batch_size,
            )

            total_loss = 0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            
            for batch in pbar:
                # Reset network state
                functional.reset_net(self.model, batch_size=self.cfg.train.batch_size, device=self.device)
                # 与评估阶段保持一致，重置膜电位到合理范围，避免状态遗留导致数值问题
                neuron_module = self.model.flybrain.brain.neuron
                init.uniform_v_(
                    neuron_module,
                    # set_reset_value=True,
                    # batch_size=self.config.batch_size,
                )
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                    # targets = batch[1] # 如果有
                else:
                    inputs = batch
                    
                inputs = inputs.to(self.device)
                optimizer.zero_grad()
                
                output, states = self.model(inputs)
                
                loss, loss_dict = self.model.compute_loss(output, states)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
            
            logger.info(f"Epoch {epoch} finished. Avg Loss: {total_loss/len(self.train_loader):.4f}")
            
            # 保存 Checkpoint
            if epoch % 10 == 0:
                self._save_checkpoint(epoch)

    def run_visualize(self, states, is_eval, epoch_id):
        base_dir = Path(self.cfg.out_dir)
        if is_eval:
            epoch_figure_dir = base_dir / f"epoch_eval_{epoch_id}"
        else:
            epoch_figure_dir = base_dir / f"epoch_{epoch_id}"
        epoch_figure_dir.mkdir(parents=True, exist_ok=True)

        #here
        # 1. 使用新函数按类别选择神经元 
        selected_indices, _ = select_neurons_by_class(
            self.model.connectome_data, 
            num_per_class=4
        )

        #breakpoint()
        data = prepare_data_from_dict(states, self.model.params, self.cfg, self.model.connectome_data)
        visualize_results(**data, selected_neurons=selected_indices, epoch_figure_dir=epoch_figure_dir)

        # 3. 计算并保存 multi-scale fano factor
        #self._compute_and_save_multiscale_ff(states, epoch_figure_dir)

    def run_metrics(self, states, is_eval, epoch_id):
        base_dir = Path(self.cfg.out_dir)
        if is_eval:
            epoch_metrics_dir = base_dir / f"epoch_eval_{epoch_id}"
        else:
            epoch_metrics_dir = base_dir / f"epoch_{epoch_id}"
        epoch_metrics_dir.mkdir(parents=True, exist_ok=True)

        metrics_payload = self.compute_metrics(states)
        metrics = metrics_payload.get("metrics", {})
        objective_values = metrics_payload.get("objective_values")
        if objective_values is not None:
            metrics["objective_values"] = objective_values

        with open(epoch_metrics_dir / "metrics.txt", "w") as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")


    def _save_results(self, states, name):
        base_dir = Path(self.cfg.out_dir)
        epoch_figure_dir = base_dir / f"{name}"
        epoch_figure_dir.mkdir(parents=True, exist_ok=True)

        data = prepare_data_from_dict(states, self.model.params, self.cfg, self.model.connectome_data)
        save_states_full(**data, epoch_figure_dir=epoch_figure_dir)
        # 保存states

    def _save_checkpoint(self, epoch):
        pass

# 每一个e神经元的输入ie权重比例
# jidianliu分布
# grid search log scale，少算几个
# dt的影响
# voltage scale
# GLIF各个参数分布情况
