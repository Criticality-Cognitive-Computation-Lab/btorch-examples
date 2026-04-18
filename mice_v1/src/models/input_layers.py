from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Optional
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
# from brevitas.nn import QuantLinear
# from brevitas.quant import Int4WeightPerTensorFloatDecoupled as WeightQuantTensor

#from .serialisation import save_dict_array, save_weight
#from .ultis import build_rec_layer_from_conn#, load_microns_connectome
#from .sim import load_and_preprocess_mice
from btorch.models.constrain import HasConstraint

import ipdb

# Import rate regularization functions
# from .rate_regularization import (
#     calculate_laminar_rate_loss,
#     set_up_firing_rate_regularization,
# )


class BaseGenericInputLayer(nn.Module, HasConstraint):
    """
    通用输入层基类。
    负责处理共用的权重初始化、参数注册、约束逻辑和日志记录。
    具体的输入注入方式（DC vs Poisson）由子类实现。
    """
    def __init__(self, 
                 num_input_neurons: int, 
                 n_total_neurons: int, 
                 input_indices: np.ndarray | list, 
                 device: str, 
                 dtype: torch.dtype,
                 voltage_scale: torch.Tensor | None = None,
                 weight_init_dist: str = "ones",
                 weight_init_params: dict | None = None,
                 seed: int | None = None,
                 ):
        super().__init__()
        self.num_input_neurons = num_input_neurons
        self.n_total_neurons = n_total_neurons
        self.device = device
        self.dtype = dtype
        self.weight_init_dist = weight_init_dist
        self.weight_init_params = weight_init_params or {}
        
        # 1. 设备检查
        if device.startswith('cuda') and not torch.cuda.is_available():
            print(f"⚠️  警告: 指定设备 {device} 不可用，切换到CPU")
            self.device = 'cpu'
            
        # 2. 注册索引 Buffer
        self.register_buffer("input_indices", torch.tensor(input_indices, dtype=torch.long, device=self.device))
        
        # 3. 初始化权重 (Logic Core)
        logger = logging.getLogger(__name__)

        # 3.1 基础初始化
        scaling_weights = self._initialize_weights(num_input_neurons, seed)

        # 3.3 日志记录 (Before Scale)
        self._log_stats(logger, "init(before_vscale)", scaling_weights)

        # 4. 处理 Voltage Scale (不可训练参数)
        if voltage_scale is not None:
            if not isinstance(voltage_scale, torch.Tensor):
                voltage_scale = torch.as_tensor(voltage_scale, dtype=dtype, device=device)
            assert voltage_scale.ndim == 1 and voltage_scale.numel() >= n_total_neurons
            
            vs_inputs = voltage_scale[self.input_indices]
            # 防止除零
            vs_inputs = torch.where(vs_inputs == 0, torch.ones_like(vs_inputs), vs_inputs)
            self.register_buffer("voltage_scale_inputs", vs_inputs)
            
            # 预除以 voltage_scale
            scaling_weights = scaling_weights / vs_inputs
        else:
            self.register_buffer("voltage_scale_inputs", torch.ones(len(input_indices), dtype=dtype, device=device))
        
        # 3.4 日志记录 (After Scale)
        self._log_stats(logger, "init(after_vscale)", scaling_weights)

        # 5. 注册为可训练参数
        self.register_parameter("scaling_weights", nn.Parameter(scaling_weights))
        self.scaling_weights.register_hook(self._grad_hook)

        # Debug controls for injected current diagnostics.
        self._debug_logged_once = False
        self._debug_enabled = True

    def _initialize_weights(self, num_neurons: int, seed: int | None = None) -> torch.Tensor:
        """生成基础权重张量"""
        if seed is not None:
            torch.manual_seed(seed)
        
        params = self.weight_init_params
        if self.weight_init_dist == "identical":
            return torch.full((num_neurons,), params.get('value', 1.0), dtype=self.dtype, device=self.device)
        elif self.weight_init_dist == "uniform":
            w = torch.empty(num_neurons, dtype=self.dtype, device=self.device)
            return w.uniform_(params.get('low', 0.0), params.get('high', 1.0))
        elif self.weight_init_dist == "normal":
            return torch.normal(params.get('mean', 0.0), params.get('std', 1.0), 
                              size=(num_neurons,), dtype=self.dtype, device=self.device)
        elif self.weight_init_dist == "lognormal":
            normal_w = torch.normal(params.get('mean', 0.0), params.get('std', 1.0), 
                                  size=(num_neurons,), dtype=self.dtype, device=self.device)
            return torch.exp(normal_w)
        else:
            raise ValueError(f"Unknown distribution: {self.weight_init_dist}")

    def _grad_hook(self, g):
        if g is None: return g
        g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        return g.clamp_(-1e3, 1e3)

    def _log_stats(self, logger, prefix, weights):
        """统计日志辅助函数"""
        try:
            name = self.__class__.__name__
            if self.group_labels:
                for g in sorted(set(self.group_labels)):
                    mask = torch.tensor([l == g for l in self.group_labels], device=self.device)
                    if mask.any():
                        w = weights[mask]
                        logger.info(f"[{name}] {prefix} group={g} mean={w.mean():.4e} std={w.std():.4e}")
            else:
                logger.info(f"[{name}] {prefix} ALL mean={weights.mean():.4e} std={weights.std():.4e}")
        except Exception:
            pass

    def constrain(self, w_min: float = 1e-6, w_max: float = 5.0):
        with torch.no_grad():
            w = self.scaling_weights.data
            w = torch.nan_to_num(w, nan=1.0, posinf=w_max, neginf=w_min)
            w.clamp_(min=w_min, max=w_max)
            self.scaling_weights.data.copy_(w)

    def log_injected_current_debug(
        self,
        *,
        current_input: torch.Tensor,
        scaled_current: torch.Tensor,
    ) -> None:
        """Log one-shot diagnostics for real injected current strength.

        This is the exact place where external input is transformed and then
        injected to selected neurons.
        """
        if not self._debug_enabled or self._debug_logged_once:
            return

        logger = logging.getLogger(__name__)
        try:
            ci = current_input.detach()
            sc = scaled_current.detach()
            sw = self.scaling_weights.detach()

            def _stats(t: torch.Tensor):
                return {
                    "shape": tuple(t.shape),
                    "min": float(t.min().item()),
                    "max": float(t.max().item()),
                    "mean": float(t.mean().item()),
                    "std": float(t.std().item()),
                }

            logger.info("[InputDebug] input_indices=%d / total_neurons=%d", int(self.num_input_neurons), int(self.n_total_neurons))
            logger.info("[InputDebug] current_input stats: %s", _stats(ci))
            logger.info("[InputDebug] scaling_weights stats: %s", _stats(sw))
            logger.info("[InputDebug] scaled_current stats: %s", _stats(sc))

            if hasattr(self, "voltage_scale_inputs") and self.voltage_scale_inputs is not None:
                vs = self.voltage_scale_inputs.detach()
                logger.info("[InputDebug] voltage_scale_inputs stats: %s", _stats(vs))

            # Per-neuron averaged injected current across batch/time:
            # scaled_current shape [T, B, N_in]
            inj_mean_per_neuron = sc.mean(dim=(0, 1))

            # Optional ratio check against I_thr if provided by model.
            i_thr_inputs = getattr(self, "debug_i_thr_inputs", None)
            if i_thr_inputs is not None:
                i_thr = i_thr_inputs.detach()
                safe_thr = torch.where(i_thr.abs() < 1e-12, torch.full_like(i_thr, 1e-12), i_thr)
                ratio = inj_mean_per_neuron / safe_thr
                logger.info("[InputDebug] I_thr(inputs) stats: %s", _stats(i_thr))
                logger.info("[InputDebug] inj_mean/I_thr stats: %s", _stats(ratio))

                k = int(min(10, inj_mean_per_neuron.numel()))
                logger.info(
                    "[InputDebug] first_%d mean_inj=%s",
                    k,
                    [float(x) for x in inj_mean_per_neuron[:k].cpu()],
                )
                logger.info(
                    "[InputDebug] first_%d I_thr=%s",
                    k,
                    [float(x) for x in i_thr[:k].cpu()],
                )
                logger.info(
                    "[InputDebug] first_%d ratio=%s",
                    k,
                    [float(x) for x in ratio[:k].cpu()],
                )
            else:
                logger.info("[InputDebug] debug_i_thr_inputs not attached; skip inj/I_thr ratio check.")

        except Exception as exc:
            logger.warning("[InputDebug] failed to log injected current diagnostics: %s", exc)
        finally:
            self._debug_logged_once = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DCInputLayer(BaseGenericInputLayer):
    """直流输入层：忽略输入值的内容，只关注形状，注入恒定电流。"""
    
    def __init__(self, dc_current: float = 0.5, voltage_scale: torch.Tensor | None = None, **kwargs):
        super().__init__(voltage_scale=voltage_scale, **kwargs)
        self.voltage_scale = voltage_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [T, batch_size, input_dim]
        returns: [T, batch_size, n_total_neurons]
        """
        # ========================================================
        # 【新增】智能形状适配逻辑
        # ========================================================
        input_dim = x.shape[-1]
        
        if input_dim == 1:
            # Case 1: 标量输入 [T, B, 1]。自动广播到所有选中的 input_indices
            # 这是做参数优化最推荐的格式，省显存
            current_input = x 
        
        elif input_dim == self.n_total_neurons:
            # Case 2: 全量输入 [T, B, 1050]。用户希望屏蔽掉非输入神经元
            # 使用 input_indices 进行切片 (Slicing/Masking)
            # x[..., indices] 会取出 [T, B, 797]
            current_input = x[..., self.input_indices]
            
        elif input_dim == self.num_input_neurons:
            # Case 3: 维度已匹配 [T, B, 797]。直接使用
            current_input = x
            
        else:
            raise RuntimeError(
                f"Input dimension mismatch! Expected 1, {self.num_input_neurons} (input subset), "
                f"or {self.n_total_neurons} (total). Got {input_dim}."
            )

        # ========================================================
        # 计算注入电流
        # ========================================================
        # current_input: [..., 1] or [..., 797]
        # scaling_weights: [797] -> unsqueeze -> [1, 1, 797]
        
        # 此时广播机制会正常工作
        scaled_current = current_input * self.scaling_weights.unsqueeze(0).unsqueeze(0)
        self.log_injected_current_debug(current_input=current_input, scaled_current=scaled_current)

        T, batch_size = x.shape[0], x.shape[1]
        
        # 映射到全网络索引 (Sparse -> Dense)
        inp = torch.zeros((T, batch_size, self.n_total_neurons), device=self.device, dtype=self.dtype)
        inp[..., self.input_indices] = scaled_current

        inp = inp.permute(1, 0, 2)
        
        return inp

# PoissonInputLayer 也做同样的修改逻辑，如果你的任务涉及泊松
class PoissonInputLayer(BaseGenericInputLayer):
    # ... __init__ 不变 ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ... 这里的形状适配逻辑与上方 DCInputLayer 完全一致，直接复制上面的 if-elif-else 块 ...
        
        input_dim = x.shape[-1]
        if input_dim == 1:
            current_input = x 
        elif input_dim == self.n_total_neurons:
            current_input = x[..., self.input_indices]
        elif input_dim == self.num_input_neurons:
            current_input = x
        else:
             raise RuntimeError(f"Input mismatch: Got {input_dim}, expected 1, {self.num_input_neurons} or {self.n_total_neurons}")

        scaled_x = current_input * self.scaling_weights.unsqueeze(0).unsqueeze(0)
        self.log_injected_current_debug(current_input=current_input, scaled_current=scaled_x)

        inp = torch.zeros((*scaled_x.shape[:-1], self.n_total_neurons), device=self.device, dtype=self.dtype)
        inp[..., self.input_indices] = scaled_x
        
        inp = inp.permute(1, 0, 2)
        return inp

def build_input_adapter(
    config,
    voltage_scale: torch.Tensor,
    input_indices: np.ndarray | list,
    device: str, 
    dtype: torch.dtype,
) -> nn.Module:

    input_cfg = getattr(config.network, 'input_layer', {})
    weight_init_params = {
        'dist': getattr(input_cfg, 'type', 'ones'),
        'params': {}
    }

    # 3. 根据不同的分布类型加载参数
    if weight_init_params['dist'] == 'lognormal':
        weight_init_params['params'].update({
            'mean': getattr(input_cfg, 'mean', 0.0),
            'std': getattr(input_cfg, 'std', 1.0)
        })
    elif weight_init_params['dist'] == 'normal':
        weight_init_params['params'].update({
            'mean': getattr(input_cfg, 'mean', 0.0),
            'std': getattr(input_cfg, 'std', 1.0)
        })
    elif weight_init_params['dist'] == 'uniform':
        weight_init_params['params'].update({
            'low': getattr(input_cfg, 'low', 0.0),
            'high': getattr(input_cfg, 'high', 1.0)
        })

    common_kwargs = {
        'num_input_neurons': int(len(input_indices)),
        'n_total_neurons': config.network.n_neuron,
        'input_indices': input_indices,
        'device': device,
        'dtype': dtype,
        'voltage_scale': voltage_scale,
        'weight_init_dist': weight_init_params['dist'],
        'weight_init_params': weight_init_params['params'],
        'seed': getattr(input_cfg, 'seed', None)  # 可选：添加随机种子
    }
    # 2. 实例化
    if config.dataset.type == 'dc' or config.dataset.type == 'noisydc' or config.dataset.type == 'noisydc_ou':
        print(f"[InputFactory] Creating DCInputLayer")
        return DCInputLayer(**common_kwargs)
    elif config.dataset.type == 'poisson':
        print(f"[InputFactory] Creating PoissonInputLayer")
        return PoissonInputLayer(**common_kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {config.dataset.type}")