"""
GLIF突触模型实现
支持不同类型的突触连接使用不同的alpha函数时间常数
基于Arkhipov et al., 2018的L4模型数据
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, Tuple


class GLIFSynapse(nn.Module):
    """
    GLIF突触模型，支持连接特定的时间常数
    
    突触电流动力学：
    I_syn(t) = (e * W_GLIF / tau_syn) * t * exp(-t / tau_syn)
    
    其中：
    - I_syn: 突触后电流
    - W_GLIF: 连接权重
    - tau_syn: 突触时间常数（根据连接类型不同）
    - e: 自然对数的底数
    """
    
    def __init__(
        self,
        n_neuron: int,
        tau_syn_matrix: torch.Tensor,
        linear: nn.Module,
        step_mode: str = "s",
        backend: str = "torch",
    ):
        """
        初始化GLIF突触模型
        
        Args:
            n_neuron: 神经元数量
            tau_syn_matrix: 突触时间常数矩阵 (n_neurons, n_neurons)
            linear: 线性连接模块
            step_mode: 步进模式
            backend: 后端
        """
        super().__init__()
        
        self.n_neuron = n_neuron
        self.linear = linear
        self.step_mode = step_mode
        self.backend = backend
        
        # 注册突触时间常数矩阵
        self.register_buffer("tau_syn_matrix", torch.as_tensor(tau_syn_matrix, dtype=torch.float32))
        
        # 计算衰减常数矩阵
        self.register_buffer("syn_decay_matrix", torch.exp(-1.0 / self.tau_syn_matrix))
        
        # 初始化突触状态
        self.register_memory("psc", 0.0, n_neuron)  # 突触后电流
        self.register_memory("h", 0.0, n_neuron)    # 辅助变量
        
    def conductance_charge(self):
        """计算突触后电流"""
        # 使用矩阵形式的衰减
        self.psc = self.syn_decay_matrix @ self.psc + self.syn_decay_matrix @ self.h
        return self.psc
    
    def adaptation_charge(self, z: torch.Tensor):
        """更新突触状态"""
        wz = self.linear(z)  # 线性变换
        
        # 使用矩阵形式更新辅助变量
        # h = syn_decay * h + (e / tau_syn) * wz
        e_over_tau = torch.e / self.tau_syn_matrix
        self.h = self.syn_decay_matrix @ self.h + e_over_tau @ wz
    
    def register_memory(self, name: str, value: Union[float, torch.Tensor], n_neuron: int):
        """注册内存变量"""
        if isinstance(value, (int, float)):
            value = torch.full((n_neuron,), value, dtype=torch.float32)
        self.register_buffer(name, value, persistent=False)


class GLIFSynapseApprox(nn.Module):
    """
    GLIF突触模型的近似实现
    使用每个神经元的平均时间常数，适用于大规模网络
    """
    
    def __init__(
        self,
        n_neuron: int,
        tau_syn_array: torch.Tensor,
        linear: nn.Module,
        step_mode: str = "s",
        backend: str = "torch",
    ):
        """
        初始化近似GLIF突触模型
        
        Args:
            n_neuron: 神经元数量
            tau_syn_array: 每个神经元的平均时间常数 (n_neurons,)
            linear: 线性连接模块
            step_mode: 步进模式
            backend: 后端
        """
        super().__init__()
        
        self.n_neuron = n_neuron
        self.linear = linear
        self.step_mode = step_mode
        self.backend = backend
        
        # 注册时间常数数组
        self.register_buffer("tau_syn", torch.as_tensor(tau_syn_array, dtype=torch.float32))
        
        # 计算衰减常数
        self.register_buffer("syn_decay", torch.exp(-1.0 / self.tau_syn))
        
        # 初始化突触状态
        self.register_memory("psc", 0.0, n_neuron)
        self.register_memory("h", 0.0, n_neuron)
        
    def conductance_charge(self):
        """计算突触后电流"""
        self.psc = self.syn_decay * self.psc + self.syn_decay * self.h
        return self.psc
    
    def adaptation_charge(self, z: torch.Tensor):
        """更新突触状态"""
        wz = self.linear(z)
        self.h = self.syn_decay * self.h + torch.e / self.tau_syn * wz
    
    def register_memory(self, name: str, value: Union[float, torch.Tensor], n_neuron: int):
        """注册内存变量"""
        if isinstance(value, (int, float)):
            value = torch.full((n_neuron,), value, dtype=torch.float32)
        self.register_buffer(name, value, persistent=False)


def create_glif_synapse_times(
    n_neurons: int,
    excitatory_mask: np.ndarray,
    tau_syn_ee: float = 5.5,
    tau_syn_ie: float = 8.5,
    tau_syn_ei: float = 2.8,
    tau_syn_ii: float = 5.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建GLIF突触时间常数矩阵和数组
    
    Args:
        n_neurons: 神经元数量
        excitatory_mask: 兴奋性神经元掩码
        tau_syn_ee: E→E突触时间常数
        tau_syn_ie: I→E突触时间常数
        tau_syn_ei: E→I突触时间常数
        tau_syn_ii: I→I突触时间常数
        
    Returns:
        tau_syn_matrix: 完整的突触时间常数矩阵
        tau_syn_array: 每个神经元的平均时间常数
    """
    inhibitory_mask = ~excitatory_mask
    
    # 创建突触时间常数矩阵
    tau_syn_matrix = np.zeros((n_neurons, n_neurons), dtype=np.float32)
    
    for i in range(n_neurons):
        for j in range(n_neurons):
            if excitatory_mask[i] and excitatory_mask[j]:
                tau_syn_matrix[i, j] = tau_syn_ee  # E→E
            elif excitatory_mask[i] and inhibitory_mask[j]:
                tau_syn_matrix[i, j] = tau_syn_ei  # E→I
            elif inhibitory_mask[i] and excitatory_mask[j]:
                tau_syn_matrix[i, j] = tau_syn_ie  # I→E
            elif inhibitory_mask[i] and inhibitory_mask[j]:
                tau_syn_matrix[i, j] = tau_syn_ii  # I→I
    
    # 计算每个神经元的平均时间常数
    tau_syn_array = np.mean(tau_syn_matrix, axis=0)
    
    return tau_syn_matrix, tau_syn_array

def create_double_exponential_synapse_times(
    n_neurons: int,
    excitatory_mask: np.ndarray,
    tau_rise_ee: float = 2.8,
    tau_rise_ie: float = 5.5,
    tau_rise_ei: float = 8.5,
    tau_rise_ii: float = 2.8,
    tau_decay_ee: float = 5.5,
    tau_decay_ie: float = 8.5,
    tau_decay_ei: float = 2.8,
    tau_decay_ii: float = 5.5,
    latency_ee: float = 0.0,
    latency_ie: float = 0.0,
    latency_ei: float = 0.0,
    latency_ii: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    创建双指数突触时间常数矩阵和数组
    """
    tau_rise_matrix = np.zeros((n_neurons, n_neurons), dtype=np.float32)
    tau_decay_matrix = np.zeros((n_neurons, n_neurons), dtype=np.float32)
    latency_matrix = np.zeros((n_neurons, n_neurons), dtype=np.float32)

    inhibitory_mask = ~excitatory_mask
    
    tau_rise_matrix[excitatory_mask][:, excitatory_mask] = tau_rise_ee
    tau_decay_matrix[excitatory_mask][:, excitatory_mask] = tau_decay_ee
    latency_matrix[excitatory_mask][:, excitatory_mask] = latency_ee
    tau_rise_matrix[inhibitory_mask][:, excitatory_mask] = tau_rise_ie
    tau_decay_matrix[inhibitory_mask][:, excitatory_mask] = tau_decay_ie
    latency_matrix[inhibitory_mask][:, excitatory_mask] = latency_ie
    tau_rise_matrix[excitatory_mask][:, inhibitory_mask] = tau_rise_ei
    tau_decay_matrix[excitatory_mask][:, inhibitory_mask] = tau_decay_ei
    latency_matrix[excitatory_mask][:, inhibitory_mask] = latency_ei
    tau_rise_matrix[inhibitory_mask][:, inhibitory_mask] = tau_rise_ii
    tau_decay_matrix[inhibitory_mask][:, inhibitory_mask] = tau_decay_ii
    latency_matrix[inhibitory_mask][:, inhibitory_mask] = latency_ii

    
    tau_rise_array = np.mean(tau_rise_matrix, axis=0)
    tau_decay_array = np.mean(tau_decay_matrix, axis=0)
    latency_array = np.mean(latency_matrix, axis=0)
    return tau_rise_matrix, tau_decay_matrix, latency_matrix


def analyze_synapse_times(tau_syn_matrix: np.ndarray, excitatory_mask: np.ndarray) -> dict:
    """
    分析突触时间常数分布
    
    Args:
        tau_syn_matrix: 突触时间常数矩阵
        excitatory_mask: 兴奋性神经元掩码
        
    Returns:
        包含统计信息的字典
    """
    inhibitory_mask = ~excitatory_mask
    
    # 提取不同类型的连接
    ee_connections = tau_syn_matrix[excitatory_mask][:, excitatory_mask]
    ie_connections = tau_syn_matrix[inhibitory_mask][:, excitatory_mask]
    ei_connections = tau_syn_matrix[excitatory_mask][:, inhibitory_mask]
    ii_connections = tau_syn_matrix[inhibitory_mask][:, inhibitory_mask]
    
    analysis = {
        'ee_mean': np.mean(ee_connections),
        'ee_std': np.std(ee_connections),
        'ie_mean': np.mean(ie_connections),
        'ie_std': np.std(ie_connections),
        'ei_mean': np.mean(ei_connections),
        'ei_std': np.std(ei_connections),
        'ii_mean': np.mean(ii_connections),
        'ii_std': np.std(ii_connections),
        'overall_mean': np.mean(tau_syn_matrix),
        'overall_std': np.std(tau_syn_matrix),
        'n_ee': np.sum(ee_connections > 0),
        'n_ie': np.sum(ie_connections > 0),
        'n_ei': np.sum(ei_connections > 0),
        'n_ii': np.sum(ii_connections > 0),
    }
    
    return analysis


if __name__ == "__main__":
    # 测试GLIF突触模型
    print("GLIF突触模型测试")
    
    n_neurons = 100
    excitatory_mask = np.zeros(n_neurons, dtype=bool)
    excitatory_mask[:80] = True  # 前80个是兴奋性
    
    tau_syn_matrix, tau_syn_array = create_glif_synapse_times(
        n_neurons, excitatory_mask
    )
    
    analysis = analyze_synapse_times(tau_syn_matrix, excitatory_mask)
    
    print(f"突触时间常数分析:")
    print(f"  E→E: {analysis['ee_mean']:.1f} ± {analysis['ee_std']:.1f} ms (n={analysis['n_ee']})")
    print(f"  I→E: {analysis['ie_mean']:.1f} ± {analysis['ie_std']:.1f} ms (n={analysis['n_ie']})")
    print(f"  E→I: {analysis['ei_mean']:.1f} ± {analysis['ei_std']:.1f} ms (n={analysis['n_ei']})")
    print(f"  I→I: {analysis['ii_mean']:.1f} ± {analysis['ii_std']:.1f} ms (n={analysis['n_ii']})")
    print(f"  总体: {analysis['overall_mean']:.1f} ± {analysis['overall_std']:.1f} ms")
