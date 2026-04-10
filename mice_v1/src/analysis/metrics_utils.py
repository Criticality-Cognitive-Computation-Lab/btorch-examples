import torch
import numpy as np

def get_cv(spikes):
    """Calculate Coefficient of Variation of ISI.
    Transfers data to CPU to avoid GPU OOM.
    """
    if spikes.numel() == 0:
        return 0.0
        
    # Move to CPU immediately
    spikes = spikes.detach().cpu()

    # spikes shape: [T, B, N] or [T, N]
    if spikes.ndim == 3:
        # Flatten Batch and Neuron dimensions: [T, B*N]
        spikes = spikes.reshape(spikes.shape[0], -1) 
    
    n_neurons = spikes.shape[1]
    cvs = []
    
    # Use vectorized operations where possible, or loop efficiently
    # Note: Looping is slow in Python, but for analysis it's often acceptable.
    # We stick to the loop for safety with variable ISI lengths.
    for i in range(n_neurons):
        spike_times = torch.where(spikes[:, i])[0].float()
        if len(spike_times) > 2:
            isis = torch.diff(spike_times)
            mean_isi = torch.mean(isis)
            std_isi = torch.std(isis)
            if mean_isi > 1e-6: # Avoid division by zero
                cvs.append((std_isi / mean_isi).item())
    
    if len(cvs) == 0:
        return 0.0
    return np.mean(cvs)

def calculate_balance_metrics(psc_e, psc_i, min_lag_ms=1.0, max_lag_ms=20, dt=1.0):
    """
    计算 E/I 平衡指标：幅度比率 + 互相关延迟。
    强制在 CPU 上计算以避免 GPU OOM。
    
    psc_e: [T, B, N] (正值)
    psc_i: [T, B, N] (负值)
    """
    # ==========================================
    # 1. 强制转移到 CPU，释放 GPU 压力
    # ==========================================
    # 注意：这里我们使用 float32 以节省内存，如果精度要求极高可用 float64
    I_exc = psc_e.detach().cpu().float()
    I_inh = psc_i.detach().cpu().float()
    
    # 转换为 [T, B*N] 进行统一统计
    if I_exc.ndim == 3:
        I_exc = I_exc.reshape(I_exc.shape[0], -1)
    if I_inh.ndim == 3:
        I_inh = I_inh.reshape(I_inh.shape[0], -1)
    
    T = I_exc.shape[0]
    min_lag_steps = max(1, int(min_lag_ms / dt))
    max_lag_steps = int(max_lag_ms / dt)
    
    # ==========================================
    # 2. 幅度平衡 (Magnitude Balance)
    # ==========================================
    # 目标：总兴奋 ≈ 总抑制 (绝对值)
    mean_exc = I_exc.mean()
    mean_inh = I_inh.mean() # 通常是负值
    
    total_pos = mean_exc
    total_neg = abs(mean_inh)
    
    if total_neg < 1e-9:
        loss_magnitude = 10.0 # 惩罚无抑制
    else:
        loss_magnitude = abs(total_pos / total_neg - 1.0).item()
        
    # ==========================================
    # 3. 时间相关性 (Correlation & Lag)
    # ==========================================
    # 筛选活跃神经元
    # 全局标准差筛选
    std_exc = I_exc.std(dim=0)
    # I_inh 是负值，取反变为正值方便计算（并不影响 std）
    I_inh_inv = -I_inh 
    std_inh = I_inh_inv.std(dim=0)
    
    valid_mask = (std_exc > 1e-6) & (std_inh > 1e-6)
    if valid_mask.sum() == 0:
        return loss_magnitude, 1.0, 0.0 # 无效相关性
        
    # 仅保留有效神经元的数据
    # 注意：此时数据已在 CPU，内存充足
    exc_valid = I_exc[:, valid_mask]
    inh_valid = I_inh_inv[:, valid_mask]
    
    T_valid = T - max_lag_steps
    
    corrs_per_lag = []
    epsilon = 1e-8
    
    # 预先切片并计算 Exc 部分（因为它不随 lag 滑动，只是长度截断）
    # exc_sub 对应时间窗 [0 : T-max_lag]
    exc_slice = exc_valid[:T_valid, :]
    
    # [关键修复]: 在切片内部重新计算中心化和标准差，防止局部平直导致的 Correlation 爆炸
    exc_sub_centered = exc_slice - exc_slice.mean(dim=0, keepdim=True)
    exc_sub_std = exc_sub_centered.std(dim=0)
    
    # 循环计算不同延迟
    for lag in range(min_lag_steps, max_lag_steps + 1):
        # inh_sub 对应时间窗 [lag : lag + T-max_lag]
        inh_slice = inh_valid[lag : lag + T_valid, :]
        
        # 局部中心化
        inh_sub_centered = inh_slice - inh_slice.mean(dim=0, keepdim=True)
        inh_sub_std = inh_sub_centered.std(dim=0)
        
        # 计算协方差 (Covariance)
        cov = (exc_sub_centered * inh_sub_centered).mean(dim=0)
        
        # 安全除法：检查局部标准差是否过小
        safe_mask = (exc_sub_std > epsilon) & (inh_sub_std > epsilon)
        
        corr = torch.zeros_like(cov)
        corr[safe_mask] = cov[safe_mask] / (exc_sub_std[safe_mask] * inh_sub_std[safe_mask] + epsilon)
        
        corrs_per_lag.append(corr)
        
    # 堆叠结果 [n_lags, n_valid_neurons]
    corrs_matrix = torch.stack(corrs_per_lag)
    
    # 对每个神经元找最大相关性及其对应的延迟索引
    max_corrs, best_lag_indices = torch.max(corrs_matrix, dim=0)
    
    # 计算平均最大相关性 (Loss = 1 - Mean_Max_Corr)
    loss_corr = 1.0 - max_corrs.mean().item()
    
    # 计算平均延迟
    best_lags = best_lag_indices + min_lag_steps
    avg_lag_ms = best_lags.float().mean().item() * dt
    
    return loss_magnitude, loss_corr, avg_lag_ms