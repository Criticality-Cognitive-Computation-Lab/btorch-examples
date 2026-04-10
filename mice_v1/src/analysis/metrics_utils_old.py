# src/analysis/metrics_utils.py
import torch
import numpy as np

def get_cv(spikes):
    """Calculate Coefficient of Variation of ISI."""
    if spikes.numel() == 0:
        return 0.0
        
    # spikes shape: [T, B, N] or [T, N]
    if spikes.ndim == 3:
        # 如果有 Batch 维度，将其展平或取平均，这里简单处理为对所有神经元计算
        # 更好的做法可能是对 Batch 取平均，这里简化为 reshape
        spikes = spikes.reshape(spikes.shape[0], -1) # [T, B*N]
    
    n_neurons = spikes.shape[1]
    cvs = []
    
    spikes_cpu = spikes.detach().cpu()
    
    for i in range(n_neurons):
        spike_times = torch.where(spikes_cpu[:, i])[0].float()
        if len(spike_times) > 2:
            isis = torch.diff(spike_times)
            mean_isi = torch.mean(isis)
            std_isi = torch.std(isis)
            if mean_isi > 0:
                cvs.append((std_isi / mean_isi).item())
    
    if len(cvs) == 0:
        return 0.0
    return np.mean(cvs)

def calculate_balance_metrics(psc_e, psc_i, min_lag_ms=1.0, max_lag_ms=20, dt=1.0):
    """
    计算 E/I 平衡指标：幅度比率 + 互相关延迟。
    psc_e: [T, B, N] (正值)
    psc_i: [T, B, N] (负值)
    """
    # 转换为 [T, B*N] 进行统一统计
    I_exc = psc_e.reshape(psc_e.shape[0], -1)
    I_inh = psc_i.reshape(psc_i.shape[0], -1)
    
    T = I_exc.shape[0]
    min_lag_steps = max(1, int(min_lag_ms / dt))
    max_lag_steps = int(max_lag_ms / dt)
    
    # 1. 幅度平衡 (Magnitude Balance)
    # 目标：总兴奋 ≈ 总抑制 (绝对值)
    mean_exc = I_exc.mean()
    mean_inh = I_inh.mean() # 通常是负值
    
    total_pos = mean_exc
    total_neg = abs(mean_inh)
    
    if total_neg < 1e-9:
        loss_magnitude = 10.0 # 惩罚无抑制
    else:
        loss_magnitude = abs(total_pos / total_neg - 1.0).item()
        
    # 2. 时间相关性 (Correlation & Lag)
    # 中心化
    exc_centered = I_exc - I_exc.mean(dim=0, keepdim=True)
    inh_centered = (-I_inh) - (-I_inh).mean(dim=0, keepdim=True) # 取 I 的反相(变正)来算相关性
    
    std_exc = exc_centered.std(dim=0)
    std_inh = inh_centered.std(dim=0)
    
    valid_mask = (std_exc > 1e-6) & (std_inh > 1e-6)
    if valid_mask.sum() == 0:
        return loss_magnitude, 1.0, 0.0 # 无效相关性
        
    exc_valid = exc_centered[:, valid_mask]
    inh_valid = inh_centered[:, valid_mask]
    
    # 计算互相关寻找最佳延迟
    T_valid = T - max_lag_steps
    exc_sub = exc_valid[:T_valid, :]
    
    corrs_per_lag = []
    epsilon = 1e-8
    # 预先计算 exc_sub 的局部标准差和中心化
    # 必须在切片上重新计算 mean 和 std，不能用全局的！
    exc_slice = I_exc[:T_valid, valid_mask] # 使用原始数据 I_exc
    exc_sub_centered = exc_slice - exc_slice.mean(dim=0, keepdim=True)
    exc_sub_std = exc_sub_centered.std(dim=0)
    for lag in range(min_lag_steps, max_lag_steps + 1):
        # 同样处理 inh_sub
        inh_slice = (-I_inh)[lag : lag + T_valid, valid_mask] # 使用原始数据 I_inh 取反
        inh_sub_centered = inh_slice - inh_slice.mean(dim=0, keepdim=True)
        inh_sub_std = inh_sub_centered.std(dim=0)
        # Pearson Correlation
        cov = (exc_sub_centered * inh_sub_centered).mean(dim=0)
        # 增加局部 std 的安全检查
        safe_mask = (exc_sub_std > epsilon) & (inh_sub_std > epsilon)
        corr = torch.zeros_like(cov)
        # 只有当局部都有波动时才计算相关性，否则设为 0
        corr[safe_mask] = cov[safe_mask] / (exc_sub_std[safe_mask] * inh_sub_std[safe_mask] + epsilon)
        
        corrs_per_lag.append(corr)
        
    corrs_matrix = torch.stack(corrs_per_lag)
    max_corrs, best_lag_indices = torch.max(corrs_matrix, dim=0)
    
    loss_corr = 1.0 - max_corrs.mean().item()
    best_lags = best_lag_indices + min_lag_steps
    avg_lag_ms = best_lags.float().mean().item() * dt
    
    return loss_magnitude, loss_corr, avg_lag_ms