from torch.utils.data import DataLoader
from .simulated_dataset import NoisyDCDataset, PoissonNoiseDataset, NoisyDC_OUDataset
# 假设上述 Dataset 类在 simulated_dataset.py 中定义
from .other import is_train_mode

import numpy as np

def create_dataloaders(cfg, i_thr=None):
    """
    根据 cfg.dataset.type 创建 DataLoader
    """
    ds_cfg = cfg.dataset
    sim_cfg = cfg.simulation
    
    # 确定输入神经元数量 (需从 model 或 config 获取，这里假设已注入 cfg)
    num_input = cfg.network.get("n_neuron") 

    # 获取神经元数量
    n_neurons = cfg.network.get("n_neuron")
    
    # [核心逻辑]: 确定输入电流的均值 (Mean DC)
    dc_current_vector = None
    
    # 模式 A: 使用统一的固定电流 (原逻辑)
    if ds_cfg.get("current_mode", "fixed") == "fixed":
        current_val = ds_cfg.get("current", 0.0)
        dc_current_vector = float(current_val)
        
    # 模式 B: 基于 Rheobase 的缩放 (新逻辑)
    elif ds_cfg.get("current_mode") == "rheobase_scaled":
        if i_thr is None:
            raise ValueError("dataset mode is 'rheobase_scaled' but i_thr was not provided!")
        
        # 将 i_thr 转为 array
        i_thr_arr = np.array(i_thr, dtype=np.float32)
        
        # 获取缩放因子 (例如 1.1 表示 110% Rheobase)
        scale_factor = ds_cfg.get("scale_factor", 1.0)
        
        # 计算最终输入向量 [N]
        dc_current_vector = i_thr_arr * scale_factor
        
        # 安全检查: 确保维度匹配
        # 如果输入层是全连接，数据集维度通常等于神经元总数
        if len(dc_current_vector) != n_neurons:
             # 如果只给部分神经元输入，这里需要根据 input_indices 进行 mask 处理
             # 暂时假设这里是全量输入
             pass
        
        print(f"⚖️  Rheobase Scaling Enabled: Mean I_thr={np.mean(i_thr_arr):.2f} pA, Factor={scale_factor}")

    # --- 数据集实例化 ---
    
    if ds_cfg.type == "noisydc":
        print(f"📦 Creating NoisyDC Dataset (std={ds_cfg.std})")
        train_ds = NoisyDCDataset(
            num_samples=1000, 
            T=sim_cfg.T,
            num_neurons=n_neurons,
            dc_mean=dc_current_vector, # 传入向量或标量
            noise_std=ds_cfg.std
        )
        test_ds = NoisyDCDataset(
            num_samples=100,
            T=sim_cfg.T,
            num_neurons=n_neurons,
            dc_mean=dc_current_vector,
            noise_std=ds_cfg.std
        )
    elif ds_cfg.type == "noisydc_ou":
        print(f"📦 Creating NoisyDC_OU Dataset (std={ds_cfg.std})")
        train_ds = NoisyDC_OUDataset(
            num_samples=1000,
            T=sim_cfg.T,
            num_neurons=n_neurons,
            dc_mean=dc_current_vector,
            noise_std=ds_cfg.std,
            scale_noise_with_mean=ds_cfg.scale_noise_with_mean
        )
        test_ds = NoisyDC_OUDataset(
            num_samples=100,
            T=sim_cfg.T,
            num_neurons=n_neurons,
            dc_mean=dc_current_vector,
            noise_std=ds_cfg.std, 
            scale_noise_with_mean=ds_cfg.scale_noise_with_mean
        )
        
    elif ds_cfg.type == "poisson":
        print("📦 Creating Poisson Dataset")
        # ... logic for poisson ...
        train_ds = PoissonNoiseDataset(
            num_samples=1000,
            T=sim_cfg.T,
            num_neurons=num_input,
            firing_rate=ds_cfg.rate
        )
        test_ds = PoissonNoiseDataset(
            num_samples=100,
            T=sim_cfg.T,
            num_neurons=num_input,
            firing_rate=ds_cfg.rate
        )

    elif ds_cfg.type == "dc":
        print("📦 Creating DC Dataset")
        train_ds = NoisyDCDataset(
            num_samples=1000,
            T=sim_cfg.T,
            num_neurons=num_input,
            dc_mean=ds_cfg.current,
        )
        test_ds = NoisyDCDataset(
            num_samples=100,
            T=sim_cfg.T,
            num_neurons=num_input,
            dc_mean=ds_cfg.current,
        )
    else:
        raise ValueError(f"Unknown dataset type: {ds_cfg.type}")

    # 如果只有 Simulation 模式，可能不需要 Train Loader
    if not is_train_mode(cfg):
        train_loader = None
        test_loader = DataLoader(test_ds, batch_size=cfg.simulation.batch_size, shuffle=True, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=cfg.simulation.batch_size, shuffle=True, drop_last=True)
        
    # 这里为了演示简化，实际应该也创建 test_loader
    return train_loader, test_loader