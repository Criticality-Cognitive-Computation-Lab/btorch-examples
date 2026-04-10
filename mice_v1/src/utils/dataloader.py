# def create_data_loaders(
#     config: TrainingConfig,
#     network_config: "NetworkConfig" = None,
# ) -> tuple[data.DataLoader, data.DataLoader]:
#     """Create train and test data loaders.
    
#     支持三种模式：
#     1. MNIST数据集（原始模式）
#     2. 泊松噪声数据集（poisson_noise=True）
#     3. 直流输入数据集（dc_input=True）
#     """
    
#     if config.dc_input:
#         # 使用直流输入数据集
#         print(f"🚀 使用直流输入数据集")
#         print(f"   直流电流: {config.dc_input_current} pA")
#         print(f"   时间步数 T: {config.T}")
#         print(f"   输入神经元数: {network_config.num_input_neuron if network_config else 'unknown'}")
        
#         # 确定输入神经元数
#         num_input_neurons = network_config.num_input_neuron if network_config else 64
        
#         # 生成直流输入数据集
#         train_dataset = NoisyDCDataset(
#             num_samples=config.quick_train_samples if config.quick_validation else 10000,
#             T=config.T,
#             num_neurons=num_input_neurons,
#             dc_mean=config.dc_input_current,
#             noise_std=config.dc_input_current_std,
#             seed=config.seed,
#         )
        
#         test_dataset = NoisyDCDataset(
#             num_samples=config.quick_test_samples if config.quick_validation else 2000,
#             T=config.T,
#             num_neurons=num_input_neurons,
#             dc_mean=config.dc_input_current,
#             noise_std=config.dc_input_current_std,
#             seed=config.seed + 1,
#         )
        
#         print(f"   训练样本数: {len(train_dataset)}")
#         print(f"   测试样本数: {len(test_dataset)}")
    
#     elif config.poisson_noise:
#         # 使用泊松噪声数据集
#         print(f"🚀 使用泊松噪声数据集")
#         print(f"   发放率: {config.poisson_noise_intensity}")
#         print(f"   时间步数 T: {config.T}")
#         print(f"   输入神经元数: {network_config.num_input_neuron if network_config else 'unknown'}")
        
#         # 确定输入神经元数
#         num_input_neurons = network_config.num_input_neuron if network_config else 64
        
#         # 生成泊松噪声数据集
#         train_dataset = PoissonNoiseDataset(
#             num_samples=config.quick_train_samples if config.quick_validation else 10000,
#             T=config.T,
#             num_neurons=num_input_neurons,
#             firing_rate=config.poisson_noise_intensity,
#             dt=config.dt,
#             seed=config.seed,
#         )
        
#         test_dataset = PoissonNoiseDataset(
#             num_samples=config.quick_test_samples if config.quick_validation else 2000,
#             T=config.T,
#             num_neurons=num_input_neurons,
#             firing_rate=config.poisson_noise_intensity,
#             dt=config.dt,
#             seed=config.seed + 1,
#         )
        
#         print(f"   训练样本数: {len(train_dataset)}")
#         print(f"   测试样本数: {len(test_dataset)}")
        
#     else:
#         # 使用MNIST数据集（原始模式）
#         transform = torchvision.transforms.Compose(
#             [
#                 torchvision.transforms.ToTensor(),
#                 torchvision.transforms.Normalize((0,), (1,)),
#             ]
#         )

#         train_dataset = torchvision.datasets.MNIST(
#             root=config.data_dir,
#             train=True,
#             transform=transform,
#             download=True,
#         )

#         test_dataset = torchvision.datasets.MNIST(
#             root=config.data_dir,
#             train=False,
#             transform=transform,
#             download=True,
#         )

#         # 快速验证模式：使用数据集子集
#         if config.quick_validation:
#             print(f"🚀 快速验证模式已启用")
#             print(f"   训练样本数: {config.quick_train_samples} (原始: {len(train_dataset)})")
#             print(f"   测试样本数: {config.quick_test_samples} (原始: {len(test_dataset)})")
            
#             # 创建子集
#             train_indices = torch.randperm(len(train_dataset))[:config.quick_train_samples]
#             test_indices = torch.randperm(len(test_dataset))[:config.quick_test_samples]
            
#             train_dataset = data.Subset(train_dataset, train_indices)
#             test_dataset = data.Subset(test_dataset, test_indices)
            
#             print(f"   实际训练样本数: {len(train_dataset)}")
#             print(f"   实际测试样本数: {len(test_dataset)}")

#     train_loader = data.DataLoader(
#         dataset=train_dataset,
#         batch_size=config.batch_size,
#         shuffle=True,
#         drop_last=True,
#         num_workers=0 if (config.poisson_noise or config.dc_input) else config.num_workers,  # 非MNIST数据集不需要多进程
#         pin_memory=True,
#     )

#     test_loader = data.DataLoader(
#         dataset=test_dataset,
#         batch_size=config.batch_size,
#         shuffle=False,
#         drop_last=False,
#         num_workers=0 if (config.poisson_noise or config.dc_input) else config.num_workers,  # 非MNIST数据集不需要多进程
#         pin_memory=True,
#     )

#     return train_loader, test_loader


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