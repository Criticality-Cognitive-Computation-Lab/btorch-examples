import numpy as np
import torch
import torchvision

from torch.utils import data

import ipdb

class PoissonNoiseDataset(data.Dataset):
    """模拟数据集类，生成泊松噪声脉冲序列。
    
    功能类似于MNIST数据集，但直接提供符合设定发放率的泊松噪声脉冲序列。
    """
    
    def __init__(self, 
                 num_samples: int,
                 T: int,
                 num_neurons: int,
                 firing_rate: float = 0.5,
                 dt: float = 1.0,
                 seed: int = None):
        """
        Args:
            num_samples: 数据集中的样本数
            T: 仿真时间步数
            num_neurons: 神经元个数（输入维度）
            firing_rate: 泊松过程的发放率 (0-1)
            dt: 时间步长 (ms)
            seed: 随机种子
        """
        self.num_samples = num_samples
        self.T = T
        self.num_neurons = num_neurons
        self.firing_rate = firing_rate
        self.dt = dt
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # 预生成所有数据
        self.data = self._generate_poisson_spikes()
    
    def _generate_poisson_spikes(self) -> torch.Tensor:
        """生成泊松脉冲序列。
        
        Returns:
            spikes: 形状为 [num_samples, T, num_neurons] 的张量
        """
        prob = self.firing_rate * self.dt/1000

        #breakpoint()
        
        # 生成所有样本的随机数矩阵
        rand_matrix = torch.rand(self.num_samples, self.T, self.num_neurons)
        
        # 如果随机数小于 prob，则视为有脉冲 (1.0)，否则为 0
        poisson_spikes = (rand_matrix < prob).float()

        #breakpoint()经过检查1788时生成的dataset并无问题
        
        return poisson_spikes
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """返回单个样本的泊松脉冲序列。
        
        Args:
            idx: 样本索引
        
        Returns:
            spikes: 形状为 [T, num_neurons] 的张量
        """
        return self.data[idx]


class DCCurrentDataset(data.Dataset):
    """直流输入数据集类，生成恒定电流输入。
    
    为每个样本生成相同的恒定电流输入，形状为 [T, num_neurons]。
    """
    
    def __init__(self, 
                 num_samples: int,
                 T: int,
                 num_neurons: int,
                 dc_current: float = 0.5,
                 seed: int = None):
        """
        Args:
            num_samples: 数据集中的样本数
            T: 仿真时间步数
            num_neurons: 神经元个数（输入维度）
            dc_current: 直流电流大小 (pA)
            seed: 随机种子（用于一致性，虽然DC输入是确定的）
        """
        self.num_samples = num_samples
        self.T = T
        self.num_neurons = num_neurons
        self.dc_current = dc_current
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # 预生成所有数据（所有样本相同）
        self.data = self._generate_dc_current()
    
    def _generate_dc_current(self) -> torch.Tensor:
        """生成恒定电流输入。
        
        Returns:
            current: 形状为 [num_samples, T, num_neurons] 的张量，
                    所有时间步和神经元的值都是 dc_current
        """
        # 创建恒定电流张量：所有值都是 dc_current
        dc_input = torch.full(
            (self.num_samples, self.T, self.num_neurons),
            self.dc_current,
            dtype=torch.float32
        )
        return dc_input
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """返回单个样本的恒定电流输入。
        
        Args:
            idx: 样本索引
        
        Returns:
            current: 形状为 [T, num_neurons] 的张量
        """
        return self.data[idx]

class NoisyDCDataset(data.Dataset):
    """带噪声的直流输入数据集。
    
    模拟扩散近似（Diffusion Approximation）：
    Input(t) = DC_Level + Noise(t)
    其中 Noise(t) ~ Gaussian(0, std)
    """
    
    def __init__(self, 
                 num_samples: int,
                 T: int,
                 num_neurons: int,
                 dc_mean: float = 0.5,    # 平均电流强度 (对应 mu)
                 noise_std: float = 0.1,  # 噪声标准差 (对应 sigma)
                 dt: float = 1.0,         # 仿真步长，用于校准噪声物理量纲(可选)
                 seed: int = None):
        
        self.num_samples = num_samples
        self.T = T
        self.num_neurons = num_neurons

        # [逻辑处理]: 统一转为 tensor shape [num_neurons]
        if isinstance(dc_mean, (float, int)):
            self.dc_mean = torch.full((num_neurons,), float(dc_mean), dtype=torch.float32)
        else:
            self.dc_mean = torch.as_tensor(dc_mean, dtype=torch.float32)
            if self.dc_mean.shape[0] != num_neurons:
                raise ValueError(f"dc_mean dimension {self.dc_mean.shape} mismatch with num_neurons {num_neurons}")

        #self.dc_mean = dc_mean
        self.noise_std = noise_std
        self.dt = dt
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # 预生成数据
        self.data = self._generate_noisy_current()
    
    def _generate_noisy_current(self) -> torch.Tensor:
        """生成带噪声的电流输入"""
        # 1. 生成纯直流基底
        # dc_mean 是 [N], 需要扩展为 [Samples, T, N]
        #breakpoint()
        mean_tensor = self.dc_mean.unsqueeze(0).unsqueeze(0).expand(
            self.num_samples, self.T, self.num_neurons
        )
        #breakpoint()
        
        # 2. 生成高斯白噪声
        noise = torch.randn_like(mean_tensor) * self.noise_std * mean_tensor
        
        return mean_tensor + noise
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]

# class NoisyDC_OUDataset(data.Dataset):
#     """带 Ornstein-Uhlenbeck (OU) 噪声的直流输入数据集。
    
#     模拟具有时间自相关性和均值回归特性的背景突触噪声：
#     tau_ou * dI(t)/dt = -(I(t) - dc_mean) + sigma * sqrt(2 * tau_ou) * xi(t)
    
#     其中：
#     - dc_mean 是长期的目标均值
#     - noise_std (sigma) 是稳态下电流波动的标准差
#     - tau_ou 是噪声的时间常数（例如模拟 AMPA 受体时取 2~5 ms）
#     """
    
#     def __init__(self, 
#                  num_samples: int,
#                  T: int,
#                  num_neurons: int,
#                  dc_mean: float = 0.5,    # 平均电流强度 (对应 mu)
#                  noise_std: float = 0.1,  # 稳态噪声标准差 (对应 sigma)
#                  tau_ou: float = 5.0,     # OU 过程的时间常数 (ms)
#                  dt: float = 1.0,         # 仿真步长 (ms)
#                  seed: int = None):
        
#         self.num_samples = num_samples
#         self.T = T
#         self.num_neurons = num_neurons
        
#         # [逻辑处理]: 统一转为 tensor shape [num_neurons]
#         if isinstance(dc_mean, (float, int)):
#             self.dc_mean = torch.full((num_neurons,), float(dc_mean), dtype=torch.float32)
#         else:
#             self.dc_mean = torch.as_tensor(dc_mean, dtype=torch.float32)
#             if self.dc_mean.shape[0] != num_neurons:
#                 raise ValueError(f"dc_mean dimension {self.dc_mean.shape} mismatch with num_neurons {num_neurons}")
                
#         self.noise_std = noise_std
#         self.tau_ou = tau_ou
#         self.dt = dt
        
#         if seed is not None:
#             torch.manual_seed(seed)
#             np.random.seed(seed)
            
#         # 预生成数据
#         self.data = self._generate_ou_current()

#     def _generate_ou_current(self) -> torch.Tensor:
#         """使用 Euler-Maruyama 方法生成 OU 噪声电流序列"""
#         # 将 dc_mean 扩展为 [num_samples, num_neurons] 的基础形状
#         mu = self.dc_mean.unsqueeze(0).expand(self.num_samples, self.num_neurons)
        
#         # 准备输出张量 [num_samples, T, num_neurons]
#         ou_spikes = torch.zeros(self.num_samples, self.T, self.num_neurons, dtype=torch.float32)
        
#         # 预先计算迭代系数
#         decay = self.dt / self.tau_ou
#         # 为了让稳态标准差保持为 noise_std，这里根据欧拉离散化公式计算单步注入的噪声尺度
#         noise_scale = self.noise_std * np.sqrt(2.0 * self.dt / self.tau_ou)
        
#         # 初始化当前电流状态 (可以选择从均值开始，或者加上稳态标准差的随机初始化)
#         # 这里选择带随机初始化的方式，确保在 t=0 时刻系统就已经处于平稳分布
#         I_current = mu + torch.randn_like(mu) * self.noise_std
        
#         for t in range(self.T):
#             # 记录当前时间步的电流
#             ou_spikes[:, t, :] = I_current
            
#             # 生成独立的高斯白噪声 dW
#             dW = torch.randn(self.num_samples, self.num_neurons, dtype=torch.float32)
            
#             # 向量化更新下一步的电流状态
#             I_current = I_current + (mu - I_current) * decay + noise_scale * dW
            
#         return ou_spikes
        
#     def __len__(self) -> int:
#         return self.num_samples
        
#     def __getitem__(self, idx: int) -> torch.Tensor:
#         """返回单个样本的 OU 噪声电流序列。
        
#         Returns:
#             current: 形状为 [T, num_neurons] 的张量
#         """
#         return self.data[idx]

class NoisyDC_OUDataset(data.Dataset):
    def __init__(self, 
                 num_samples: int,
                 T: int,
                 num_neurons: int,
                 dc_mean: float | torch.Tensor, # 可以是标量，也可以是形状为 [num_neurons] 的张量
                 noise_std: float | torch.Tensor = 0.1,  # 现在它可以是标量系数，也可以是张量
                 tau_ou: float = 5.0,
                 dt: float = 1.0,
                 seed: int = None,
                 scale_noise_with_mean: bool = True): # 新增控制开关
        
        self.num_samples = num_samples
        self.T = T
        self.num_neurons = num_neurons
        
        # 1. 处理 dc_mean 异质性
        if isinstance(dc_mean, (float, int)):
            self.dc_mean = torch.full((num_neurons,), float(dc_mean), dtype=torch.float32)
        else:
            self.dc_mean = torch.as_tensor(dc_mean, dtype=torch.float32)
            
        # 2. 处理 noise_std 异质性
        if isinstance(noise_std, (float, int)):
            if scale_noise_with_mean:
                # 核心修改：如果开启按均值缩放，则相当于 noise_std 传入的是变异系数 (CV)
                # 实际的绝对噪声强度 = 比例因子 * |均值电流|
                # 这样 I_thr 大的神经元，分到的绝对噪声方差也大
                self.noise_std = float(noise_std) * torch.abs(self.dc_mean)
            else:
                # 否则作为统一的绝对值
                self.noise_std = torch.full((num_neurons,), float(noise_std), dtype=torch.float32)
        else:
            # 允许用户直接传入外部计算好的、针对每个神经元的专属噪声标准差张量
            self.noise_std = torch.as_tensor(noise_std, dtype=torch.float32)
            
        if self.noise_std.shape[0] != num_neurons:
            raise ValueError(f"noise_std dimension {self.noise_std.shape} mismatch with num_neurons {num_neurons}")

        self.tau_ou = tau_ou
        self.dt = dt
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        self.data = self._generate_ou_current()

    def _generate_ou_current(self) -> torch.Tensor:
        # 扩展为 [num_samples, num_neurons]
        mu = self.dc_mean.unsqueeze(0).expand(self.num_samples, self.num_neurons)
        sigma = self.noise_std.unsqueeze(0).expand(self.num_samples, self.num_neurons)
        
        ou_spikes = torch.zeros(self.num_samples, self.T, self.num_neurons, dtype=torch.float32)
        decay = self.dt / self.tau_ou
        
        # 注意这里：noise_scale 此时是一个形状为 [num_samples, num_neurons] 的张量
        noise_scale = sigma * np.sqrt(2.0 * self.dt / self.tau_ou)
        
        # 初始化
        I_current = mu + torch.randn_like(mu) * sigma
        
        for t in range(self.T):
            ou_spikes[:, t, :] = I_current
            dW = torch.randn(self.num_samples, self.num_neurons, dtype=torch.float32)
            # 由于 noise_scale 是逐元素的，异质性噪声在这里被完美应用
            I_current = I_current + (mu - I_current) * decay + noise_scale * dW
            
        return ou_spikes

    def __len__(self) -> int:
        return self.num_samples
        
    def __getitem__(self, idx: int) -> torch.Tensor:
        """返回单个样本的 OU 噪声电流序列。
        
        Returns:
            current: 形状为 [T, num_neurons] 的张量
        """
        return self.data[idx]