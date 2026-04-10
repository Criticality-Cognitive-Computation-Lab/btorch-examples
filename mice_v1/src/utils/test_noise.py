import torch
import matplotlib.pyplot as plt

# 假设你的数据集类保存在名为 dataset.py 的文件中
# 如果在同一个文件中，请直接将这段代码追加到文件末尾
from src.utils.simulated_dataset import NoisyDCDataset, NoisyDC_OUDataset

def plot_noise_comparison():
    # --- 1. 设置仿真参数 ---
    num_samples = 1       # 只需要1个样本进行测试
    T = 200               # 观察 200 个时间步 (例如 200 ms)
    num_neurons = 3       # 观察 3 个独立的神经元
    
    dc_mean = 1.05        # 基础直流均值 (模拟 1.05 * I_thr)
    noise_std = 0.15      # 噪声标准差
    tau_ou = 5.0          # OU 过程时间常数 (ms)
    dt = 1.0              # 时间步长 (ms)

    # --- 2. 实例化数据集 ---
    print("Generating White Noise Dataset...")
    white_noise_ds = NoisyDCDataset(
        num_samples=num_samples, T=T, num_neurons=num_neurons, 
        dc_mean=dc_mean, noise_std=noise_std, dt=dt, seed=42
    )

    print("Generating OU Noise Dataset...")
    ou_noise_ds = NoisyDC_OUDataset(
        num_samples=num_samples, T=T, num_neurons=num_neurons, 
        dc_mean=dc_mean, noise_std=noise_std, tau_ou=tau_ou, dt=dt, seed=42
    )

    # --- 3. 提取数据 ---
    # 取出第 0 个样本，形状为 [T, num_neurons]
    white_noise_data = white_noise_ds[0].numpy()
    ou_noise_data = ou_noise_ds[0].numpy()
    
    time_axis = torch.arange(T).numpy() * dt

    # --- 4. 绘制对比图 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)

    # 绘制高斯白噪声 (NoisyDCDataset)
    for i in range(num_neurons):
        ax1.plot(time_axis, white_noise_data[:, i], alpha=0.7, label=f'Neuron {i+1}')
    ax1.axhline(dc_mean, color='black', linestyle='--', label='DC Mean (1.05)')
    ax1.set_title('Gaussian White Noise (NoisyDCDataset)')
    ax1.set_ylabel('Input Current')
    ax1.legend(loc='upper right')

    # 绘制 OU 噪声 (NoisyDC_OUDataset)
    for i in range(num_neurons):
        ax2.plot(time_axis, ou_noise_data[:, i], alpha=0.7, label=f'Neuron {i+1}')
    ax2.axhline(dc_mean, color='black', linestyle='--', label='DC Mean (1.05)')
    ax2.set_title(f'Ornstein-Uhlenbeck Noise (NoisyDC_OUDataset, tau={tau_ou}ms)')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Input Current')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
    plt.savefig('noise_comparison.png')
    plt.close()
    print("Noise comparison plot saved to noise_comparison.png")

if __name__ == "__main__":
    plot_noise_comparison()