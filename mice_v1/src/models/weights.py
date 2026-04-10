import torch
import logging
from btorch.connectome import augment
import numpy as np
import pandas as pd
import scipy.sparse as sp

import ipdb

logger = logging.getLogger(__name__)


def generate_random_weights(conn_type: str, distribution: str, params: dict, n_connections: int, seed: int = None) -> np.ndarray:
    """
    为指定连接类型生成随机权重
    
    Args:
        conn_type: 连接类型 ('EE', 'EI', 'IE', 'II')
        distribution: 分布类型 ('lognormal', 'normal')
        params: 分布参数字典
        n_connections: 连接数量
        seed: 随机种子
    
    Returns:
        长度为 n_connections 的权重数组
    """
    if n_connections == 0:
        return np.array([], dtype=np.float32)
    
    rng = np.random.default_rng(seed)
    
    if distribution == 'lognormal':
        # lognormal分布：log(weight) ~ N(mu, sigma^2)
        # 权重绝对值取lognormal分布
        # mu = params.get('mu', 0.0)
        sigma = params.get('sigma', 1.0)
        mu = -0.5*(sigma**2) #利用lognormal的性质，只需要指定sigma，可以保证实际权重均值为1，再通过scale方法调整整体权重，sigma指定长尾程度
        weights = rng.lognormal(mean=mu, sigma=sigma, size=n_connections)
        
    elif distribution == 'normal':
        # normal分布：weight ~ N(mu, sigma^2)
        mu = params.get('mu', 0.0)
        sigma = params.get('sigma', 1.0)
        weights = rng.normal(loc=mu, scale=sigma, size=n_connections)
        
    else:
        raise ValueError(f"Unsupported distribution type: {distribution}")
    
    return weights.astype(np.float32)

def make_ei_conn_mat_with_random_weights(
    conn_data: tuple[sp.sparray, pd.DataFrame],  # <--- 修改输入签名
    neurons: pd.DataFrame,
    connections: pd.DataFrame = None,
    weight_distributions: dict[str, dict] = None,
    weight_seed: int = None,
    weight_scales: dict[str, float] = None,
    neuron_params: dict = None, 
    use_balanced_ie: bool = True,
    ie_ratio_g: float = 4.0,
):
    """
    为 make_hetersynapse_conn 生成的扩展矩阵赋予随机权重。
    
    Args:
        conn_data: (expanded_mat, receptor_df) 元组
        ... (其他参数保持不变)
    """
    
    # 解包输入
    expanded_mat, receptor_df = conn_data
    # 确保是 COO 格式以便快速索引
    coo_mat = expanded_mat.tocoo()
    
    # 1. 初始化配置 (保持不变)
    if weight_distributions is None:
        print("Warning: Using random weights, but no weight distributions found, using default values")
        weight_distributions = {
            'EE': {'distribution': 'normal', 'params': {'mu': 1.0, 'sigma': 0.1}},
            'EI': {'distribution': 'normal', 'params': {'mu': 1.0, 'sigma': 0.1}},
            'IE': {'distribution': 'normal', 'params': {'mu': 1.0, 'sigma': 0.1}},
            'II': {'distribution': 'normal', 'params': {'mu': 1.0, 'sigma': 0.1}}
        }
    
    if weight_scales is None:
        weight_scales = {'EE': 1.0, 'EI': 1.0, 'IE': 1.0, 'II': 1.0}

    # 获取 voltage_scale
    voltage_scale_array = None
    if neuron_params is not None and 'voltage_scale' in neuron_params:
        voltage_scale_array = np.array(neuron_params['voltage_scale'], dtype=np.float32)
    else:
        voltage_scale_array = np.ones(len(neurons), dtype=np.float32)
        
    n_neurons = len(neurons)
    n_receptors = len(receptor_df) # 例如 4 (EE, EI, IE, II)

    # --- 2. 构建掩码 (利用 receptor_df 极速构建) ---
    # receptor_df 结构: [receptor_index, pre_receptor_type, post_receptor_type]
    # expanded_mat 的列索引 j 对应的 receptor_index = j % n_receptors
    
    # 计算 COO 矩阵中每个非零元素对应的 receptor_index
    # 注意：make_hetersynapse_conn 的列结构是 [post_neuron_idx * n_receptors + receptor_idx]
    # 所以取模运算可以直接得到 receptor_index
    edge_receptor_indices = coo_mat.col % n_receptors

    #breakpoint()
    
    # 从 DataFrame 构建查找表 (Dict 或 Array)
    # Map: receptor_index -> "EE"/"EI"/...
    rec_idx_to_type = {}
    for _, row in receptor_df.iterrows():
        idx = row['receptor_index']
        pre_t = row['pre_receptor_type']
        post_t = row['post_receptor_type']
        rec_idx_to_type[idx] = f"{pre_t}{post_t}" # e.g. "EE", "IE"

    #breakpoint()
    # 生成每条边的连接类型字符串数组 (比之前的 map 快得多)
    # 使用 numpy 的 take 或者列表推导
    # 为了速度，我们可以先映射成整数再转 (0:EE, 1:EI...)，或者直接用掩码
    
    masks = {}
    counts = {}
    
    for conn_type in ['EE', 'EI', 'IE', 'II']:
        # 找到属于该类型的所有 receptor_indices
        target_indices = [
            idx for idx, type_str in rec_idx_to_type.items() 
            if type_str == conn_type
        ]

        #breakpoint()
        
        # 构建掩码：如果边的 receptor_index 在目标列表中，则为 True
        mask = np.isin(edge_receptor_indices, target_indices)
        #breakpoint()
        masks[conn_type] = mask
        counts[conn_type] = np.sum(mask)


    # --- 3. 生成基础随机权重 ---
    random_weights = {}
    target_types = ['EE', 'EI', 'II'] if use_balanced_ie else ['EE', 'IE', 'EI', 'II']
    
    # (假设 generate_random_weights 在外部或此处定义，同原代码)
    gen_weights_func = globals().get('generate_random_weights', 
                                     lambda t, d, p, c, s: np.random.normal(p['mu'], p['sigma'], c).astype(np.float32) if d=='normal' else np.ones(c, dtype=np.float32))

    for conn_type in target_types:
        count = counts[conn_type]
        if count == 0:
            random_weights[conn_type] = np.array([], dtype=np.float32)
            continue
        
        dist_config = weight_distributions.get(conn_type, {'distribution': 'constant', 'params': {}})
        w = gen_weights_func(conn_type, dist_config.get('distribution'), dist_config.get('params'), count, weight_seed)
        w *= weight_scales.get(conn_type, 1.0)
        random_weights[conn_type] = np.abs(w)

    # --- 4. [I-E 平衡逻辑] (适配新结构) ---
    if use_balanced_ie:
        print(f"\n[Balanced IE] 计算 I-E 均衡权重 (g={ie_ratio_g})...")
        if counts['EE'] > 0 and counts['IE'] > 0:
            # A. 计算 E 输入 (Magnitude)
            # 这里的 coo_mat.col // n_receptors 才是真实的 post_neuron_id
            real_post_ids = coo_mat.col // n_receptors
            
            ee_post_ids = real_post_ids[masks['EE']]
            ee_weights = random_weights['EE']
            total_e_input = np.bincount(ee_post_ids, weights=ee_weights, minlength=n_neurons)
            
            # B. 计算 I 输入度 (In-degree)
            ie_post_ids = real_post_ids[masks['IE']]
            i_in_degree = np.bincount(ie_post_ids, minlength=n_neurons)
            
            # C. 映射回每条连接
            post_total_e = total_e_input[ie_post_ids]
            post_i_degree = i_in_degree[ie_post_ids]
            
            # D. 计算目标均值
            valid_mask = post_i_degree > 0
            mu_vec = np.zeros_like(post_total_e, dtype=np.float32)
            mu_vec[valid_mask] = (ie_ratio_g * post_total_e[valid_mask]) / post_i_degree[valid_mask]
            
            # E. 采样
            rng = np.random.default_rng(weight_seed)
            sigma_vec = 0.25 * mu_vec
            raw_samples = rng.standard_normal(len(mu_vec), dtype=np.float32)
            ie_weights_final = np.maximum(mu_vec + sigma_vec * raw_samples, 0.0)
            
            random_weights['IE'] = ie_weights_final
            print(f"  -> IE weights generated (Count={counts['IE']})")
        else:
            random_weights['IE'] = np.zeros(counts['IE'], dtype=np.float32)

    # --- 5. 组装最终权重 ---
    scaled_data = np.zeros_like(coo_mat.data, dtype=np.float32)
    
    for c_type, mask in masks.items():
        if counts[c_type] > 0 and c_type in random_weights:
            # 注意：base_val 来自 coo_mat.data，通常是 syn_count (正数)
            # 我们不需要处理正负号，因为 expand 矩阵已经物理隔离了 E 和 I 通道
            # 在 SNN 仿真时，模型会根据 receptor_type 自动决定电流符号 (I 通常对应负电流或翻转电位)
            base_val = coo_mat.data[mask]
            rand_mag = random_weights[c_type]
            scaled_data[mask] = base_val * rand_mag

    # --- 6. 应用 Voltage Scale ---
    if voltage_scale_array is not None:
        real_post_ids = coo_mat.col // n_receptors # 还原为神经元 ID
        valid_idx_mask = real_post_ids < len(voltage_scale_array)
        
        divisors = np.ones_like(scaled_data)
        divisors[valid_idx_mask] = voltage_scale_array[real_post_ids[valid_idx_mask]]
        divisors[divisors == 0] = 1.0
        
        scaled_data = scaled_data / divisors

    # --- 7. 返回结果 ---
    # 返回一个新的稀疏矩阵，保持原来的形状
    scaled_mat = sp.coo_array(
        (scaled_data, (coo_mat.row, coo_mat.col)),
        shape=coo_mat.shape,
        dtype=np.float32
    )
    
    # 保持接口兼容，只返回矩阵，但这个矩阵现在是 hetersynaptic 的
    return scaled_mat.tocsr()

def make_ei_conn_mat_from_conn(
    conn_data: tuple[sp.sparray, pd.DataFrame],
    neurons: pd.DataFrame,
    connections: pd.DataFrame,
    scales: dict[str, float] | None = None,
    weight_cfg: dict | None = None,
    neuron_params: dict | None = None,
):
    """将 connectome 的逐边权重写入 hetersynaptic 扩展连接矩阵。

    目标：仿照 make_ei_conn_mat_with_random_weights 的缩放逻辑：
    - expanded_mat.data 通常是 syn_count（或其它基值）
    - connections['weight'] 提供每条 (pre_simple_id, post_simple_id) 的突触强度
    - 针对 EE/EI/IE/II 依据 receptor_df 生成 mask，并用 scales 做整体缩放

    注意：expanded_mat 的列是 "post_neuron_id * n_receptors + receptor_index"。
    """

    #breakpoint()

    expanded_mat, receptor_df = conn_data
    coo_mat = expanded_mat.tocoo()

    if connections is None or len(connections) == 0:
        raise ValueError("connections is required and must contain per-edge 'weight'.")
    if "pre_simple_id" not in connections.columns or "post_simple_id" not in connections.columns:
        raise ValueError("connections must contain columns: pre_simple_id, post_simple_id")
    if "weight" not in connections.columns:
        raise ValueError("connections must contain column: weight")

    if scales is None:
        scales = {"EE": 1.0, "EI": 1.0, "IE": 1.0, "II": 1.0}

    # voltage_scale：与 random 版本一致
    if neuron_params is not None and "voltage_scale" in neuron_params:
        voltage_scale_array = np.array(neuron_params["voltage_scale"], dtype=np.float32)
    else:
        voltage_scale_array = np.ones(len(neurons), dtype=np.float32)

    n_neurons = len(neurons)
    n_receptors = len(receptor_df)

    # -------------------------------------------------
    # 1) 构建 receptor_index -> "EE"/"EI"/"IE"/"II" 映射
    # -------------------------------------------------
    rec_idx_to_type: dict[int, str] = {}
    for _, row in receptor_df.iterrows():
        idx = int(row["receptor_index"])
        pre_t = str(row["pre_receptor_type"])
        post_t = str(row["post_receptor_type"])
        rec_idx_to_type[idx] = f"{pre_t}{post_t}"

    # 对每个非零元算 receptor_index
    edge_receptor_indices = coo_mat.col % n_receptors

    masks: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for conn_type in ["EE", "EI", "IE", "II"]:
        target_indices = [idx for idx, t in rec_idx_to_type.items() if t == conn_type]
        mask = np.isin(edge_receptor_indices, target_indices)
        masks[conn_type] = mask
        counts[conn_type] = int(np.sum(mask))

    # -------------------------------------------------
    # 2) 构建 (pre_simple_id, post_simple_id) -> weight 的查找表
    #    要求键唯一（否则同一对神经元有多条权重会歧义）
    # -------------------------------------------------
    # 用 MultiIndex Series 做快速对齐
    weight_series = connections.set_index(["pre_simple_id", "post_simple_id"])["weight"]
    if weight_series.index.has_duplicates:
        # 如果你的 connections 允许同一对 (pre,post) 多条记录，你需要先聚合（sum/mean）再用。
        raise ValueError(
            "connections has duplicate (pre_simple_id, post_simple_id). "
            "Please aggregate first (e.g., groupby sum/mean)."
        )

    # -------------------------------------------------
    # 3) 为每个非零元找到对应的 (pre_id, post_id)
    # -------------------------------------------------
    real_pre_ids = coo_mat.row.astype(np.int64)
    real_post_ids = (coo_mat.col // n_receptors).astype(np.int64)

    # 用 MultiIndex reindex 对齐到 COO 顺序，保证严格一一对应
    query_index = pd.MultiIndex.from_arrays(
        [real_pre_ids, real_post_ids], names=["pre_simple_id", "post_simple_id"]
    )
    conn_weights = weight_series.reindex(query_index).to_numpy(dtype=np.float32)

    n_missing = int(np.isnan(conn_weights).sum())
    if n_missing > 0:
        # 这通常意味着：expanded_mat 里存在某些边，但 connections 表里没有对应权重
        missing_examples = query_index[np.isnan(conn_weights)][:5]
        raise ValueError(
            f"{n_missing} edges in expanded_mat have no corresponding weight in connections. "
            f"Examples: {list(missing_examples)}"
        )

    # -------------------------------------------------
    # 4) 应用 EE/EI/IE/II 的整体 scale，并乘以 base_val（syn_count）
    # -------------------------------------------------
    scaled_data = np.zeros_like(coo_mat.data, dtype=np.float32)

    for conn_type, mask in masks.items():
        if counts[conn_type] == 0:
            continue
        base_val = coo_mat.data[mask].astype(np.float32)
        w = conn_weights[mask]
        w = np.abs(w) * float(scales.get(conn_type, 1.0))
        scaled_data[mask] = base_val * w

    # -------------------------------------------------
    # 5) 应用 voltage_scale（与 random 版本一致）
    # -------------------------------------------------
    if voltage_scale_array is not None:
        valid_idx_mask = real_post_ids < len(voltage_scale_array)
        divisors = np.ones_like(scaled_data)
        divisors[valid_idx_mask] = voltage_scale_array[real_post_ids[valid_idx_mask]]
        divisors[divisors == 0] = 1.0
        scaled_data = scaled_data / divisors

    scaled_mat = sp.coo_array(
        (scaled_data, (coo_mat.row, coo_mat.col)),
        shape=coo_mat.shape,
        dtype=np.float32,
    )
    return scaled_mat.tocsr()


class WeightInitializer:
    @staticmethod
    def apply(conn_mats, neurons, connections, neuron_params, weight_cfg):
        #breakpoint()
        """
        策略模式：根据 weight_cfg.type 决定如何处理权重
        """
        mode = weight_cfg.get("type", "random")
        
        # 提取全局缩放因子
        scales = {
            'EE': weight_cfg.get('weight_ee_scale', 1.0) * weight_cfg.get('weight_scale', 1.0),
            'EI': weight_cfg.get('weight_ei_scale', 1.0) * weight_cfg.get('weight_scale', 1.0),
            'IE': weight_cfg.get('weight_ie_scale', 1.0) * weight_cfg.get('weight_scale', 1.0),
            'II': weight_cfg.get('weight_ii_scale', 1.0) * weight_cfg.get('weight_scale', 1.0),
        }

        if mode == "random":
            logger.info("⚖️  Using RANDOM weight initialization strategy")
            # 构造分布参数字典 (适配原 utils.py init_weights 的接口)
            # 这里将 Hydra Config 转换为 utils 需要的格式
            weight_distributions = {
                'EE': {'distribution': weight_cfg.get('weight_dist_ee', 'normal'), 
                       'params': {'mu': weight_cfg.get('weight_mu_ee', 1.0), 'sigma': weight_cfg.get('weight_sigma_ee', 0.2)}},
                'EI': {'distribution': weight_cfg.get('weight_dist_ei', 'normal'),
                       'params': {'mu': weight_cfg.get('weight_mu_ei', 1.0), 'sigma': weight_cfg.get('weight_sigma_ei', 0.2)}},
                'IE': {'distribution': weight_cfg.get('weight_dist_ie', 'normal'),
                       'params': {'mu': weight_cfg.get('weight_mu_ie', 1.0), 'sigma': weight_cfg.get('weight_sigma_ie', 0.2)}},
                'II': {'distribution': weight_cfg.get('weight_dist_ii', 'normal'),
                       'params': {'mu': weight_cfg.get('weight_mu_ii', 1.0), 'sigma': weight_cfg.get('weight_sigma_ii', 0.2)}},
                'EE': {'distribution': weight_cfg.get('weight_dist_ee', 'normal'), 
                       'params': {'mu': weight_cfg.get('weight_mu_ee', 1.0), 'sigma': weight_cfg.get('weight_sigma_ee', 0.2)}},
                'EI': {'distribution': weight_cfg.get('weight_dist_ei', 'normal'),
                       'params': {'mu': weight_cfg.get('weight_mu_ei', 1.0), 'sigma': weight_cfg.get('weight_sigma_ei', 0.2)}},
                'IE': {'distribution': weight_cfg.get('weight_dist_ie', 'normal'),
                       'params': {'mu': weight_cfg.get('weight_mu_ie', 1.0), 'sigma': weight_cfg.get('weight_sigma_ie', 0.2)}},
                'II': {'distribution': weight_cfg.get('weight_dist_ii', 'normal'),
                       'params': {'mu': weight_cfg.get('weight_mu_ii', 1.0), 'sigma': weight_cfg.get('weight_sigma_ii', 0.2)}},
            }
            print(f"conn_mats: {conn_mats}")
            # 调用原 utils 中的逻辑 (augment)
            conn_mat = make_ei_conn_mat_with_random_weights(
                conn_mats, neurons, connections, weight_distributions, 
                weight_cfg.get('seed', 42), scales, neuron_params,
                use_balanced_ie=weight_cfg.get('use_balanced_ie', False),
                ie_ratio_g=weight_cfg.get('ie_ratio_g', 4.0)
            )
            
        elif mode == "from_conn":
            logger.info("⚖️  Using PRE-LOADED weights from connectome (Detailed)")
            # 模式 2：假设 connectome 数据中包含了权重，或者使用 Detailed 逻辑
            # 调用原 utils.py 中的 make_ei_conn_mat_from_conn
            # 此时我们主要依赖 scales，但基础数值来自 connectome 文件
            conn_mat = make_ei_conn_mat_from_conn(
                conn_mats, neurons, connections, scales, weight_cfg, neuron_params
            )
            
        else:
            raise ValueError(f"Unknown weight initialization mode: {mode}")

        return conn_mat