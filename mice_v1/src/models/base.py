import logging

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.input_layers import build_input_adapter
from src.models.weights import WeightInitializer
from src.utils.preprocess import load_and_preprocess_mice
from src.utils.utils import load_neuron_args, load_synapse_args
from src.models import brain # 你的 flybrain 定义
from torch.nn import functional as F
from src.utils.other import is_train_mode

LOGGER = logging.getLogger(__name__)


class BaseRSNN(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.dtype = getattr(torch, cfg.get("dtype", "float32"))
        
        
        # 1. 加载 Connectome 数据 (Nodes & Edges)
        # 无论哪种权重模式，拓扑结构通常来自 connectome (除非是纯随机网络，暂按你的逻辑处理)
        #logger.info(f"🧠 Loading connectome from {network_cfg.conn_path}")
        self.connectome_data = load_and_preprocess_mice(cfg.network.conn_path)
        neurons, conn_mats, connections = self.connectome_data
        #conn_mats实际上有两部分：expanded_conn和receptor_df
        expanded_conn, receptor_df = conn_mats
        
        # 2. 确定 Input/Output 索引
        # TODO: 这里需要实现 select_indices 逻辑，根据 conf/input/partial.yaml
        #breakpoint()
        self.input_indices = self._select_input_neurons(neurons, cfg.input)
        # 将 indices 注入到 cfg 中以便后续使用
        # network_cfg.input_indices = self.input_indices 
        self._log_input_selection(neurons)
        
        # 3. 加载神经元参数 (GLIF)
        neuron_args, neuron_params = load_neuron_args(cfg.network, neurons)
        self.neuron_params = neuron_params
        
        # 4. 初始化权重 (核心改动：委托给 WeightInitializer)
        expanded_conn = WeightInitializer.apply(
            conn_mats, neurons, connections, neuron_params, cfg.network.weight
        )
        #WeightInitializer只修改expanded_conn，需要用到conn_mats（包含receptor_df和expanded_conn），但不修改receptor_df
        
        # 5. 加载突触参数
        synapse_args, synapse_module_type = load_synapse_args(
            cfg.network.synapse, neurons, expanded_conn, receptor_df
        )
        self.synapse_args = synapse_args
        print(f"synapse_args: {synapse_args}")
        print(f"synapse_module_type: {synapse_module_type}")
        
        # 6. 构建 FlyBrain
        self.flybrain = brain.FlyBrain(
            synapse_args=synapse_args, 
            neuron_args=neuron_args, 
            synapse_module_type=synapse_module_type
        )
        
        # 7. 构建 Input Layer (委托给 Factory)
        # 获取 voltage_scale 用于归一化
        vs = torch.tensor(neuron_params['voltage_scale'], device=self.device, dtype=self.dtype)

        #print(f"before build_input_adapter")
        #breakpoint()
        
        self.input_layer = build_input_adapter(
            config=cfg,    # e.g. noisydc info
            voltage_scale=vs,
            input_indices=self.input_indices,
            device=self.device,
            dtype=self.dtype
        )

        # Attach per-input-neuron I_thr for injected-current diagnostics.
        try:
            i_thr = self.neuron_params.get("I_thr", None)
            if i_thr is not None:
                i_thr_t = torch.as_tensor(i_thr, device=self.device, dtype=self.dtype)
                in_idx_t = torch.as_tensor(self.input_indices, device=self.device, dtype=torch.long)
                self.input_layer.debug_i_thr_inputs = i_thr_t[in_idx_t]
                LOGGER.info(
                    "Attached input-layer I_thr debug tensor: shape=%s",
                    tuple(self.input_layer.debug_i_thr_inputs.shape),
                )
        except Exception as exc:
            LOGGER.warning("Failed to attach I_thr debug tensor to input layer: %s", exc)

        self.flybrain.input_layer = self.input_layer

        self.params = {
            'neuron': self.neuron_params,
            'synapse': self.synapse_args,
        }

    def _select_input_neurons(self, neurons_df, input_cfg):
        # 简化的选择逻辑，实际应根据 input_cfg.type (all/partial) 实现
        if input_cfg.type == "all":
            return list(range(len(neurons_df)))
        elif input_cfg.type == "e_only":
            #选择所有的exc神经元
            exc_indices = neurons_df[neurons_df['EI'] == 'E'].index
            return list(exc_indices)

        else:
            # 随机选择 num_input_neuron 个
            count = input_cfg.get("num_input_neuron", 64)
            return list(range(min(count, len(neurons_df))))

    def _log_input_selection(self, neurons_df) -> None:
        try:
            n_total = int(len(neurons_df))
            n_input = int(len(self.input_indices))
            input_type = str(self.cfg.input.get("type", "unknown"))
            if "EI" in neurons_df.columns and n_input > 0:
                e_mask = (neurons_df["EI"].to_numpy() == "E")
                idx = torch.as_tensor(self.input_indices, dtype=torch.long).cpu().numpy()
                e_in = int(e_mask[idx].sum())
                i_in = int(n_input - e_in)
                LOGGER.info(
                    "Input selection | type=%s n_input=%d n_total=%d E_in=%d I_in=%d",
                    input_type,
                    n_input,
                    n_total,
                    e_in,
                    i_in,
                )
            else:
                LOGGER.info(
                    "Input selection | type=%s n_input=%d n_total=%d",
                    input_type,
                    n_input,
                    n_total,
                )
        except Exception as exc:
            LOGGER.warning("Failed to log input selection: %s", exc)

    def forward(self, x):
        # Simulation Mode: 纯前向，无 Loss
        #breakpoint()
        
        output, states = self.flybrain(x)
        return output, states

    def compute_loss(self, output, states, targets=None):
        """
        Train Mode: 子类或 Mixin 可以覆盖此方法
        但为了简单，可以在这里实现通用的 Loss 路由
        """
        # 如果 cfg.train 不存在，说明是 Sim 模式，不应调用此方法
        if not is_train_mode(self.cfg):
            return torch.tensor(0.0), {}
            
        total_loss = 0.0
        loss_dict = {}
        
        # 遍历 cfg.train.costs 定义的 Loss
        costs_cfg = self.cfg.train.costs
        
        # 示例：Voltage Loss
        if "voltage" in costs_cfg:
            v_cfg = costs_cfg.voltage
            v = states['neuron']['v']
            # 这里简单写，实际可以用你的 VoltageRegularizer 类
            l_v = torch.mean(F.relu(v - v_cfg.get("threshold", -30))**2) * v_cfg.get("weight", 1.0)
            total_loss += l_v
            loss_dict['voltage'] = l_v
            
        # 示例：Rate Loss
        if "rate" in costs_cfg:
            # ... rate loss logic ...
            pass
            
        return total_loss, loss_dict