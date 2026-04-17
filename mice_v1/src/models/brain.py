from typing import Any, overload

import numpy as np
import pandas as pd
import torch

from btorch.utils.dict_utils import unflatten_dict
from btorch.models.neurons.glif import GLIF3
from btorch.models.rnn import RecurrentNN
from btorch.models.shape import expand_leading_dims
from btorch.models.synapse import AlphaPSC, HeterSynapsePSC#, HeterSynapseDualPSC #, GLIFAlphaPSCFull

import ipdb


def get_simple_id(df: pd.DataFrame) -> np.ndarray:
    return df.simple_id.to_numpy()


class NeuronEmbedMapLayer(torch.nn.Module):
    def __init__(self, neurons: pd.DataFrame):
        super().__init__()
        self.neuron_embed_map: dict[str, Any] = {}
        self.define_default(neurons)
        self.n_neuron = len(neurons)

    @overload
    def register_neuron_embed(
        self, key: dict, embed: None = ..., neuron_id: None = ..., *args, override=False
    ): ...
    def register_neuron_embed(
        self, key: str, embed: torch.nn.Module, neuron_id, *args, override=False
    ):
        if isinstance(key, dict):
            dict_k_emb_arg = key
            for k, emb_arg in dict_k_emb_arg.items():
                self.register_neuron_embed(k, *emb_arg, override)
            return

        assert override or key not in self.neuron_embed_map
        # assume neuron_id_name is never modified afterwards,
        # so storing a referencce is safe
        self.neuron_embed_map[key] = (embed, neuron_id, *args)
        # so that embed appears in self.modules
        setattr(self, key, embed)

    def define_default(self, neurons):
        pass


# TODO: make this independent of torch
class EnvInputLayer(NeuronEmbedMapLayer):
    def define_default(self, neurons):
        from ..connectome import neuron_population

        vision_neuron_id = np.hstack(
            [
                get_simple_id(v)
                for v in neuron_population.get_optics(neurons, ("R7", "R8")).values()
            ]
        )
        vision_embed = torch.nn.Sequential(
            torch.nn.Linear(in_features=1, out_features=len(vision_neuron_id)),
            # torch.nn.LayerNorm(len(vision_neuron_id)),
            torch.nn.ReLU(),
        )
        self.register_neuron_embed("vision", vision_embed, vision_neuron_id)
        johnston_neuron_id = np.hstack(
            [
                get_simple_id(v)
                for v in neuron_population.get_johnston(neurons, ("C", "E")).values()
            ]
        )
        johnston_embed = torch.nn.Sequential(
            torch.nn.Linear(in_features=1, out_features=len(johnston_neuron_id)),
            # torch.nn.LayerNorm(len(johnston_neuron_id)),
            torch.nn.ReLU(),
        )
        self.register_neuron_embed("wind_gravity", johnston_embed, johnston_neuron_id)
        an_neuron_id = get_simple_id(neuron_population.get_AN(neurons))
        an_embed = torch.nn.Sequential(
            torch.nn.Linear(in_features=1, out_features=len(an_neuron_id)),
            # torch.nn.LayerNorm(len(an_neuron_id)),
            torch.nn.ReLU(),
        )
        self.register_neuron_embed("an", an_embed, an_neuron_id)

    def forward(self, observ: dict[str, Any]):
        # TODO: merge these into one scatter
        ret = None
        for k, v in observ.items():
            embed, neuron_id = self.neuron_embed_map[k]
            src = embed(v)
            neuron_id = torch.tensor(neuron_id, dtype=torch.long, device=src.device)
            index = expand_leading_dims(neuron_id, src.shape[:-1])
            if ret is None:
                ret = torch.zeros(
                    src.shape[:-1] + (self.n_neuron,),
                    device=src.device,
                    dtype=src.dtype,
                )
            ret = ret.scatter_add(dim=-1, index=index, src=src)
        return ret


class EnvOutputLayer(NeuronEmbedMapLayer):
    def define_default(self, neurons):
        from ..connectome import neuron_population

        dn_neuron_id = get_simple_id(neuron_population.get_DN(neurons))
        dn_embed = torch.nn.Identity()
        self.register_neuron_embed("dn", dn_embed, dn_neuron_id, "neuron.v")
        mbon_neuron_id = get_simple_id(neuron_population.get_mbon(neurons))
        mbon_embed = torch.nn.Identity()
        self.register_neuron_embed("mbon", mbon_embed, mbon_neuron_id, "neuron.v")

    def forward(self, x: dict[str, Any]):
        # TODO: merge these into one scatter
        ret = {}
        for name, (embed, neuron_id, out_attr) in self.neuron_embed_map.items():
            ret[name] = embed(x[out_attr][..., neuron_id])

        return ret


class FlyBrain(torch.nn.Module):
    def __init__(
        self,
        synapse_args: dict,
        neuron_args: dict,
        neuron_module_type: type = GLIF3,
        synapse_module_type: type = AlphaPSC,
        input_layer: torch.nn.Module = None,
        output_layer: torch.nn.Module = None,
    ):
        super().__init__()

        # neurons
        neuron = neuron_module_type(
            **{
                "v_rest": -52.0,  # mV
                "v_threshold": -50.0,  # mV
                "v_reset": -60.0,  # mV
                "tau": 10.0,  # ms
                "hard_reset": True,
                **neuron_args,
            },
        )
        #print(f"in FlyBrain, neuron_args: {neuron_args}")
        #print(f"neuron: {neuron}")
        
        # Create synapse with appropriate type
        if synapse_module_type is AlphaPSC:
            # AlphaPSC: 需要 n_neuron；若外部未提供则注入
            if "n_neuron" not in synapse_args:
                synapse = AlphaPSC(n_neuron=neuron_args["n_neuron"], **synapse_args)
            else:
                synapse = AlphaPSC(**synapse_args)
        # elif synapse_module_type in [GLIFAlphaPSCFull]:
        #     print(f"synapse_module_type: {synapse_module_type}")
        #     print(f"synapse_args: {synapse_args}")
        #     # For GLIFAlphaPSC variants, we need tau_syn_matrix instead of tau_syn
        #     synapse_args_glif = synapse_args.copy()
        #     if "tau_syn_matrix" not in synapse_args_glif:
        #         print(f"Warning: tau_syn_matrix is not provided, creating it from tau_syn")
        #         # If tau_syn_matrix is not provided, create it from tau_syn
        #         tau_syn = synapse_args_glif.get("tau_syn")
        #         if isinstance(tau_syn, torch.Tensor) and tau_syn.dim() == 1:
        #             # Convert 1D tau_syn to matrix by broadcasting
        #             n_neuron = neuron_args["n_neuron"]
        #             tau_syn_matrix = tau_syn.unsqueeze(0).expand(n_neuron, n_neuron)
        #             synapse_args_glif["tau_syn_matrix"] = tau_syn_matrix
        #         else:
        #             # Scalar tau_syn - create uniform matrix
        #             n_neuron = neuron_args["n_neuron"]
        #             tau_syn_matrix = torch.full((n_neuron, n_neuron), tau_syn)
        #             synapse_args_glif["tau_syn_matrix"] = tau_syn_matrix
            
            # Note: For GLIFAlphaPSCFull we no longer replace the linear layer.
            # The connection-specific tau scaling is handled inside GLIFAlphaPSCFull.
            
            # synapse = synapse_module_type(n_neuron=neuron_args["n_neuron"], **synapse_args_glif)
        elif synapse_module_type is HeterSynapsePSC:
            # 其他自定义突触（如 HeterSynapsePSC）：
            # 若未显式提供 n_neuron，则注入；否则尊重外部参数，避免重复传参
            if "n_neuron" not in synapse_args:
                #breakpoint()
                synapse = synapse_module_type(n_neuron=neuron_args["n_neuron"], **synapse_args)
            else:
                #breakpoint()
                synapse = synapse_module_type(**synapse_args)
        
        # elif synapse_module_type is HeteroDoubleExponentialPSC:
        #     # 使用HeteroDoubleExponentialPSC，传递tau_rise_matrix和tau_decay_matrix
        #     synapse = synapse_module_type(n_neuron=neuron_args["n_neuron"], **synapse_args)
        # elif synapse_module_type is HeteroDualExponentialPSC:
        #     # 使用HeteroDualExponentialPSC，传递tau_dual_rise_matrix和tau_dual_decay_matrix
        #     synapse = synapse_module_type(n_neuron=neuron_args["n_neuron"], **synapse_args)
        elif synapse_module_type is HeterSynapseDualPSC:
            if "n_neuron" not in synapse_args:
                synapse = synapse_module_type(n_neuron=neuron_args["n_neuron"], **synapse_args)
            else:
                synapse = synapse_module_type(**synapse_args)
        else:
            raise NotImplementedError(f"Unsupported synapse module type: {synapse_module_type}")

        #breakpoint()
            
        # 根据模块类型，稳定记录额外的状态（如神经元 ASC、E/I 分解后的电流）
        # GLIF3 的 after-spike current 对应 memory 名为 neuron.Iasc
        state_names = ["neuron.v", "neuron.Iasc", "synapse.psc"]
        # 不依赖 hasattr（MemoryModule 的 register_memory 可能不直接暴露为属性），
        # 对已知支持 E/I 拆分的实现，直接加入对应状态名
        if isinstance(synapse, HeterSynapsePSC) or isinstance(synapse, HeterSynapseDualPSC):
            state_names.extend(["synapse.psc_e", "synapse.psc_i"])

        state_names.extend(["synapse.psc_e_all", "synapse.psc_all"])

        #breakpoint()

        self.brain = RecurrentNN(
            neuron=neuron,
            synapse=synapse,
            update_state_names=tuple(state_names),
            step_mode="m",
        )
        self.input_layer = input_layer
        self.output_layer = output_layer

    def forward(self, observ: torch.Tensor | dict[str, Any]):
        #breakpoint()
        if self.input_layer is not None:
            inp = self.input_layer(observ) #[T, bs, n_neuron]
        else:
            inp = observ

        #breakpoint()
        print(f"device of inp: {inp.device}")
        #print(f"device of self.brain: {self.brain.device}")
        spike, states = self.brain(inp) #spike: [T, bs, n_neuron], states:['neuron.v', 'synapse.psc', 'synapse.psc_e', 'synapse.psc_i']
        #states['neuron.v']: [T, bs, n_neuron]
        #states['synapse.psc']: [T, bs, n_neuron]
        #states['synapse.psc_e']: [T, bs, n_neuron]
        #states['synapse.psc_i']: [T, bs, n_neuron]
        #breakpoint()

        if "synapse.psc_e" in states:
            #print(f"in forward, synapse.psc_e: {states['synapse.psc_e'].shape}")
            #print(f"in forward, inp: {inp.shape}")
            #breakpoint()
            # 注意：inp 的形状通常是 [T, B, N] 或 [B, N]，states 也是。
            # 直接相加即可，PyTorch 会自动处理广播
            states["synapse.psc_e_all"] = states["synapse.psc_e"] + inp

        # 2. 同时也应该更新总电流 psc，保持数据一致性
        if "synapse.psc" in states:
            #print(f"in forward, synapse.psc: {states['synapse.psc'].shape}")
            #print(f"in forward, inp: {inp.shape}")
            #breakpoint()
            states["synapse.psc_all"] = states["synapse.psc"] + inp

        states["synapse.I_ext"] = inp
        states["neuron.spike"] = spike

        if self.output_layer is not None:
            out = self.output_layer(states)
        else:
            out = None

        brain_out = unflatten_dict(states, dot=True)

        return out, brain_out
