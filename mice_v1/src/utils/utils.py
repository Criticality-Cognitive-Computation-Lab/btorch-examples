import argparse
import os
import json
import glob
import numpy as np
import scipy.sparse as sp
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

import torch

from btorch import connectome
from btorch.connectome import augment
from btorch.models import environ, functional, init, linear
from src.models import brain
from btorch.models.synapse import AlphaPSC, HeterSynapsePSC#, HeterSynapseDualPSC #, HeterogeneousAlphaPSC (最后这个是msy的版本，暂时不用), GLIFAlphaPSCFull, 
#from btorch.models.linear import GLIFSparseConn
from btorch.connectome.connection import make_hetersynapse_conn
#from btorch.models.benchmark import PerfTimer
from src.utils.preprocess import load_and_preprocess, load_and_preprocess_mice
from btorch.utils import dict_utils, hdf5_utils
from btorch.utils.yaml_utils import save_yaml

from src.models.glif_synapse import *

import ipdb

#from . import sim_vis
#from . import branching_ratio_analysis
#from . import glif_synapse
#from .utils import choose_device, choose_float_dtype

#load glif paras from json files
def load_glif_params_for_neurons(neurons: pd.DataFrame, glif_dir: str) -> dict[int, dict]:
    #要求neurons['cell_class']和'simple_id'存在
    assert 'cell_class' in neurons.columns, "neurons dataframe must contain 'cell_class' column"
    assert 'simple_id' in neurons.columns, "neurons dataframe must contain 'simple_id' column"

    # 预先为每个cell_class收集可用的json文件路径，避免逐行I/O
    unique_classes = neurons['cell_class'].dropna().unique()
    #print(f"unique_classes: {unique_classes}")
    class_to_files: dict[str, list[str]] = {}
    for cls in unique_classes:
        cls_dir = os.path.join(glif_dir, str(cls))
        files = sorted(glob.glob(os.path.join(cls_dir, '*.json')))
        if len(files) == 0:
            # 若该类没有可用文件，跳过；后续将报错以提醒
            class_to_files[cls] = []
        else:
            class_to_files[cls] = files

    # 为每个神经元随机选择一个对应cell_class下的json文件（分组采样以减少Python层循环）
    selected_paths = pd.Series(index=neurons.index, dtype=object)
    rng = np.random.default_rng()
    for cls, idx in neurons.groupby('cell_class').groups.items():
        files = class_to_files.get(cls, [])
        if len(files) == 0:
            raise FileNotFoundError(f"No GLIF json files found for cell_class '{cls}' under {glif_dir}")
        choices = rng.integers(0, len(files), size=len(idx))
        selected_paths.loc[idx] = np.asarray(files, dtype=object)[choices]

    # 去重后批量加载所需json，减少重复I/O
    unique_paths = pd.unique(selected_paths.values.astype(object))
    path_to_params: dict[str, dict] = {p: load_glif_params_from_json(p) for p in unique_paths}

    # 组装为 {simple_id: glif_params_dict}
    simple_ids = neurons['simple_id'].to_numpy()
    print(f"simple_ids[0]: {simple_ids[0]}")
    ret: dict[int, dict] = {
        int(sid): path_to_params[str(p)] for sid, p in zip(simple_ids, selected_paths.values)
    }
    return ret

def load_glif_params_from_json(json_file: str):
    with open(json_file, 'r') as f:
        glif_params = json.load(f)
    return glif_params

def convert_allen_glif_json_to_glif3_params(raw: dict) -> dict:
    """
    将Allen GLIF JSON (SI单位) 转换为GLIF3期望的单位 (ms, mV, pA)。

    返回与GLIF3构造函数兼容的字典:
      - v_rest (mV), v_reset (mV), v_threshold (mV)
      - c_m (pF), tau (ms), tau_ref (ms)
      - k (1/ms) shape (n_Iasc,), asc_amps (pA) shape (n_Iasc,)
    """
    coeffs = raw.get('coeffs', {})
    coeff_C = coeffs.get('C', 1.0)
    coeff_th = coeffs.get('th_inf', 1.0)
    coeff_asc = coeffs.get('asc_amp_array', [1.0 for _ in raw.get('asc_amp_array', [])])
    #print(f"coeff_asc: {coeff_asc}")

    # 电容: F -> pF
    C_F = float(raw.get('C', 0.0))
    c_m_pF = C_F * float(coeff_C) * 1e12

    # 时间常数: tau = R_input * C  (Ohm * F = s) -> ms
    R_input = float(raw.get('R_input', 0.0))
    tau_ms = R_input * C_F * 1000.0
    R_input = R_input * 1e-9

    # 不应期: spike_cut_length * (dt * dt_multiplier) -> s -> ms
    spike_cut_len = float(raw.get('spike_cut_length', 0.0))
    glif_dt = float(raw.get('dt', 5e-05))  # GLIF模型内部时间步长(秒)
    dt_mul = float(raw.get('dt_multiplier', 1.0))
    tau_ref_ms = spike_cut_len * glif_dt * 1000.0

    # 电压: 使用Allen GLIF的内部零基线作为静息电位，但El_reference需要在画图时使用
    El_reference = float(raw.get('El_reference', -0.07))  # V
    El_internal = float(raw.get('El', 0.0))  # V, 内部计算的基准电位
    th_inf = float(raw.get('th_inf', raw.get('init_threshold', 0.02)))

    El_reference = El_reference * 1000.0
    
    v_threshold_mV = th_inf * float(coeff_th) * 1000.0
    
    v_rest_mV = El_internal * 1000.0  # V -> mV
    v_reset_mV = El_internal * 1000.0  # V -> mV, 重置为与静息电位相同

    #voltage_scale = th_inf * 1000.0 #TODO: 可能要改成v_threshold_mV
    voltage_scale = v_threshold_mV

    #breakpoint()

    v_threshold_mV = v_threshold_mV / voltage_scale #==1.0
    El_internal = El_internal / voltage_scale #==0.0
    v_rest_mV = v_rest_mV / voltage_scale #==0.0
    v_reset_mV = v_reset_mV / voltage_scale #==0.0

    # 峰后电流
    asc_tau_array_s = raw.get('asc_tau_array', [])
    asc_amp_array_A = raw.get('asc_amp_array', [])
    
    # 确保是列表格式
    if isinstance(asc_tau_array_s, (int, float)):
        asc_tau_array_s = [float(asc_tau_array_s)]
    if isinstance(asc_amp_array_A, (int, float)):
        asc_amp_array_A = [float(asc_amp_array_A)]
    
    # 只取第一个成分/取全部成分
    if len(asc_tau_array_s) > 0 and len(asc_amp_array_A) > 0:
        # 取第一个时间常数和振幅 TODO
        # tau_s = float(asc_tau_array_s[0])
        # amp_A = float(asc_amp_array_A[0])
        
        # 取对应的系数（如果存在）
        #coeff = coeff_asc[0] if len(coeff_asc) > 0 else 1.0
        coeff = coeff_asc #两个都保留，是个列表
        
        # 计算k和asc_amps
        k_inv_ms = [0.0 if tau_s <= 0 else 1.0 / (tau_s * 1000.0) for tau_s in asc_tau_array_s] #asc_tau_array中的
        asc_amps_pA = [amp_A * float(coeff[i]) * 1e12/voltage_scale for i, amp_A in enumerate(asc_amp_array_A)]
    else:
        # 如果没有峰后电流参数，使用空列表
        k_inv_ms = [] #k_inv_ms的命名是k，inv_ms代表单位是ms。k_j=1/tau_j
        asc_amps_pA = []

    #breakpoint()

    return {
        'v_rest': v_rest_mV,
        'v_reset': v_reset_mV,
        'v_threshold': v_threshold_mV,
        'c_m': c_m_pF,
        'tau': tau_ms,
        'tau_ref': tau_ref_ms,
        'k': k_inv_ms,
        'asc_amps': asc_amps_pA,
        'voltage_scale': voltage_scale,
        'El_reference': El_reference,
    }

def load_neuron_args(network_config, neurons):
    n_neurons = len(neurons)

    if getattr(network_config, 'use_glif_params', True):
        if not getattr(network_config, 'glif_dir', None):
            raise ValueError("--use_glif_params is set but --glif_dir is not provided")

        # 1) select one json per neuron and load
        sid_to_raw = load_glif_params_for_neurons(neurons, network_config.glif_dir)
        #print(f"sid_to_raw first pair: {list(sid_to_raw.items())[0]}")
        # 2) convert units and collect lengths for ASC
        #不带开头_的是从test_glif.py中来的
        converted = {sid: convert_allen_glif_json_to_glif3_params(raw) for sid, raw in sid_to_raw.items()}
        max_Iasc = max((len(p['k']) for p in converted.values()), default=0)

        #打印converted的第一对key和value
        #print(f"converted first pair: {list(converted.items())[0]}")

        # 3) build per-neuron tensors
        order = neurons['simple_id'].to_numpy()
        v_rest = np.zeros(n_neurons, dtype=np.float32)
        v_reset = np.zeros(n_neurons, dtype=np.float32)
        v_th = np.zeros(n_neurons, dtype=np.float32)
        c_m = np.zeros(n_neurons, dtype=np.float32)
        tau = np.zeros(n_neurons, dtype=np.float32)
        tau_ref = np.zeros(n_neurons, dtype=np.float32)
        k = np.zeros((n_neurons, max_Iasc), dtype=np.float32)
        asc_amps = np.zeros((n_neurons, max_Iasc), dtype=np.float32)
        voltage_scale = np.zeros(n_neurons, dtype=np.float32)
        El_reference = np.zeros(n_neurons, dtype=np.float32)
        # 新增：初始化阈值电流数组
        I_thr = np.zeros(n_neurons, dtype=np.float32)  # <--- 新增
        # Fill using vectorized index mapping via array lookup (loop over unique params length unavoidable)
        sid_to_idx = {int(sid): idx for idx, sid in enumerate(order)}
        for sid, params in converted.items():
            i = sid_to_idx[int(sid)]
            v_rest[i] = params['v_rest']
            v_reset[i] = params['v_reset']
            v_th[i] = params['v_threshold']
            c_m[i] = params['c_m']
            tau[i] = params['tau']
            tau_ref[i] = params['tau_ref']
            ki = np.asarray(params['k'], dtype=np.float32)
            ai = np.asarray(params['asc_amps'], dtype=np.float32)
            k[i, : ki.shape[0]] = ki
            asc_amps[i, : ai.shape[0]] = ai
            voltage_scale[i] = params['voltage_scale']
            El_reference[i] = params['El_reference']

            # 新增：计算该神经元的阈值电流
            # 计算该神经元的物理阈值电流 (pA)
            # 这里的 params['v_threshold'] 是归一化后的值 (约为1.0)
            # DCInputLayer 会除以 voltage_scale，所以我们需要在这里乘以 voltage_scale 还原为物理单位
            #breakpoint()
            if params['tau'] > 1e-9:
                # 1. 计算模型内部归一化电流: (V_th_norm - V_rest_norm) * C_m / tau
                #breakpoint()
                I_internal_norm = (params['v_threshold'] - params['v_rest']) * params['c_m'] / params['tau']
                
                # 2. 还原为物理电流 (pA): I_physical = I_internal_norm * voltage_scale
                I_thr[i] = I_internal_norm * params['voltage_scale'] 
            else:
                I_thr[i] = 1e9

        #breakpoint()

        neuron_args = {
            "n_neuron": n_neurons,
            # use per-neuron parameters
            "v_rest": torch.from_numpy(v_rest),
            "v_reset": torch.from_numpy(v_reset),
            "v_threshold": torch.from_numpy(v_th),
            "c_m": torch.from_numpy(c_m),
            "tau": torch.from_numpy(tau),
            "tau_ref": torch.from_numpy(tau_ref),
            "k": torch.from_numpy(k),
            "asc_amps": torch.from_numpy(asc_amps),
            "hard_reset": True,  # match Allen GLIF reset to zero
            # Note: voltage_scale and El_reference are not passed to GLIF3,
            # they are only used for visualization
            # 新增：将计算好的阈值电流作为 tensor 传入（如果后续模型需要使用）
            # "I_thr": torch.from_numpy(I_thr), # <--- 新增
        }
        neuron_params = {
            "use_glif_params": True,
            "max_Iasc": int(max_Iasc),
            # expose per-neuron params for visualization/analysis
            "v_rest": v_rest.tolist(),
            "v_threshold": v_th.tolist(),
            "v_reset": v_reset.tolist(),
            "c_m": c_m.tolist(),
            "tau": tau.tolist(),
            "tau_ref": tau_ref.tolist(),
            "voltage_scale": voltage_scale.tolist(),
            "El_reference": El_reference.tolist(),
            # 新增：将阈值电流暴露给参数字典，方便后续统计（如 mean/std 计算）
            "I_thr": I_thr.tolist(), # <--- 新增
        }
        #breakpoint()
    else:
        neuron_args = {
            "n_neuron": n_neurons,
            **connectome.typical_params["neuron"],
            "c_m": network_config.c_m,
            "k": network_config.k,
            "asc_amps": network_config.asc_amps,
            "hard_reset": network_config.use_hard_reset,
            "gradient_scale": network_config.gradient_scale,
        }
        neuron_params = neuron_args.copy()
        neuron_args["c_m"] = (
            network_config.c_m
            if network_config.disable_empirical_c_m
            else network_config.c_m * augment.empirical_membrane_capacitance_mice(neurons)
        )
        neuron_params["disable_empirical_c_m"] = network_config.disable_empirical_c_m

    return neuron_args, neuron_params

def load_synapse_args(network_config, neurons, expanded_conn, receptor_df):
    # 根据GLIF模型要求设置不同类型的突触时间常数
    # 基于Arkhipov et al., 2018的L4模型数据
    #print(f"network_config: {network_config}")
    if network_config.type == 'alpha':
        print(f"entering alpha synapse times")
        tau_syn_ee = getattr(network_config, 'tau_syn_ee', 5.5)  # ms - excitatory to excitatory
        tau_syn_ie = getattr(network_config, 'tau_syn_ie', 8.5)  # ms - inhibitory to excitatory  
        tau_syn_ei = getattr(network_config, 'tau_syn_ei', 2.8)  # ms - excitatory to inhibitory
        tau_syn_ii = getattr(network_config, 'tau_syn_ii', 5.8)  # ms - inhibitory to inhibitory
    elif network_config.type == 'doubleexp':
        print(f"entering double exponential synapse times")
        tau_syn_rise_ee = getattr(network_config, 'tau_syn_rise_ee', 2.114)  # ms - excitatory to excitatory
        tau_syn_decay_ee = getattr(network_config, 'tau_syn_decay_ee', 2.124)  # ms - excitatory to excitatory
        tau_syn_rise_ie = getattr(network_config, 'tau_syn_rise_ie', 2.114)  # ms - inhibitory to excitatory
        tau_syn_decay_ie = getattr(network_config, 'tau_syn_decay_ie', 2.124)  # ms - inhibitory to excitatory
        tau_syn_rise_ei = getattr(network_config, 'tau_syn_rise_ei', 2.114)  # ms - excitatory to inhibitory
        tau_syn_decay_ei = getattr(network_config, 'tau_syn_decay_ei', 2.124)  # ms - excitatory to inhibitory
        tau_syn_rise_ii = getattr(network_config, 'tau_syn_rise_ii', 2.114)  # ms - inhibitory to inhibitory
        tau_syn_decay_ii = getattr(network_config, 'tau_syn_decay_ii', 2.124)  # ms - inhibitory to inhibitory
        latency_ee = getattr(network_config, 'latency_ee', 2.25)  # ms - latency
        latency_ie = getattr(network_config, 'latency_ie', 2.25)  # ms - latency
        latency_ei = getattr(network_config, 'latency_ei', 2.25)  # ms - latency
        latency_ii = getattr(network_config, 'latency_ii', 2.25)  # ms - latency
    else:
        # 使用默认的统一时间常数
        tau_syn_ee = tau_syn_ie = tau_syn_ei = tau_syn_ii = connectome.typical_params["tau_syn"]
    
    # 使用GLIF突触模型创建突触时间常数
    n_neurons = len(neurons)
    
    # 确定神经元类型（兴奋性/抑制性）
    if 'cell_class' in neurons.columns:
        #print(f"all cell class: {neurons['cell_class'].unique()}")
        #含有pyr，et或it的都算excitatory
        # 根据cell_class设置神经元类型
        excitatory_mask = neurons['cell_class'].str.contains('pyr|et|it', case=False, na=False) 
        #用正则匹配是否包含任一子串 'pyr'、'et'、'it'（不区分大小写）；缺失值按 False 处理。
        #返回 pd.Series[bool]，True 表示该行的 cell_class 含上述任一关键词
        
        # 如果没有明确的类型标识，使用默认的E/I比例
        if not excitatory_mask.any():
            print("Warning: No excitatory neurons found, using default E/I ratio")
            n_excitatory = int(0.8 * n_neurons)
            excitatory_mask = np.zeros(n_neurons, dtype=bool)
            excitatory_mask[:n_excitatory] = True
    else:
        # 如果没有cell_class列，使用默认的E/I比例
        print("Warning: No cell_class column found, using default E/I ratio")
        n_excitatory = int(0.8 * n_neurons)
        excitatory_mask = np.zeros(n_neurons, dtype=bool)
        excitatory_mask[:n_excitatory] = True

    if network_config.type == 'alpha':
        impl = getattr(network_config, 'synapse_impl', 'hetersyn')
        if impl not in ("hetersyn"):
            raise NotImplementedError(f"Unsupported synapse implementation: {impl}")
        
        elif impl == "hetersyn":
            #expanded_conn = expanded_conn.transpose()
            expanded_conn = expanded_conn.tocsr().astype(np.float32)
            # 创建GLIF突触时间常数
            tau_syn_matrix, tau_syn_array = create_glif_synapse_times(
                n_neurons, excitatory_mask, tau_syn_ee, tau_syn_ie, tau_syn_ei, tau_syn_ii
            )
            # 分析突触时间常数分布
            synapse_analysis = analyze_synapse_times(tau_syn_matrix, excitatory_mask)
            
            # 打印突触时间常数设置信息
            # print(f"\n突触时间常数设置 (基于GLIF模型):")
            # print(f"  E→E: {tau_syn_ee:.1f} ms (n={synapse_analysis['n_ee']})")
            # print(f"  I→E: {tau_syn_ie:.1f} ms (n={synapse_analysis['n_ie']})") 
            # print(f"  E→I: {tau_syn_ei:.1f} ms (n={synapse_analysis['n_ei']})")
            # print(f"  I→I: {tau_syn_ii:.1f} ms (n={synapse_analysis['n_ii']})")
            # print(f"  平均时间常数: {synapse_analysis['overall_mean']:.1f} ± {synapse_analysis['overall_std']:.1f} ms")


            # 受体维度展开法：构建 (pre, post * n_receptor_pair) 的连接
            # 需要 neurons['EI'] 存在；若不存在则由 excitatory_mask 自动生成
            if 'EI' not in neurons.columns:
                neurons = neurons.copy()
                neurons['EI'] = np.where(excitatory_mask, 'E', 'I')
            
            # 受体对数量（索引从0开始，连续）
            n_receptor = int(receptor_df['receptor_index'].max()) + 1

            # 为每个 receptor_index 指定 τ：根据 (pre_type, post_type) 选择值
            if {'pre_receptor_type', 'post_receptor_type'}.issubset(receptor_df.columns):
                idx_to_pair = receptor_df.sort_values('receptor_index')[['pre_receptor_type', 'post_receptor_type']].to_numpy()
            else:
                # 兼容连接模式下的列名（不太可能走到这里，因为上面指定了 neuron 模式）
                raise ValueError("receptor mapping missing expected columns")

            pair_to_tau = {
                ('E', 'E'): float(tau_syn_ee),
                ('E', 'I'): float(tau_syn_ei),
                ('I', 'E'): float(tau_syn_ie),
                ('I', 'I'): float(tau_syn_ii),
            }
            tau_per_index = [pair_to_tau[(str(p[0]), str(p[1]))] for p in idx_to_pair]
            # 展开为 [post * n_receptor]，每个 post 复制一遍 receptor 顺序
            tau_vec = torch.as_tensor(tau_per_index, dtype=torch.float32).repeat(n_neurons)
            #print(f"expanded_conn shape: {expanded_conn.shape}")
            # 标注每个 receptor_index 是否来自兴奋性突触（由 pre_receptor_type 决定）
            receptor_is_exc = torch.as_tensor([str(p[0]) == 'E' for p in idx_to_pair], dtype=torch.bool)
            # 目标: 对每一列应用符号 (E->正, I->负)
            # 矩阵列布局: [P0_R0, P0_R1... P0_R3, P1_R0...]
            # 符号模式: receptor_is_exc_np (长度R) 重复 N 次
            signs_pattern = np.where(receptor_is_exc, 1.0, -1.0) # [1, 1, -1, -1]
            #breakpoint()
            signs_all = np.tile(signs_pattern, n_neurons)           # [1, 1, -1, -1, 1, 1, ...] (长度 5372)
            #breakpoint()
            
            # 应用符号 (multiply 支持广播: (M,N) * (1,N) -> 列缩放)
            # 注意：scipy sparse multiply 接受 (1, N) 数组作为对列的操作
            expanded_conn = expanded_conn.multiply(signs_all[np.newaxis, :])
            #breakpoint()
            print("DEBUG: Signs injected into Alpha matrix (columns scaled).")
            synapse_args = {
                "n_neuron": n_neurons,
                "n_receptor": n_receptor,
                "receptor_type_index": receptor_df,
                "linear": linear.SparseConn(expanded_conn),
                #"linear":torch.nn.Linear(expanded_conn.shape[0], expanded_conn.shape[1]),
                "base_psc": AlphaPSC,  # 使用 AlphaPSC 一致公式
                "tau_syn": tau_vec,
                "g_max": 1.0,
                "receptor_is_exc": receptor_is_exc,
            }
            synapse_module_type = HeterSynapsePSC

    elif network_config.type == 'doubleexp':
        impl = getattr(network_config, 'synapse_impl', 'hetersyndual')
        if impl not in ("hetersyndual"):
            raise NotImplementedError(f"Unsupported synapse implementation: {impl}")
        if impl == "hetersyndual":
            expanded_conn = expanded_conn.tocsr().astype(np.float32)
            #expanded_conn = linear.SparseConn(expanded_conn)
            print(f"DEBUG: Transposed matrix for SparseConn shape: {expanded_conn.shape}") # 应该是 (5372, 1343)
            # ================= [核心修复 END] ===================

            # 创建GLIF突触时间常数
            tau_dual_rise_matrix, tau_dual_decay_matrix, latency_matrix = create_double_exponential_synapse_times(
                n_neurons, excitatory_mask, tau_syn_rise_ee, tau_syn_rise_ie, tau_syn_rise_ei, tau_syn_rise_ii, tau_syn_decay_ee, tau_syn_decay_ie, tau_syn_decay_ei, tau_syn_decay_ii, latency_ee, latency_ie, latency_ei, latency_ii
            )

            # 受体维度展开法：构建 (pre, post * n_receptor_pair) 的连接
            # 需要 neurons['EI'] 存在；若不存在则由 excitatory_mask 自动生成
            if 'EI' not in neurons.columns:
                neurons = neurons.copy()
                neurons['EI'] = np.where(excitatory_mask, 'E', 'I')
            
            # 受体对数量（索引从0开始，连续）
            n_receptor = int(receptor_df['receptor_index'].max()) + 1

            # 为每个 receptor_index 指定 τ：根据 (pre_type, post_type) 选择值
            if {'pre_receptor_type', 'post_receptor_type'}.issubset(receptor_df.columns):
                idx_to_pair = receptor_df.sort_values('receptor_index')[['pre_receptor_type', 'post_receptor_type']].to_numpy()
                # array([['E', 'E'],
                #     ['E', 'I'],
                #     ['I', 'E'],
                #     ['I', 'I']], dtype=object)
            else:
                # 兼容连接模式下的列名（不太可能走到这里，因为上面指定了 neuron 模式）
                raise ValueError("receptor mapping missing expected columns")

            pair_to_rise_tau = {
                ('E', 'E'): float(tau_syn_rise_ee),
                ('E', 'I'): float(tau_syn_rise_ie),
                ('I', 'E'): float(tau_syn_rise_ei),
                ('I', 'I'): float(tau_syn_rise_ii),
            }
            pair_to_decay_tau = {
                ('E', 'E'): float(tau_syn_decay_ee),
                ('E', 'I'): float(tau_syn_decay_ie),
                ('I', 'E'): float(tau_syn_decay_ei),
                ('I', 'I'): float(tau_syn_decay_ii),
            }
            #breakpoint()
            tau_rise_per_index = [pair_to_rise_tau[(str(p[0]), str(p[1]))] for p in idx_to_pair]
            tau_decay_per_index = [pair_to_decay_tau[(str(p[0]), str(p[1]))] for p in idx_to_pair]
            # 展开为 [post * n_receptor]，每个 post 复制一遍 receptor 顺序
            tau_rise_vec = torch.as_tensor(tau_rise_per_index, dtype=torch.float32).repeat(n_neurons)
            tau_decay_vec = torch.as_tensor(tau_decay_per_index, dtype=torch.float32).repeat(n_neurons)
            #print(f"expanded_conn shape: {expanded_conn.shape}")
            # 标注每个 receptor_index 是否来自兴奋性突触（由 pre_receptor_type 决定）
            receptor_is_exc = torch.as_tensor([str(p[0]) == 'E' for p in idx_to_pair], dtype=torch.bool)
            # 目标: 对每一列应用符号 (E->正, I->负)
            # 矩阵列布局: [P0_R0, P0_R1... P0_R3, P1_R0...]
            # 符号模式: receptor_is_exc_np (长度R) 重复 N 次
            signs_pattern = np.where(receptor_is_exc, 1.0, -1.0) # [1, 1, -1, -1]
            #breakpoint()
            signs_all = np.tile(signs_pattern, n_neurons)           # [1, 1, -1, -1, 1, 1, ...] (长度 5372)
            # 对 (1343, 5372) 的矩阵进行列缩放
            expanded_conn = expanded_conn.multiply(signs_all[np.newaxis, :])
            print("DEBUG: Signs injected into DualExp matrix (columns scaled).")
            synapse_args = {
                "n_neuron": n_neurons,
                "n_receptor": n_receptor,
                "linear": linear.SparseConn(expanded_conn),
                #"linear":torch.nn.Linear(expanded_conn.shape[0], expanded_conn.shape[1]),
                #"base_psc": AlphaPSC,  # 使用 AlphaPSC 一致公式
                "tau_rise": tau_rise_vec,
                "tau_decay": tau_decay_vec,
                "latency": latency_matrix[0][0],
                "receptor_is_exc": receptor_is_exc,
            }
            synapse_module_type = HeterSynapseDualPSC
    else:
        print("Warning: Using AlphaPSC")
        raise NotImplementedError("Traditional AlphaPSC is not implemented")
        # # 使用传统AlphaPSC，传递平均时间常数
        # synapse_args = {
        #     "tau_syn": torch.from_numpy(tau_syn_array),
        #     "linear": linear.SparseConn(conn_mat),
        # }
        # synapse_module_type = brain.AlphaPSC
    
        # synapse_params = {
        #     "tau_syn_ee": tau_syn_ee,
        #     "tau_syn_ie": tau_syn_ie,
        #     "tau_syn_ei": tau_syn_ei,
        #     "tau_syn_ii": tau_syn_ii,
        #     "tau_syn_matrix": tau_syn_matrix.tolist(),  # 保存完整矩阵用于分析
        #     "synapse_analysis": synapse_analysis,  # 保存分析结果
        #     "weight_scale": network_config.weight_scale,
        #     #"ie_rebalance_ratio": network_config.ie_rebalance_ratio,
        #     "use_alpha_synapse_times": getattr(network_config, 'use_alpha_synapse_times', True),
        # }

    return synapse_args, synapse_module_type #主要用的似乎是synapse_args


#def load_microns_connectome(microns_connectome_path):
    #该函数对应于from fly_unnamed.sim import load_and_preprocess_mice功能，都保存并返回的是neurons, conn_mats, connections，保存为csv
    #从microns h5文件中直接切取不同大小的column，包含neuron type等信息，保存为csv.gz

    #neuron.csv.gz:
#        root_id layer        type  x_position  y_position  z_position  simple_id cell_class EI
# 0     2548   L23  Excitatory   2330.2764   3072.5786   2601.5989          0     l23pyr  E
# 1     2829   L23  Excitatory   2370.1384   3128.2993   2629.4683          1     l23pyr  E
# 2     3515   L23  Excitatory   2396.0210   2860.6125   2683.4084          2     l23pyr  E
# 3     4497   L23  Excitatory   2315.5415   3151.9329   2660.4956          3     l23pyr  E
# 4     4553   L23  Excitatory   2301.4976   3086.3730   2608.7458          4     l23pyr  E

    #connections.csv.gz:
    #['pre_id', 'post_id', 'pre_type', 'post_type', 'pre_simple_id', 'post_simple_id', 'pre_position', 'post_position', 'syn_count', 'EI']
#    pre_id  post_id pre_type post_type  ...                       pre_position                      post_position syn_count EI
# 0     253      270     REST      REST  ...   (2367.9788, 3018.436, 2667.1328)   (2319.5432, 3026.5056, 2689.148)         1  I

    



# def build_rec_layer_from_conn(connectome_data, network_config):
#     neurons, conn_mats, connections = connectome_data
#     neuron_args, neuron_params = load_neuron_args(network_config, neurons)
#     conn_mat = init_weights(network_config, neurons, conn_mats, connections, neuron_params)
#     # 提取所有神经元的阈值电流
#     all_I_thr = np.array(neuron_params["I_thr"])

#     # 计算统计量
#     network_mean_threshold = np.mean(all_I_thr)
#     network_std_threshold = np.std(all_I_thr)

#     print(f"Network I_thr Mean: {network_mean_threshold:.4e} pA")
#     print(f"Network I_thr Std:  {network_std_threshold:.4e} pA")
#     #breakpoint()
#     synapse_args, synapse_params, synapse_module_type = load_synapse_args(network_config, neurons, conn_mat)
#     #breakpoint()
#     model = brain.FlyBrain(
#         synapse_args=synapse_args, 
#         neuron_args=neuron_args,
#         synapse_module_type=synapse_module_type
#     )
#     params = {
#         "neuron": neuron_params,
#         "synapse": synapse_params,
#     }
#     return model, params