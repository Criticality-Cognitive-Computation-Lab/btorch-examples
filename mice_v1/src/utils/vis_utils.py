import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse

from btorch import visualisation
from .multitraces import plot_population_by_field, plot_population_by_field_single_plot
from btorch.analysis.metrics import select_on_metric
from btorch.analysis.voltage import voltage_overshoot
from src.utils.other import simple_id_to_root_id_func

import ipdb


def prepare_data_from_dict(result, model_param, cfg, connectome_data):
    """Prepare data from result dictionary for analysis."""
    time_points = np.arange(cfg.simulation.T)# * cfg.simulation.dt
    #T的含义是时间步，不是单位秒的时间，time_points也是时间步的索引

    return {
        "result": result,
        "model_param": model_param,
        "cfg": cfg,
        "connectome_data": connectome_data,
        "time_points": time_points,
    }

def select_neurons_for_visualization(
    result,
    model_param,
    cfg,
    connectome_data,
    top_k=60,
    final_selection_count=24,
):
    """Select neurons for visualization based on metrics or provided indices.
    
    Returns:
        tuple: (selected_neuron_indices, neuron_info_df)
            - selected_neuron_indices: numpy array of selected neuron indices
            - neuron_info_df: pandas DataFrame containing information about selected neurons
    """

    #here
    # Select neurons based on voltage overshoot metric
    v_rest_ref = model_param["neuron"].get("v_rest")
    v_th_ref = model_param["neuron"].get("v_threshold")
    if isinstance(v_rest_ref, (list, tuple, np.ndarray)):
        v_rest_ref = float(np.nanmean(v_rest_ref))
    if isinstance(v_th_ref, (list, tuple, np.ndarray)):
        v_th_ref = float(np.nanmean(v_th_ref))

    overshoot_metric = voltage_overshoot(
        result["neuron"]["v"],
        mode="mse_threshold",
        V_reset=v_rest_ref,
        V_th=v_th_ref,
    )
    selected_neurons = select_on_metric(overshoot_metric, mode="any", num=top_k)
    selected_neurons = selected_neurons[-final_selection_count:]

    # 获取神经元信息
    neuron_df = connectome_data[0]
    input_indices = selected_neurons
    
    # 创建选中神经元的信息DataFrame
    selected_info = []
    for idx in selected_neurons:
        neuron_data = neuron_df.iloc[idx]
        is_input = idx in input_indices
        
        info = {
            'neuron_index': idx,
            'root_id': neuron_data.name,  # 假设index是root_id
            'cell_class': neuron_data.get('cell_class', 'Unknown'),
            'layer': neuron_data.get('layer', 'Unknown'),
            'is_input': is_input
        }
        selected_info.append(info)
    
    info_df = pd.DataFrame(selected_info)
    
    '''
    # 打印选中神经元的信息
    print("\n选中的神经元信息:")
    print("-" * 80)
    for _, row in info_df.iterrows():
        input_status = "是输入神经元" if row['is_input'] else "非输入神经元"
        print(f"神经元 {row['neuron_index']} (root_id: {row['root_id']}):")
        print(f"    类型: {row['cell_class']}")
        print(f"    层级: {row['layer']}")
        print(f"    {input_status}")
        print("-" * 40)
    '''

    return selected_neurons, info_df

def plot_voltage(result, model_param, cfg, selected_neurons, root_ids, cell_classes=None, epoch_figure_dir=None):
    """Plot voltage responses for selected neurons."""
    figname = "voltage_responses"
    # Prepare reference threshold line: handle per-neuron arrays or scalar
    vth = model_param["neuron"].get("v_threshold")
    voltage_scale = model_param["neuron"].get("voltage_scale")
    El_reference = model_param["neuron"].get("El_reference")
    if isinstance(vth, (list, tuple, np.ndarray)):
        vth_arr = np.asarray(vth)
        try:
            vth_ref = vth_arr[selected_neurons]
        except Exception:
            vth_ref = float(np.nanmean(vth_arr))
    else:
        vth_ref = vth

    # Apply voltage scaling to reference line as well
    if voltage_scale is not None and El_reference is not None:
        # Convert to numpy arrays for indexing
        voltage_scale = np.asarray(voltage_scale)
        El_reference = np.asarray(El_reference)
        
        if hasattr(voltage_scale, '__len__') and len(voltage_scale) > 1:
            v_scale = voltage_scale[selected_neurons]
        else:
            v_scale = voltage_scale
        if hasattr(El_reference, '__len__') and len(El_reference) > 1:
            el_ref = El_reference[selected_neurons]
        else:
            el_ref = El_reference
        vth_ref = vth_ref * v_scale + el_ref
    
    fig = plot_population_by_field(
        result["neuron"],
        "v",
        name="Voltage (mV)",
        population=selected_neurons,
        ref_line=vth_ref,
        ref_name="V_th",
        plot_args={"color": "b"},
        title_format=lambda arg: f"[{arg[0]}] = {arg[1]}, neuron {root_ids[arg[0]]}" + (f" ({cell_classes[arg[0]]})" if cell_classes is not None else ""),
        voltage_scale=voltage_scale if voltage_scale is not None else None,
        El_reference=El_reference if El_reference is not None else None,
    )
    fig.savefig(f"{epoch_figure_dir}/{figname}.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, _ = plot_population_by_field_single_plot(
        result["neuron"],
        "v",
        name="Voltage (mV)",
        population=selected_neurons,
        ref_line=vth_ref,
        ref_name="V_th",
        legend_format=lambda arg: f"{root_ids[arg[0]]}",
        voltage_scale=voltage_scale if voltage_scale is not None else None,
        El_reference=El_reference if El_reference is not None else None,
    )
    fig[0].savefig(f"{epoch_figure_dir}/{figname}_single.pdf", bbox_inches="tight")
    plt.close(fig[0])
    return fig

def plot_post_synapse_current(result, model_param, cfg, selected_neurons, root_ids, cell_classes=None, connectome_data=None, time_points=None, epoch_figure_dir=None, time_range_ms=None):
    """Plot post-synapse current responses for selected neurons.
    同时分别绘制总PSC、Excitatory PSC (Epsc) 与 Inhibitory PSC (Ipsc)。
    并在群体层面绘制每个cell_class的PSC/E/I的总和或均值。
    另外：仿照 plot_voltage，在绘图时支持 voltage_scale 与 El_reference 的缩放与偏移。
    """
    #breakpoint()
    times = time_points
    if times is None:
        times = np.arange(cfg.simulation.T) * cfg.simulation.dt

    #breakpoint()
    # 可选：时间段裁剪（单位 ms），例如 time_range_ms=(200, None) 表示只画 200ms 之后
    if time_range_ms is not None:
        t_start, t_end = time_range_ms
        # 兼容 hydra/yaml 里把 None 写成字符串的情况（例如 [200, 'None']）
        if isinstance(t_start, str) and t_start.lower() in {"none", "null"}:
            t_start = None
        if isinstance(t_end, str) and t_end.lower() in {"none", "null"}:
            t_end = None
        times_np = np.asarray(times)
        mask = np.ones(times_np.shape[0], dtype=bool)
        if t_start is not None:
            mask &= times_np >= t_start
        if t_end is not None:
            mask &= times_np <= t_end
        times = times_np[mask]

        #breakpoint()

        def _slice_time(x):
            #breakpoint()
            if x is None:
                return None
            # torch tensor
            if hasattr(x, "dim") and callable(getattr(x, "dim")):
                if x.dim() >= 1 and x.shape[0] == mask.shape[0]:
                    return x[mask]
                return x
            # numpy array
            if isinstance(x, np.ndarray):
                if x.ndim >= 1 and x.shape[0] == mask.shape[0]:
                    return x[mask]
                return x
            return x

        # 同步裁剪 synapse 中所有以 T 为第 0 维的数据
        if "synapse" in result and isinstance(result["synapse"], dict):
            for k, v in list(result["synapse"].items()):
                result["synapse"][k] = _slice_time(v)

    # 读取电压缩放与参考偏置（若存在）
    voltage_scale = model_param["neuron"].get("voltage_scale")
    El_reference = None
    # 构造局部synapse视图，加入 |Ipsc| 字段便于绘图
    syn = result.get("synapse", {})
    #breakpoint()
    syn_view = dict(syn)
    if "psc_i" in syn:
        #try:
        syn_view["psc_i_abs"] = syn["psc_i"].abs()
        # except Exception:
        #     breakpoint()
        #     raise Exception("psc_i is not in synapse")

    # 个体：总PSC
    fig_total = plot_population_by_field(
        syn_view,
        "psc",
        population=selected_neurons,
        name="post-synapse I_total (pA)",
        plot_args={"color": "purple"},
        title_format=lambda arg: f"[{arg[0]}] = {arg[1]}, neuron {root_ids[arg[0]]}" + (f" ({cell_classes[arg[0]]})" if cell_classes is not None else ""),
        voltage_scale=voltage_scale if voltage_scale is not None else None,
        El_reference=El_reference if El_reference is not None else None,
    )
    fig_total.savefig(
        f"{epoch_figure_dir}/post_synapse_current_responses.pdf", bbox_inches="tight"
    )
    plt.close(fig_total)

    # E 分量
    if "psc_e" in result.get("synapse", {}):
        fig_e = plot_population_by_field(
            result["synapse"],
            "psc_e",
            population=selected_neurons,
            name="post-synapse I_E (pA)",
            plot_args={"color": "r"},
            title_format=lambda arg: f"[{arg[0]}] = {arg[1]}, neuron {root_ids[arg[0]]}" + (f" ({cell_classes[arg[0]]})" if cell_classes is not None else ""),
            voltage_scale=voltage_scale if voltage_scale is not None else None,
            El_reference=El_reference if El_reference is not None else None,
        )
        fig_e.savefig(
            f"{epoch_figure_dir}/post_synapse_E_current_responses.pdf", bbox_inches="tight"
        )
        plt.close(fig_e)
    else:
        fig_e = None

    # I 分量（取绝对值）
    if "psc_i_abs" in syn_view:
        fig_i = plot_population_by_field(
            syn_view,
            "psc_i_abs",
            population=selected_neurons,
            name="post-synapse I_I |abs| (pA)",
            plot_args={"color": "b"},
            title_format=lambda arg: f"[{arg[0]}] = {arg[1]}, neuron {root_ids[arg[0]]}" + (f" ({cell_classes[arg[0]]})" if cell_classes is not None else ""),
            voltage_scale=voltage_scale if voltage_scale is not None else None,
            El_reference=El_reference if El_reference is not None else None,
        )
        fig_i.savefig(
            f"{epoch_figure_dir}/post_synapse_I_current_responses.pdf", bbox_inches="tight"
        )
        plt.close(fig_i)
    else:
        fig_i = None

    # I 分量（带符号，额外保留）
    if "psc_i" in syn:
        fig_i_signed = plot_population_by_field(
            syn,
            "psc_i",
            population=selected_neurons,
            name="post-synapse I_I (pA)",
            plot_args={"color": "b"},
            title_format=lambda arg: f"[{arg[0]}] = {arg[1]}, neuron {root_ids[arg[0]]}" + (f" ({cell_classes[arg[0]]})" if cell_classes is not None else ""),
            voltage_scale=voltage_scale if voltage_scale is not None else None,
            El_reference=El_reference if El_reference is not None else None,
        )
        fig_i_signed.savefig(
            f"{epoch_figure_dir}/post_synapse_I_current_responses_signed.pdf", bbox_inches="tight"
        )
        plt.close(fig_i_signed)

    # 合并图：每个子图三条曲线（总/E/I(abs)）
    # try:
    syn = result.get("synapse", {})
    series_list = []
    color_list = []
    # 顺序：总、E、I(abs)
    if "psc" in syn_view:
        series_list.append(syn_view["psc"]) ; color_list.append({"color": "purple", "label": "psc"})
    if "psc_e" in syn_view:
        series_list.append(syn_view["psc_e"]) ; color_list.append({"color": "r", "label": "Epsc"})
    if "psc_i_abs" in syn_view:
        series_list.append(syn_view["psc_i_abs"]) ; color_list.append({"color": "b", "label": "Ipsc"})
    
    series_list = [series.cpu().numpy() for series in series_list]
    if len(series_list) >= 2:
        # 叠第三维
        data_combo = np.stack(series_list, axis=-1)  # [T, N, K]
        res_combo = {"psc_combo": data_combo}
        fig_combo = plot_population_by_field(
            res_combo,
            "psc_combo",
            name="post-synapse I (total/E/|I|)",
            population=selected_neurons,
            plot_args=color_list,
            title_format=lambda arg: f"[{arg[0]}] = {arg[1]}, neuron {root_ids[arg[0]]}" + (f" ({cell_classes[arg[0]]})" if cell_classes is not None else ""),
            voltage_scale=voltage_scale if voltage_scale is not None else None,
            El_reference=El_reference if El_reference is not None else None,
        )
        fig_combo.savefig(
            f"{epoch_figure_dir}/post_synapse_current_combined.pdf", bbox_inches="tight"
        )
        plt.close(fig_combo)
    
    # 合并图（带符号I）：总/E/I(signed)
    if "psc" in syn and "psc_e" in syn and "psc_i" in syn:
        series_list_signed = [syn["psc"], syn["psc_e"], syn["psc_i"]]
        color_list_signed = [
            {"color": "purple", "label": "psc"},
            {"color": "r", "label": "Epsc"},
            {"color": "b", "label": "Ipsc"},
        ]
        series_list_signed = [series.cpu().numpy() for series in series_list_signed]
        data_combo_signed = np.stack(series_list_signed, axis=-1)
        res_combo_signed = {"psc_combo_signed": data_combo_signed}
        fig_combo_signed = plot_population_by_field(
            res_combo_signed,
            "psc_combo_signed",
            name="post-synapse I (total/E/I)",
            population=selected_neurons,
            plot_args=color_list_signed,
            title_format=lambda arg: f"[{arg[0]}] = {arg[1]}, neuron {root_ids[arg[0]]}" + (f" ({cell_classes[arg[0]]})" if cell_classes is not None else ""),
            voltage_scale=voltage_scale if voltage_scale is not None else None,
            El_reference=El_reference if El_reference is not None else None,
        )
        fig_combo_signed.savefig(
            f"{epoch_figure_dir}/post_synapse_current_combined_signed.pdf", bbox_inches="tight"
        )
        plt.close(fig_combo_signed)
    # except Exception as e:
    #     print(f"⚠️ 绘制合并PSC图失败: {e}")

    # ------ 群体层面：按 cell_class 聚合（总和或均值），合并到两张大图（abs版与signed版） ------
    # try:
    #进行群体层面聚合时，要先取出第一个batch
    for key, value in syn.items():
        syn[key]=value[:,0,:].cpu()
    #breakpoint()

    neuron_df = connectome_data[0]
    classes = neuron_df['cell_class'].unique() if 'cell_class' in neuron_df.columns else []
    if len(classes) == 0:
        return fig_total
    agg_mode = getattr(cfg, 'psc_population_agg', 'sum')  # 'sum' or 'mean'

    # 带电压缩放的聚合函数：先对每个神经元应用缩放（乘法），再进行sum/mean
    # 注意：PSC数据只需要乘以voltage_scale，不需要加减El_reference
    def _agg(arr, idx):
        if arr is None:
            return None
        if arr.ndim == 2:
            sub = arr[:, idx]  # [T, n_group]
            # 应用 per-neuron 缩放（仅乘法，不加减偏置）
            if voltage_scale is not None:
                vs = np.asarray(voltage_scale)
                try:
                    if vs.ndim == 0:
                        vs_sel = vs
                    else:
                        vs_sel = vs[idx]
                except Exception:
                    vs_sel = vs
                # 广播到 [T, n_group]
                sub = sub * vs_sel
            if agg_mode == 'mean':
                return np.nanmean(sub, axis=1)
            else:
                return np.nansum(sub, axis=1)
        return None

    import os
    os.makedirs(epoch_figure_dir, exist_ok=True)

    n_cls = len(classes)
    # 绝对值版本（|I|）的大图
    fig_abs, axes_abs = plt.subplots(n_cls, 1, figsize=(12, max(2.8*n_cls, 4)), sharex=True)
    if n_cls == 1:
        axes_abs = [axes_abs]
    # 带符号版本（I）的⼤图
    fig_sig, axes_sig = plt.subplots(n_cls, 1, figsize=(12, max(2.8*n_cls, 4)), sharex=True)
    if n_cls == 1:
        axes_sig = [axes_sig]

    for i, cls in enumerate(classes):
        #breakpoint()
        idx = np.where(neuron_df['cell_class'].values == cls)[0]
        if idx.size == 0:
            continue
        p_total = _agg(syn.get('psc', None), idx)
        p_e = _agg(syn.get('psc_e', None), idx)
        p_i_signed = _agg(syn.get('psc_i', None), idx)
        p_i_abs = _agg(np.abs(syn['psc_i'].cpu().numpy()) if 'psc_i' in syn else None, idx)

        # abs版：PSC / E / |I|
        ax = axes_abs[i]
        if p_total is not None:
            ax.plot(times, p_total, 'purple', label='PSC (total)')
        if p_e is not None:
            ax.plot(times, p_e, 'r-', label='Epsc')
        if p_i_abs is not None:
            ax.plot(times, p_i_abs, 'b-', label='|Ipsc|')
        ax.set_title(f"Group {cls} population PSC ({agg_mode})", fontsize=10)
        ax.set_ylabel('pA')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8, ncol=3, frameon=True)

        #breakpoint()

        # signed版：PSC / E / I
        ax2 = axes_sig[i]
        if p_total is not None:
            ax2.plot(times, p_total, 'purple', label='PSC (total)')
        if p_e is not None:
            ax2.plot(times, p_e, 'r-', label='Epsc')
        if p_i_signed is not None:
            ax2.plot(times, p_i_signed, 'b-', label='Ipsc (signed)')
        ax2.set_title(f"Group {cls} population PSC (signed I, {agg_mode})", fontsize=10)
        ax2.set_ylabel('pA')
        ax2.grid(True, alpha=0.3)
        if i == 0:
            ax2.legend(loc='upper right', fontsize=8, ncol=3, frameon=True)

        #breakpoint()

    axes_abs[-1].set_xlabel('Time (ms)')
    axes_sig[-1].set_xlabel('Time (ms)')
    fig_abs.tight_layout()
    fig_sig.tight_layout()
    fig_abs.savefig(f"{epoch_figure_dir}/population_psc_abs_{agg_mode}.pdf", bbox_inches='tight')
    fig_sig.savefig(f"{epoch_figure_dir}/population_psc_signed_{agg_mode}.pdf", bbox_inches='tight')
    plt.close(fig_abs)
    plt.close(fig_sig)

    # except Exception as e:
    #     print(f"⚠️ 绘制群体PSC图失败: {e}")

    return fig_total

def plot_fr_spectrum(result, model_param, cfg, epoch_figure_dir):
    """Plot population firing-rate spectrum using visualisation.plot_spectrum."""
    #breakpoint()
    spikes = result["neuron"]["spike"]
    if hasattr(spikes, "detach"):
        spikes = spikes.detach().cpu().numpy()
    else:
        spikes = np.asarray(spikes)

    #breakpoint()

    # Support [T, B, N] and [T, N]
    if spikes.ndim == 3:
        spikes = spikes[:, 0, :]
    elif spikes.ndim != 2:
        raise ValueError(f"Unexpected spike shape for spectrum plot: {spikes.shape}")

    # Population firing rate time series (Hz)
    pop_rate_hz = spikes.mean(axis=1) * (1000.0 / float(cfg.simulation.dt))
    #breakpoint()

    fig, ax = plt.subplots(figsize=(6, 4))
    visualisation.plot_spectrum(
        pop_rate_hz,
        dt=float(cfg.simulation.dt) / 1000.0,
        ax=ax,
        mode="loglog",
        show_mean=False,
        title="Population Firing-Rate Spectrum",
        label="population rate",
    )
    ax.legend(loc="best")

    fig.savefig(f"{epoch_figure_dir}/mean_rate_spectrum.pdf", bbox_inches="tight")
    plt.close(fig)
    return fig

# def plot_spike(result, model_param, cfg, sampled_neurons, connectome_data=None, time_points=None, epoch_figure_dir=None):
#     """Plot spike responses for sampled neurons with neuron type color coding."""
#     import matplotlib.pyplot as plt
#     from matplotlib.patches import Rectangle
#     import matplotlib.patches as mpatches

#     times = time_points
    
#     # Get neuron data
#     if connectome_data is not None:
#         neurons = connectome_data[0]
#     else:
#         # Fallback: create dummy neuron data
#         neurons = None
    
#     # Create figure with subplots - now with 3 rows: colorbar, raster, firing rate
#     fig = plt.figure(figsize=(16, 12))
#     gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 12], height_ratios=[10, 2], hspace=0.1, wspace=0.02)
    
#     # Main raster plot
#     ax_raster = fig.add_subplot(gs[0, 1])

#     # ================= [修复代码 START] =================
#     # 获取原始脉冲数据
#     raw_spikes = result["neuron"]["spike"] # Expect [T, B, N] or [T, N]

#     breakpoint()
    
#     # 处理 Batch 维度
#     if raw_spikes.ndim == 3:
#         # [T, B, N] -> 取第0个batch -> [T, N]
#         # 注意：这里默认可视化 Batch 0，如果需要其他 Batch，可以将其作为参数传入
#         spike_data_full = raw_spikes[:, 0, :] 
#     else:
#         spike_data_full = raw_spikes

#     # 选取指定的神经元 -> [T, n_sampled]
#     spike_data = spike_data_full.cpu().numpy()
#     # ================= [修复代码 END] ===================
    
#     # Plot raster
#     #spike_data = result["neuron"]["spike"][:, sampled_neurons].cpu().numpy()
    
#     # Find spike times and indices
#     spike_indices = []
#     spike_times = []
#     for i, neuron_idx in enumerate(sampled_neurons):
#         spikes = np.where(spike_data[:, i] > 0)[0]
#         spike_indices.extend([i] * len(spikes))
#         spike_times.extend(times[spikes])
    
#     if spike_indices:
#         ax_raster.scatter(spike_times, spike_indices, s=0.5, c='black', alpha=0.7)
    
#     ax_raster.set_xlim(times[0], times[-1])
#     ax_raster.set_ylim(0, len(sampled_neurons))
#     ax_raster.set_ylabel("Neuron Index")
    
#     # 计算产生发放的神经元数量
#     spiking_neurons = len(np.unique(spike_indices)) if spike_indices else 0
#     total_neurons = len(sampled_neurons)
#     ax_raster.set_title(f"Spike Raster Plot ({total_neurons} neurons, {spiking_neurons} spiking)")
    
#     # Firing rate plot
#     ax_fr = fig.add_subplot(gs[1, 1])
#     # Calculate firing rate
#     fr = np.mean(spike_data, axis=1) * (1000.0 / cfg.simulation.dt)  # Convert to Hz
#     ax_fr.plot(times, fr, 'b-', linewidth=1)
#     ax_fr.set_ylabel("Firing rate (Hz)")
#     ax_fr.set_xlabel("Time (ms)")
#     ax_fr.set_xlim(times[0], times[-1])
#     ax_fr.set_ylim(0, fr.max() * 1.1)
    
#     # Color bar for neuron types
#     ax_colorbar = fig.add_subplot(gs[0, 0])
#     ax_colorbar.set_xlim(0, 1)
#     ax_colorbar.set_ylim(0, len(sampled_neurons))
#     ax_colorbar.axis('off')
    
#     if neurons is not None and 'cell_class' in neurons.columns:
#         # Get neuron types for sampled neurons
#         sampled_cell_classes = neurons.iloc[sampled_neurons]['cell_class'].values
        
#         # Create color mapping for different cell types
#         unique_types = np.unique(sampled_cell_classes)
#         colors = plt.cm.tab20(np.linspace(0, 1, len(unique_types)))
#         type_to_color = dict(zip(unique_types, colors))
        
#         # Draw color bars
#         current_y = 0
#         type_ranges = {}
        
#         for i, cell_class in enumerate(sampled_cell_classes):
#             color = type_to_color[cell_class]
#             # Draw a thin vertical line for this neuron
#             ax_colorbar.add_patch(Rectangle((0.3, current_y), 0.4, 1, 
#                                           facecolor=color, alpha=0.9, edgecolor='white', linewidth=0.3))
#             current_y += 1
            
#             # Track ranges for each type
#             if cell_class not in type_ranges:
#                 type_ranges[cell_class] = {'start': i, 'end': i}
#             else:
#                 type_ranges[cell_class]['end'] = i
        
#         # Add type labels with scale lines - now horizontal
#         # Sort types by their position to avoid overlapping labels
#         sorted_types = sorted(unique_types, key=lambda x: type_ranges[x]['start'])
#         label_positions = []
        
#         for cell_class in sorted_types:
#             if cell_class in type_ranges:
#                 start_idx = type_ranges[cell_class]['start']
#                 end_idx = type_ranges[cell_class]['end']
#                 mid_y = (start_idx + end_idx) / 2
                
#                 # Check if this position would overlap with existing labels
#                 min_distance = 20  # Minimum distance between labels
#                 too_close = any(abs(mid_y - pos) < min_distance for pos in label_positions)
                
#                 if not too_close or end_idx - start_idx > 10:  # Always show labels for large ranges
#                     # Only add scale lines if the range is large enough
#                     if end_idx - start_idx > 3:  # Only for ranges with more than 3 neurons
#                         # Add scale lines (bracket-like)
#                         ax_colorbar.plot([0.8, 0.95], [start_idx, start_idx], 'k-', linewidth=2)
#                         ax_colorbar.plot([0.8, 0.95], [end_idx, end_idx], 'k-', linewidth=2)
#                         ax_colorbar.plot([0.8, 0.8], [start_idx, end_idx], 'k-', linewidth=2)
                        
#                         # Add small vertical lines at the ends
#                         ax_colorbar.plot([0.95, 0.95], [start_idx, start_idx-0.3], 'k-', linewidth=2)
#                         ax_colorbar.plot([0.95, 0.95], [end_idx, end_idx+0.3], 'k-', linewidth=2)
                    
#                     # Add label - now horizontal
#                     ax_colorbar.text(0.05, mid_y, cell_class, ha='right', va='center', 
#                                    fontsize=7, transform=ax_colorbar.transData, weight='bold')
#                     label_positions.append(mid_y)
        
#         # Add legend in the top right area to avoid blocking color bar
#         legend_elements = [mpatches.Patch(color=color, label=cell_class) 
#                           for cell_class, color in type_to_color.items()]
#         ax_colorbar.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), 
#                           fontsize=6, ncol=1, frameon=True, fancybox=True, shadow=True)
#     else:
#         # No cell class information available
#         ax_colorbar.text(0.5, 0.5, 'No cell class\ninformation\navailable', 
#                         ha='center', va='center', transform=ax_colorbar.transAxes,
#                         fontsize=10)
    
#     # Empty subplot for bottom left to maintain grid alignment
#     ax_empty = fig.add_subplot(gs[1, 0])
#     ax_empty.axis('off')
    
#     plt.tight_layout()
#     fig.savefig(f"{epoch_figure_dir}/spike_responses.jpg", dpi=250, bbox_inches='tight')
#     return fig

def plot_spike(result, model_param, cfg, sampled_neurons, connectome_data=None, time_points=None, epoch_figure_dir=None):
    """
    Plot spike responses for ALL neurons with neuron type color coding.
    注意：此函数会忽略传入的 sampled_neurons 参数，强制绘制所有神经元。
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import matplotlib.patches as mpatches
    import numpy as np

    # ================= [核心修复 START] =================
    # 1. 获取原始脉冲数据 [Time, Batch, Neurons]
    raw_spikes = result["neuron"]["spike"] 
    
    # 2. 处理 Batch 维度，默认选取第 0 个样本进行可视化
    if raw_spikes.ndim == 3:
        # [1000, 64, 1343] -> [1000, 1343]
        spike_data_full = raw_spikes[:, 0, :] 
    else:
        spike_data_full = raw_spikes

    # 3. 强制覆盖 sampled_neurons 为所有神经元索引
    # 这样可以保证后续逻辑针对全量 1343 个神经元进行
    num_total_neurons = spike_data_full.shape[1]
    sampled_neurons = np.arange(num_total_neurons)
    
    # 转为 numpy 用于绘图
    spike_data = spike_data_full.cpu().numpy()
    
    # 4. 确保时间轴存在
    if time_points is None:
        times = np.arange(spike_data.shape[0]) * cfg.simulation.dt
    else:
        times = time_points
    # ================= [核心修复 END] ===================

    # Get neuron data
    if connectome_data is not None:
        neurons = connectome_data[0]
    else:
        neurons = None
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 12], height_ratios=[10, 2], hspace=0.1, wspace=0.02)
    
    # Main raster plot
    ax_raster = fig.add_subplot(gs[0, 1])
    
    # Find spike times and indices
    spike_indices = []
    spike_times = []
    
    # 遍历所有 1343 个神经元
    for i, neuron_idx in enumerate(sampled_neurons):
        # i 对应 y轴位置 (0..1342), neuron_idx 对应神经元ID (也是0..1342)
        # spike_data[:, i] 取出第 i 个神经元的所有时间步
        spikes = np.where(spike_data[:, i] > 0)[0]
        
        if len(spikes) > 0:
            spike_indices.extend([i] * len(spikes))
            spike_times.extend(times[spikes])
    
    if spike_indices:
        # 对于全量神经元，点的大小(s)设小一点，marker用 '|' 会更清晰
        ax_raster.scatter(spike_times, spike_indices, s=0.2, c='black', alpha=0.6, marker='|')
    
    ax_raster.set_xlim(times[0], times[-1])
    ax_raster.set_ylim(0, len(sampled_neurons))
    ax_raster.set_ylabel("Neuron Index")
    
    spiking_neurons = len(np.unique(spike_indices)) if spike_indices else 0
    total_neurons = len(sampled_neurons)
    ax_raster.set_title(f"Spike Raster Plot (All {total_neurons} neurons, {spiking_neurons} spiking)")
    
    # Firing rate plot
    ax_fr = fig.add_subplot(gs[1, 1])
    # axis=1 对所有神经元求平均，得到群体平均发放率
    fr = np.mean(spike_data, axis=1) * (1000.0 / cfg.simulation.dt)  # Convert to Hz
    ax_fr.plot(times, fr, 'b-', linewidth=1)
    ax_fr.set_ylabel("Population Rate (Hz)")
    ax_fr.set_xlabel("Time (ms)")
    ax_fr.set_xlim(times[0], times[-1])
    # 加上 max(..., 1.0) 防止全0时 ylim 报错
    ax_fr.set_ylim(0, max(fr.max() * 1.1, 1.0))
    
    # Color bar for neuron types
    ax_colorbar = fig.add_subplot(gs[0, 0])
    ax_colorbar.set_xlim(0, 1)
    ax_colorbar.set_ylim(0, len(sampled_neurons))
    ax_colorbar.axis('off')
    
    if neurons is not None and 'cell_class' in neurons.columns:
        # sampled_neurons 现在是全量索引，直接iloc取值即可
        sampled_cell_classes = neurons.iloc[sampled_neurons]['cell_class'].values
        
        unique_types = np.unique(sampled_cell_classes)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_types)))
        type_to_color = dict(zip(unique_types, colors))
        
        # 绘制颜色条
        # 由于有1343个矩形，为了性能，这里依然逐个画，但去掉了边框
        for i, cell_class in enumerate(sampled_cell_classes):
            color = type_to_color[cell_class]
            ax_colorbar.add_patch(Rectangle((0.3, i), 0.4, 1, 
                                          facecolor=color, edgecolor='none', alpha=0.9))
            
        # 计算类型区间用于标注
        type_ranges = {}
        for i, cell_class in enumerate(sampled_cell_classes):
            if cell_class not in type_ranges:
                type_ranges[cell_class] = {'start': i, 'end': i}
            else:
                type_ranges[cell_class]['end'] = i
        
        sorted_types = sorted(unique_types, key=lambda x: type_ranges[x]['start'])
        label_positions = []
        
        for cell_class in sorted_types:
            if cell_class in type_ranges:
                start_idx = type_ranges[cell_class]['start']
                end_idx = type_ranges[cell_class]['end']
                mid_y = (start_idx + end_idx) / 2
                
                # 防止标签重叠的简单逻辑
                min_distance = total_neurons * 0.02 
                too_close = any(abs(mid_y - pos) < min_distance for pos in label_positions)
                
                # 只有区间够大或者不拥挤时才显示标签
                if not too_close or (end_idx - start_idx) > (total_neurons * 0.01):
                    # 如果区间较大，画一下范围线
                    if (end_idx - start_idx) > (total_neurons * 0.005):
                        ax_colorbar.plot([0.8, 0.95], [start_idx, start_idx], 'k-', linewidth=1)
                        ax_colorbar.plot([0.8, 0.95], [end_idx, end_idx], 'k-', linewidth=1)
                        ax_colorbar.plot([0.8, 0.8], [start_idx, end_idx], 'k-', linewidth=1)
                    
                    ax_colorbar.text(0.05, mid_y, cell_class, ha='right', va='center', 
                                   fontsize=7, transform=ax_colorbar.transData, weight='bold')
                    label_positions.append(mid_y)
        
        legend_elements = [mpatches.Patch(color=color, label=cell_class) 
                          for cell_class, color in type_to_color.items()]
        # 如果类型太多，调整图例列数
        ncol = 2 if len(legend_elements) > 15 else 1
        ax_colorbar.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), 
                          fontsize=6, ncol=ncol, frameon=True, shadow=True)
    else:
        ax_colorbar.text(0.5, 0.5, 'No cell class\ninfo', 
                        ha='center', va='center', transform=ax_colorbar.transAxes, fontsize=10)
    
    ax_empty = fig.add_subplot(gs[1, 0])
    ax_empty.axis('off')
    
    plt.tight_layout()
    if epoch_figure_dir:
        fig.savefig(f"{epoch_figure_dir}/spike_responses_full.jpg", dpi=200, bbox_inches='tight')
    return fig

def visualize_results(result, model_param, cfg, connectome_data, selected_neurons=None, time_points=None, epoch_figure_dir=None):
    """可视化模拟结果。
    
    Args:
        result: 模拟结果字典
        model_param: 模型参数
        cfg: 配置
        connectome_data: 连接组数据
        time_points: 时间点
        epoch_figure_dir: 绘图保存目录
    """
    import os
    os.makedirs(epoch_figure_dir, exist_ok=True)

    # result_dict = {}
    # breakpoint()
    # for key in result.keys():
    #     key_parts = key.split(".")
    #     key_group = key_parts[1]
    #     key_name = key_parts[2]
    #     if key_group not in result_dict:
    #         result_dict[key_group] = {}
    #     result_dict[key_group][key_name] = result[key][:, 0, :].cpu().numpy()
    # result = result_dict

    if selected_neurons is None:
        # 选择神经元进行可视化
        selected_neurons, neuron_info = select_neurons_for_visualization(
            result, model_param, cfg, connectome_data
        )

    neurons = connectome_data[0]
    root_id_converter = simple_id_to_root_id_func(neurons)
    selected_root_ids = root_id_converter(selected_neurons)

    total_neurons = result["neuron"]["v"].shape[-1]
    max_neurons = getattr(cfg, 'max_neurons_raster', None)
    
    if max_neurons is None:
        sampled_neurons = np.arange(total_neurons)
        if total_neurons > 10000:
            print("Warning: Large network detected. This may take longer to render and result in a large file.")
    else:
        if total_neurons <= max_neurons:
            sampled_neurons = np.arange(total_neurons)
        else:
            np.random.seed(cfg.seed)
            sampled_neurons = np.random.choice(
                total_neurons,
                max_neurons,
                replace=False,
            )

    default_figures = {
        "voltage",
        "post_synapse_current",
        "spike",
        "fr_spectrum",
        #"cv_distribution",
        #"branching_ratio",
    }

    if getattr(cfg, 'figures_to_plot', None) is None:
        figures_to_plot = default_figures
    else:
        figures_to_plot = getattr(cfg, 'figures_to_plot')

    # Generate requested figures
    visualization_results = {}

    # if "spiked_neuron" in figures_to_plot:
    #     visualization_results["spiked_neurons"] = report_neurons_spiked(result, model_param, training_config, root_id_converter)

    if "voltage" in figures_to_plot:
        # 取出所选神经元的cell_class，按selected_neurons顺序排列
        neuron_df = connectome_data[0]
        cell_classes = neuron_df.iloc[selected_neurons]['cell_class'].values if 'cell_class' in neuron_df.columns else None
        visualization_results["voltage_fig"] = plot_voltage(result, model_param, cfg, selected_neurons, selected_root_ids, cell_classes=cell_classes, epoch_figure_dir=epoch_figure_dir)

    if "post_synapse_current" in figures_to_plot:
        #breakpoint()
        #result['neuron', 'synapse']
        #result['synapse']['psc'].shape: [T, bs, n_neuron]

        visualization_results["post_synapse_fig"] = plot_post_synapse_current(
            result, model_param, cfg, selected_neurons, selected_root_ids, cell_classes=cell_classes, connectome_data=connectome_data, time_points=time_points, epoch_figure_dir=epoch_figure_dir, time_range_ms=cfg.simulation.psc_time_range_ms
        )
    if "spike" in figures_to_plot:
        visualization_results["spike_fig"] = plot_spike(
            result, model_param, cfg, sampled_neurons, connectome_data, time_points=time_points, epoch_figure_dir=epoch_figure_dir
        )
    if "fr_spectrum" in figures_to_plot:
        visualization_results["fr_spectrum_fig"] = plot_fr_spectrum(
            result, model_param, cfg, epoch_figure_dir=epoch_figure_dir
        )
    return visualization_results

#here
def select_neurons_by_class(
    connectome_data,
    num_per_class=2,
    seed=42
):
    """从每一类神经元中随机选择指定数量的神经元进行可视化。
    
    Args:
        connectome_data: 连接组数据 (包含 neurons DataFrame)
        num_per_class: 每一类选择多少个神经元
        seed: 随机种子
        
    Returns:
        tuple: (selected_neuron_indices, neuron_info_df)
    """
    # 获取神经元数据 DataFrame
    neuron_df = connectome_data[0]
    
    # 确保有 cell_class 列
    if 'cell_class' not in neuron_df.columns:
        print("⚠️ Warning: 'cell_class' column not found in neuron data.")
        return np.array([]), pd.DataFrame()

    # 设置随机种子
    np.random.seed(seed)
    
    selected_indices = []
    selected_info = []
    
    # 获取所有唯一的类别
    unique_classes = neuron_df['cell_class'].unique()
    
    print(f"\n🔍 按类别筛选神经元 (每类 {num_per_class} 个):")
    print("-" * 50)
    
    for cls in unique_classes:
        # 找到该类别的所有神经元的索引（注意：这里假设 DataFrame 的 index 就是神经元的 simple_id）
        # 如果不是，可能需要用 np.where
        class_indices = np.where(neuron_df['cell_class'] == cls)[0]
        
        if len(class_indices) == 0:
            continue
            
        # 随机选择
        if len(class_indices) <= num_per_class:
            chosen = class_indices
        else:
            chosen = np.random.choice(class_indices, num_per_class, replace=False)
            
        selected_indices.extend(chosen)
        
        # 记录信息
        for idx in chosen:
            row = neuron_df.iloc[idx]
            selected_info.append({
                'neuron_index': idx,
                'root_id': row.name,
                'cell_class': cls,
                'layer': row.get('layer', 'Unknown')
            })
            
        print(f"  - {cls}: 选中 {len(chosen)} 个 (总数: {len(class_indices)})")

    selected_indices = np.array(selected_indices)
    info_df = pd.DataFrame(selected_info)
    
    print("-" * 50)
    print(f"✅ 总共选中 {len(selected_indices)} 个代表性神经元")
    
    return selected_indices, info_df


def save_states_full(result, model_param, cfg, connectome_data, selected_neurons=None, time_points=None, epoch_figure_dir=None):
    import os

    if epoch_figure_dir is None:
        raise ValueError("epoch_figure_dir must not be None")

    os.makedirs(epoch_figure_dir, exist_ok=True)

    def _to_numpy(x):
        if x is None:
            return None
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        if hasattr(x, "numpy"):
            return x.numpy()
        return np.asarray(x)

    def _save_group_parquet(group_name: str, group_dict: dict):
        if group_dict is None or not isinstance(group_dict, dict):
            return

        #breakpoint()

        voltage_scale = model_param.get("neuron", {}).get("voltage_scale") if isinstance(model_param, dict) else None
        El_reference = model_param.get("neuron", {}).get("El_reference") if isinstance(model_param, dict) else None

        out_dir = os.path.join(str(epoch_figure_dir), f"states_{group_name}")
        os.makedirs(out_dir, exist_ok=True)

        for key, value in group_dict.items():
            if key == "spike":
                continue
            arr = _to_numpy(value)
            if arr is None:
                continue

            #breakpoint()

            if arr.ndim == 3:
                # [T, B, N] -> [T, N] (默认 batch 0)
                arr = arr[:, 0, :]
            elif arr.ndim == 2:
                # [T, N] keep
                pass
            else:
                # 其它形状统一展平保存
                arr = arr.reshape(arr.shape[0], -1) if arr.ndim >= 1 else arr.reshape(1, -1)

            # 由于 result 里除 spike 外，其它量（例如 v, psc）都经过 voltage_scale 缩放，
            # 这里在保存前恢复为原始量级。
            if voltage_scale is not None:
                vs = np.asarray(voltage_scale)
                if vs.ndim > 0 and arr.ndim >= 2 and vs.shape[0] == arr.shape[1]:
                    vs_use = vs
                else:
                    vs_use = vs

                if group_name == "neuron" and key == "v":
                    # v_scaled = (v_raw - El_reference) / voltage_scale
                    # => v_raw = v_scaled * voltage_scale + El_reference
                    if El_reference is not None:
                        el = np.asarray(El_reference)
                        if el.ndim > 0 and arr.ndim >= 2 and el.shape[0] == arr.shape[1]:
                            el_use = el
                        else:
                            el_use = el
                        arr = arr * vs_use + el_use
                elif group_name == "synapse" and key in {"psc", "psc_e", "psc_i", "psc_all", "psc_e_all"}:
                    # PSC 只需要乘回 voltage_scale
                    arr = arr * vs_use

            # 保存为长表：time, neuron_id, value
            T = arr.shape[0]
            N = arr.shape[1]
            df = pd.DataFrame(
                {
                    "time": np.repeat(np.arange(T, dtype=np.int64), N),
                    "neuron_id": np.tile(np.arange(N, dtype=np.int64), T),
                    "value": arr.reshape(-1),
                }
            )
            df.to_parquet(os.path.join(out_dir, f"{key}.parquet"), index=False)

    def _save_spike_coo(spike_tensor):
        if spike_tensor is None:
            return

        spike = _to_numpy(spike_tensor)
        if spike is None:
            return

        out_dir = os.path.join(str(epoch_figure_dir), "states_spike")
        os.makedirs(out_dir, exist_ok=True)

        spike = spike[:, 0, :] #只取batch维度的第一个

        if spike.ndim == 3:
            # [T, B, N] -> 保存每个 batch 一个 coo
            T, B, N = spike.shape
            for b in range(B):
                sp_b = spike[:, b, :]
                coo = scipy.sparse.coo_matrix(sp_b)
                df = pd.DataFrame(
                    {
                        "row": coo.row.astype(np.int64),
                        "col": coo.col.astype(np.int64),
                        "data": coo.data,
                        "shape0": np.full(coo.data.shape[0], T, dtype=np.int64),
                        "shape1": np.full(coo.data.shape[0], N, dtype=np.int64),
                    }
                )
                df.to_parquet(os.path.join(out_dir, f"spike_b{b}.parquet"), index=False)
        elif spike.ndim == 2:
            T, N = spike.shape
            coo = scipy.sparse.coo_matrix(spike)
            df = pd.DataFrame(
                {
                    "row": coo.row.astype(np.int64),
                    "col": coo.col.astype(np.int64),
                    "data": coo.data,
                    "shape0": np.full(coo.data.shape[0], T, dtype=np.int64),
                    "shape1": np.full(coo.data.shape[0], N, dtype=np.int64),
                }
            )
            df.to_parquet(os.path.join(out_dir, "spike.parquet"), index=False)
        else:
            raise ValueError(f"Unsupported spike tensor shape: {spike.shape}")

    #breakpoint()

    neuron_dict = result.get("neuron", {}) if isinstance(result, dict) else {} #keys: v, spike
    synapse_dict = result.get("synapse", {}) if isinstance(result, dict) else {} #keys: psc, psc_e, psc_i, psc_e_all, psc_all

    # neuron_dict['v'].shape: [T, b, N]
    # neuron_dict['spike'].shape: [T, b, N]
    # synapse_dict['psc'].shape: [T, b, N]

    #breakpoint()

    _save_group_parquet("neuron", neuron_dict)
    _save_group_parquet("synapse", synapse_dict)
    if isinstance(neuron_dict, dict) and "spike" in neuron_dict:
        _save_spike_coo(neuron_dict.get("spike"))


def plot_multiscale_ff_loglog(window_sizes_steps, ff_values, slope, r_value, save_dir, dt=1.0):
    """
    绘制multi-scale fano factor的log-log曲线。
    
    Args:
        window_sizes_steps: numpy.ndarray，窗口大小（步数）
        ff_values: numpy.ndarray，每个窗口大小对应的FF值
        slope: float，拟合的斜率
        r_value: float，拟合的R²值
        save_dir: str，保存目录
        dt: float，仿真步长(ms)
    """
    import os
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 转换窗口大小为毫秒
    window_sizes_ms = window_sizes_steps * dt
    
    # 移除NaN值用于绘制
    valid_idx = ~np.isnan(ff_values)
    window_sizes_valid = window_sizes_ms[valid_idx]
    ff_values_valid = ff_values[valid_idx]
    
    # 绘制数据点
    ax.loglog(window_sizes_valid, ff_values_valid, 'o-', linewidth=2, markersize=8, 
              label='Multi-scale FF', color='steelblue', alpha=0.7)
    
    # 绘制拟合线
    if not np.isnan(slope) and not np.isnan(r_value):
        # 计算拟合线
        x_fit = np.logspace(np.log10(window_sizes_valid.min()), 
                           np.log10(window_sizes_valid.max()), 100)
        
        # 从log-log拟合恢复原始参数
        # log(FF) = slope * log(window) + intercept
        # 需要计算intercept
        log_x = np.log10(window_sizes_valid)
        log_y = np.log10(ff_values_valid)
        intercept = np.mean(log_y - slope * log_x)
        
        y_fit = 10 ** (slope * np.log10(x_fit) + intercept)
        
        ax.loglog(x_fit, y_fit, '--', linewidth=2, label=f'Fit (α={slope:.3f})', 
                 color='red', alpha=0.7)
    
    # 设置标签和标题
    ax.set_xlabel('Window Size (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fano Factor', fontsize=12, fontweight='bold')
    
    # 生成标题
    if not np.isnan(slope) and not np.isnan(r_value):
        title = f'Multi-scale Fano Factor (Slope α={slope:.3f}, R²={r_value:.3f})'
    else:
        title = 'Multi-scale Fano Factor'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 添加网格
    ax.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.3)
    
    # 添加图例
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    
    # 添加参考线（斜率=0表示泊松过程）
    if window_sizes_valid.size > 0:
        y_poisson = np.ones_like(window_sizes_valid)
        ax.loglog(window_sizes_valid, y_poisson, ':', linewidth=1.5, 
                 label='Poisson (α=0)', color='gray', alpha=0.5)
    
    plt.tight_layout()
    
    # 保存图像
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'multiscale_ff_loglog.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 也保存为PDF格式
    save_path_pdf = os.path.join(save_dir, 'multiscale_ff_loglog.pdf')
    fig.savefig(save_path_pdf, bbox_inches='tight')
    
    plt.close(fig)
    
    print(f"✅ Multi-scale FF log-log plot saved to {save_path}")
