from typing import Any, Callable, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def get_figure(n_rows: int, n_cols: int, height_per_row: float, width: float):
    fig_height = max(1.0, float(n_rows) * float(height_per_row))
    fig = plt.figure(
        figsize=(float(width), fig_height),
        constrained_layout=True,
    )
    gs = fig.add_gridspec(n_rows, n_cols)
    return fig, gs


def plot_population_by_field(
    res,
    field: str,
    ts=None,
    name=None,
    population: Optional[Sequence[int]] = None,
    title_format: Callable[[Sequence[Any]], Any] = lambda arg: f"[{arg[0]}] = {arg[1]}",
    ref_line=None,
    ref_name=None,
    plot_args=None,
    voltage_scale=None,
    El_reference=None,
    batch_index: int = 0,  # <--- 新增参数：指定可视化哪个 batch
):
    # ================= [新增适配逻辑 START] =================
    # 获取原始数据
    raw_data = res[field]

    #breakpoint()
    
    # 1. 处理 4D 数据 [Time, Batch, Neurons, Lines] -> [Time, Neurons, Lines]
    # 通常出现在 plot_post_synapse_current 中 (E/I currents)
    if raw_data.ndim == 4:
        # print(f"Auto-slicing batch from 4D: {raw_data.shape} -> select batch {batch_index}")
        data = raw_data[:, batch_index, :, :] 
        
    # 2. 处理 3D 数据 [Time, Batch, Neurons] -> [Time, Neurons]
    # 判据：第三维 > 10 (认为是 Neurons)，区分于 [Time, Neurons, Lines] (Lines通常很少)
    elif raw_data.ndim == 3 and raw_data.shape[2] > 4:
        # print(f"Auto-slicing batch from 3D: {raw_data.shape} -> select batch {batch_index}")
        data = raw_data[:, batch_index, :] 
        
    else:
        data = raw_data
        
    # 确保数据在 CPU 上以便绘图 (防止 matplotlib 报错)
    if hasattr(data, 'cpu'):
        data = data.cpu().numpy()
    elif hasattr(data, 'numpy'): # if it's already a cpu tensor
        data = data.numpy()

    if name is None:
        name = field
    n_ts = data.shape[0] # 使用处理后的 data
    #breakpoint()
    if ts is None:
        ts = np.arange(n_ts)
    if population is None:
        num = data.shape[1] # 使用处理后的 data
        n_sel = 10 if num > 10 else num
        population = np.random.choice(n_sel, 10, replace=False)
    n_sel = len(population)
    fig, gs = get_figure(
        n_sel,
        1,
        1.5,
        8,
    )
    #breakpoint()
    
    if data.ndim == 3:
        # 此时如果是 3D，说明真的是 [Time, Neurons, Lines]
        n_lines = data.shape[2]
        if plot_args is None:
            plot_args = [None] * n_lines
        assert len(plot_args) == n_lines
    elif data.ndim == 2:
        n_lines = 1
    else:
        raise ValueError("res cannot contain higher dim traces")
        
    if n_lines == 1:
        if isinstance(plot_args, Sequence) and plot_args is not None:
            # 兼容旧逻辑，如果用户传了列表但只有1条线
            if len(plot_args) >= 1: 
                plot_args = plot_args[0]
            else:
                plot_args = {}

    fig.suptitle(f"{name}", fontsize=16)
    #breakpoint()
    for i, s in enumerate(population):
        ax = fig.add_subplot(gs[i, 0])
        
        # Apply voltage scaling and El_reference offset if provided
        if voltage_scale is not None and El_reference is not None:
            # Handle per-neuron scaling
            if hasattr(voltage_scale, '__len__') and len(voltage_scale) > 1:
                v_scale = voltage_scale[s]
            else:
                v_scale = voltage_scale
            if hasattr(El_reference, '__len__') and len(El_reference) > 1:
                el_ref = El_reference[s]
            else:
                el_ref = El_reference
        else:
            v_scale = 1.0
            el_ref = 0.0
            
        if n_lines > 1:
            for j in range(data[:, s].shape[-1]): # 使用 data
                # Apply voltage transformation
                val = data[:, s, j] * v_scale + el_ref
                #val = val.cpu().numpy()
                kwargs = dict(plot_args[j]) if isinstance(plot_args[j], dict) else {}
                lbl = kwargs.pop("label", f"{field}-{j}")
                ax.plot(ts, val, label=lbl, **kwargs)
        elif n_lines == 1:
            # Apply voltage transformation
            val = data[:, s] * v_scale + el_ref
            #val = val.cpu().numpy()
            
            kwargs = dict(plot_args) if isinstance(plot_args, dict) else {}
            lbl = kwargs.pop("label", f"{field}")
            ax.plot(ts, val, label=lbl, **kwargs)

        #breakpoint()
            
        if ref_line is not None:
            if ref_name is None:
                ref_name = f"{field}_ref"
            if hasattr(ref_line, '__len__') and len(ref_line) == n_sel:
                ref_l = ref_line[i]
            elif hasattr(ref_line, "size") and ref_line.size == n_ts:
                ref_l = ref_line
            else:
                ref_l = ref_line
            ax.hlines(ref_l, ts[0], ts[-1], label=ref_name, linewidth=2, color="red", alpha=0.7)
        ax.set_xlabel("Time (ms)")
        ax.set_xlim(-0.1, ts[-1] + 0.1)
        ax.set_title(title_format((i, s)))
        # ax.xaxis.set_minor_locator(MultipleLocator(2))
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", fontsize=8, frameon=True)
    return fig


def plot_population_by_field_single_plot(
    res,
    field: str,
    ts=None,
    name=None,
    population: Optional[Union[np.ndarray, Sequence[int], int]] = None,
    legend_format: Optional[
        Callable[[Sequence[Any]], Any]
    ] = lambda arg: f"[{arg[0]}] = {arg[1]}",
    ref_line: Optional[Union[float, Sequence[float]]] = None,
    ref_name: Optional[Union[str, Sequence[str]]] = None,
    plot_args: Optional[Union[dict[str, Any], Sequence[dict[str, Any]]]] = None,
    color_map: Optional[Sequence[Any]] = None,
    voltage_scale=None,
    El_reference=None,
    batch_index: int = 0, # <--- 新增参数
):
    """Plot all traces recorded in res['field'] in one figure."""

    # ================= [新增适配逻辑 START] =================
    raw_data = res[field]
    # 1. 4D -> 3D
    if raw_data.ndim == 4:
        data = raw_data[:, batch_index, :, :]
    # 2. 3D -> 2D (Heuristic)
    elif raw_data.ndim == 3 and raw_data.shape[2] > 4:
        data = raw_data[:, batch_index, :] 
    else:
        data = raw_data
        
    # Move to CPU numpy for plotting
    if hasattr(data, 'cpu'):
        data = data.cpu().numpy()
    elif hasattr(data, 'numpy'):
        data = data.numpy()
    # ================= [新增适配逻辑 END] =================

    num_dims = len(data.shape)
    if name is None:
        name = field

    if ts is None:
        ts = np.arange(data.shape[0])

    if population is None or isinstance(population, int):
        num = data.shape[1]  # 使用 data
        n_sel = (10 if num > 10 else num) if population is None else int(population)
        population = np.random.choice(num, n_sel, replace=False)

    # Handle color map
    if color_map is None:
        # Fix: cm.get_cmap is deprecated in recent mpl, use matplotlib.colormaps or cm.turbo
        try:
            cmap = plt.get_cmap("turbo")
        except:
            cmap = cm.get_cmap("turbo")
            
        # 生成均匀分布的颜色
        indices = np.linspace(0, 1, len(population))
        color_map = [cmap(idx) for idx in indices]
    else:
        assert len(color_map) == len(population)

    assert (
        ts is None or len(ts) == data.shape[0]
    ), "ts must have the same length as the first dimension of res[field]"

    if isinstance(plot_args, Sequence) and num_dims == 3:
        assert len(plot_args) == data.shape[2]
    
    # Handle plot arguments
    if plot_args is None:
        plot_args = [{}]
    elif isinstance(plot_args, dict):
        # 如果是 3D (Time, Neurons, Lines)，复制 args
        if num_dims == 3:
            plot_args = [plot_args] * data.shape[2]
        else:
            plot_args = [plot_args] # 2D 只需要一个 dict
    elif isinstance(plot_args, list) and num_dims == 2:
         # 如果是 2D 但用户传了 list，取第一个
         if len(plot_args) > 0:
             plot_args = [plot_args[0]]
         else:
             plot_args = [{}]

    # Handle reference lines (Same logic using data.shape)
    if isinstance(ref_line, Sequence) and num_dims == 3:
        assert len(ref_line) == data.shape[2]
    if isinstance(ref_line, (int, float)) or ref_line is None:
        if num_dims == 3:
            ref_line = [ref_line] * data.shape[2]
        else:
            ref_line = [ref_line]
            
    if isinstance(ref_name, Sequence) and num_dims == 3:
        assert len(ref_name) == data.shape[2]
    if isinstance(ref_name, str) or ref_name is None:
        if num_dims == 3:
            ref_name = [ref_name] * data.shape[2]
        else:
            ref_name = [ref_name]

    # Plotting
    figs = []
    # 如果是 2D，循环一次；如果是 3D，循环第三维（线条数）次
    loop_range = range(data.shape[2]) if num_dims == 3 else (0,)
    
    for i in loop_range:
        fig, ax = plt.subplots(figsize=(10, 6))
        for j, p in enumerate(population):
            # Apply voltage scaling and El_reference offset
            if voltage_scale is not None and El_reference is not None:
                if hasattr(voltage_scale, '__len__') and len(voltage_scale) > 1:
                    v_scale = voltage_scale[p]
                else:
                    v_scale = voltage_scale
                if hasattr(El_reference, '__len__') and len(El_reference) > 1:
                    el_ref = El_reference[p]
                else:
                    el_ref = El_reference
            else:
                v_scale = 1.0
                el_ref = 0.0
            
            # Select data
            raw_val = data[:, p, i] if num_dims == 3 else data[:, p]
            plot_data = raw_val * v_scale + el_ref
            #plot_data = plot_data.cpu().numpy()
            # Get plot args for this line index i
            # plot_args is a list. If 2D, it has 1 element. If 3D, it has n_lines elements.
            # However, plot_args[i] itself is a dict
            kwargs = dict(plot_args[i])
            
            ax.plot(
                ts,
                plot_data,
                label=legend_format([j, p]) if legend_format is not None else None,
                color=color_map[j],
                alpha=0.8,
                lw=1.0,
                **kwargs,
            )
            
        # Draw reference lines
        # ref_line is a list corresponding to the loop index i
        curr_ref = ref_line[i] if i < len(ref_line) else None
        curr_ref_name = ref_name[i] if i < len(ref_name) else None
        
        if curr_ref is not None:
            if hasattr(curr_ref, '__len__') and len(curr_ref) == len(population):
                for j, p in enumerate(population):
                    ax.axhline(curr_ref[j], label=curr_ref_name if j==0 else None, 
                             linewidth=2, color="red", alpha=0.7, linestyle='--')
            else:
                ax.axhline(curr_ref, label=curr_ref_name, 
                         linewidth=2, color="red", alpha=0.7, linestyle='--')
        
        if legend_format is not None or ref_name is not None:
            # 调整图例位置，避免遮挡
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
        if num_dims == 3:
            ax.set_title(f"{name} - Component {i}")
        else:
            ax.set_title(name)
        ax.set_xlabel("Time (ms)")
        plt.tight_layout() # 防止图例切边
        figs.append(fig)

    return figs, color_map
