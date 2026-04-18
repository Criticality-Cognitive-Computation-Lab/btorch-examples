import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import os
import glob
import json
import math  # 必须导入 math 库用于计算半径
import matplotlib.pyplot as plt
from pyvis.network import Network
import streamlit.components.v1 as components
import ipdb

# Set page config at module level
st.set_page_config(layout="wide", page_title="SNN Real-Data Visualizer")

@st.cache_data
def load_network_static(network_dir):
    """
    Load network topology from neurons.csv.gz and connections.csv.gz
    """
    neurons_path = os.path.join(network_dir, "neurons.csv.gz")
    conns_path = os.path.join(network_dir, "connections.csv.gz")
    
    if not os.path.exists(neurons_path) or not os.path.exists(conns_path):
        return None, None, f"Files not found in {network_dir}"

    # 1. Load Neurons
    df_neurons = pd.read_csv(neurons_path, compression='gzip')
    # Create simple_id mapping just in case
    # Assuming 'simple_id' column exists and is 0-indexed contiguous
    N = len(df_neurons)
    
    # 2. Load Connections
    df_conns = pd.read_csv(conns_path, compression='gzip')
    
    # 3. Build Graph
    G = nx.DiGraph()

    #breakpoint()
    
    # Add nodes with attributes
    # We use simple_id as the node ID in the graph for performance/array indexing
    for _, row in df_neurons.iterrows():
        G.add_node(
            int(row['simple_id']), 
            type=str(row.get('cell_class', row.get('type', 'Unknown'))),
            ei=str(row.get('EI', 'E')), # Expecting 'E' or 'I'
            layer=str(row.get('layer', 'Unknown')),
            pos_x=row.get('x_position', 0.0),
            pos_y=row.get('y_position', 0.0),
            pos_z=row.get('z_position', 0.0)
        )
    
    # Add edges
    # df_conns should have 'pre_simple_id', 'post_simple_id', 'weight'
    for _, row in df_conns.iterrows():
        G.add_edge(
            int(row['pre_simple_id']), 
            int(row['post_simple_id']), 
            pre_type = str(row.get('pre_type', 'Unknown')),
            post_type = str(row.get('post_type', 'Unknown')),
            weight=float(row['weight'])
        )
        
    return G, N, None


def _candidate_glif_dirs(network_dir):
    network_dir = os.path.abspath(network_dir)
    candidates = [
        os.path.join(os.path.dirname(network_dir), "glif_models"),
        os.path.join(os.path.dirname(os.path.dirname(network_dir)), "glif_models"),
        os.path.join(network_dir, "glif_models"),
    ]
    seen = []
    for path in candidates:
        if path not in seen:
            seen.append(path)
    return seen


def _convert_glif_json_to_physical_voltage_params(raw):
    coeffs = raw.get("coeffs", {})
    coeff_th = float(coeffs.get("th_inf", 1.0))
    el_reference_mv = float(raw.get("El_reference", -0.07)) * 1000.0
    el_internal_mv = float(raw.get("El", 0.0)) * 1000.0
    th_inf_mv = float(raw.get("th_inf", raw.get("init_threshold", 0.02))) * coeff_th * 1000.0
    voltage_scale = th_inf_mv if th_inf_mv != 0 else 1.0

    return {
        "El_reference": el_reference_mv,
        "voltage_scale": voltage_scale,
        "v_rest_scaled": el_internal_mv / voltage_scale,
        "v_threshold_scaled": th_inf_mv / voltage_scale,
        "v_rest_physical": el_internal_mv + el_reference_mv,
        "v_threshold_physical": th_inf_mv + el_reference_mv,
    }


@st.cache_data
def load_neuron_voltage_params(network_dir):
    neurons_path = os.path.join(network_dir, "neurons.csv.gz")
    if not os.path.exists(neurons_path):
        return {}, f"Neuron file not found in {network_dir}"

    glif_dir = None
    for candidate in _candidate_glif_dirs(network_dir):
        if os.path.isdir(candidate):
            glif_dir = candidate
            break

    if glif_dir is None:
        return {}, f"Could not locate a glif_models directory near {network_dir}"

    df_neurons = pd.read_csv(neurons_path, compression="gzip")
    if "simple_id" not in df_neurons.columns or "cell_class" not in df_neurons.columns:
        return {}, "neurons.csv.gz must contain 'simple_id' and 'cell_class' columns"

    class_params = {}
    for cell_class in df_neurons["cell_class"].dropna().astype(str).unique():
        class_dir = os.path.join(glif_dir, cell_class)
        json_files = sorted(glob.glob(os.path.join(class_dir, "*.json")))
        if not json_files:
            return {}, f"No GLIF json file found for cell_class '{cell_class}' under {glif_dir}"
        with open(json_files[0], "r", encoding="utf-8") as f:
            class_params[cell_class] = _convert_glif_json_to_physical_voltage_params(json.load(f))

    neuron_params = {}
    for _, row in df_neurons.iterrows():
        simple_id = int(row["simple_id"])
        cell_class = str(row["cell_class"])
        params = dict(class_params[cell_class])
        params["cell_class"] = cell_class
        neuron_params[simple_id] = params

    return neuron_params, None

@st.cache_data
def load_simulation_variable(sim_dir, var_name, num_neurons, expected_shape=None):
    """
    Generic loader for parquet state files (long format or spike coo).
    Returns a dense numpy array [N, T] or [N_neurons, T_steps].
    """
    # 1. Spikes (Sparse COO in parquet)
    if var_name == "spike":
        spike_dir = os.path.join(sim_dir, "states_spike")
        # Try finding spike.parquet or spike_b0.parquet
        files = glob.glob(os.path.join(spike_dir, "spike*.parquet"))
        if not files:
            return None
        
        # Load the first found file (assuming batch 0)
        df = pd.read_parquet(files[0])
        
        # Determine shape
        if 'shape0' in df.columns:
            T = df['shape0'].iloc[0]
            # N is passed in
        else:
            # Fallback estimation
            T = df['row'].max() + 1
        
        # Construct dense array (N, T) for easy slicing
        # spikes usually stored as: row=time, col=neuron_id
        # We want [Neuron, Time] for plotting logic usually, or [Time, Neuron]
        # Let's stick to [Neuron, Time] for consistency with previous app
        dense_spikes = np.zeros((num_neurons, T), dtype=bool)
        
        # df['row'] is time, df['col'] is neuron index
        # Filter out bounds just in case
        valid_mask = (df['col'] < num_neurons) & (df['row'] < T)
        df = df[valid_mask]
        
        dense_spikes[df['col'].values, df['row'].values] = True
        return dense_spikes

    # 2. Continuous Variables (Long format: time, neuron_id, value)
    # Search in states_neuron or states_synapse
    search_paths = [
        os.path.join(sim_dir, "states_neuron", f"{var_name}.parquet"),
        os.path.join(sim_dir, "states_synapse", f"{var_name}.parquet")
    ]
    
    found_path = None
    for p in search_paths:
        if os.path.exists(p):
            found_path = p
            break
            
    if not found_path:
        return None
        
    df = pd.read_parquet(found_path)
    
    # Estimate T
    T = df['time'].max() + 1
    
    # Pivot to [N, T]
    # Creating a zero array and filling is faster than pd.pivot for large data
    dense_arr = np.zeros((num_neurons, T), dtype=np.float32)
    
    # Ensure indices are integers
    times = df['time'].values.astype(int)
    neurons = df['neuron_id'].values.astype(int)
    values = df['value'].values
    
    # Safe fill
    mask = (neurons < num_neurons) & (times < T)
    dense_arr[neurons[mask], times[mask]] = values
    
    return dense_arr

class SNNVisualizer:
    def __init__(self, network_path=None, simulation_path=None):
        self.default_net_path = network_path or "/path/to/network"
        self.default_sim_path = simulation_path or "/path/to/simulation"

    def run(self):
        # --- Sidebar ---
        st.sidebar.title("🛠️ SNN Config")
        net_path = st.sidebar.text_input("Network Output Path", self.default_net_path)
        sim_path = st.sidebar.text_input("Simulation Output Path", self.default_sim_path)

        if not os.path.exists(net_path) or not os.path.exists(sim_path):
            st.warning("Please provide valid paths.")
            return

        # --- Load Data ---
        with st.spinner("Loading Data..."):
            G, N, err = load_network_static(net_path) # 需确保外部定义了此函数
            if err:
                st.error(err)
                return

            neuron_voltage_params, voltage_param_err = load_neuron_voltage_params(net_path)
            if voltage_param_err:
                st.error(voltage_param_err)
                return
            
            v_mem = load_simulation_variable(sim_path, "v", N) # 需确保外部定义了此函数
            spikes = load_simulation_variable(sim_path, "spike", N)
            psc = load_simulation_variable(sim_path, "psc", N)
            
            T_steps = v_mem.shape[1] if v_mem is not None else 100
            
            epsc = load_simulation_variable(sim_path, "psc_e", N)
            if epsc is None: epsc = np.zeros((N, T_steps))
            ipsc = load_simulation_variable(sim_path, "psc_i", N)
            if ipsc is None: ipsc = np.zeros((N, T_steps))
            i_ext = load_simulation_variable(sim_path, "i_ext", N) 
            if i_ext is None: i_ext = np.zeros((N, T_steps))
            i_after = load_simulation_variable(sim_path, "i_after", N)
            if i_after is None: i_after = np.zeros((N, T_steps))

        if v_mem is None or spikes is None:
            st.error("Data load failed.")
            return

        data_bundle = {
            "v_mem": v_mem, "spikes": spikes, "psc": psc,
            "epsc": epsc, "ipsc": ipsc, "i_ext": i_ext, "i_after": i_after,
            "neuron_voltage_params": neuron_voltage_params,
        }
        
        self._render_ui(G, data_bundle, T_steps)

    def _render_ui(self, G, data, total_steps):
        # 1. Selection
        st.sidebar.markdown("---")
        all_nodes = sorted(list(G.nodes()))
        selected_node = st.sidebar.selectbox("Select Center Neuron:", all_nodes, index=0)

        # 2. Filter
        st.sidebar.markdown("### 🕸️ Connection Filter")
        raw_predecessors = list(G.predecessors(selected_node))
        pre_types = sorted(list(set([G.nodes[n]['type'] for n in raw_predecessors])))
        selected_types = st.sidebar.multiselect("Filter Pre-Synaptic Types:", pre_types, default=pre_types)
        
        filtered_predecessors = [n for n in raw_predecessors if G.nodes[n]['type'] in selected_types]
        st.sidebar.caption(f"Showing {len(filtered_predecessors)} inputs")

        # 3. Time Control
        if 'time_step' not in st.session_state: st.session_state.time_step = 0
        def update_slider(): st.session_state.time_step = st.session_state.slider_val
        def update_num(): st.session_state.time_step = st.session_state.num_val
        
        st.title(f"🧠 Micro-Circuit Scope: Node {selected_node}")
        c1, c2 = st.columns([6, 1])
        with c1:
            st.slider("Time Step", 0, total_steps-1, value=st.session_state.time_step, key="slider_val", on_change=update_slider)
        with c2:
            st.number_input("Frame", min_value=0, max_value=total_steps-1, value=st.session_state.time_step, key="num_val", on_change=update_num, label_visibility="collapsed")
        
        t_step = st.session_state.time_step
        
        # 4. HUD
        self._render_hud(G.nodes[selected_node], data, selected_node, t_step, len(raw_predecessors), len(list(G.successors(selected_node))))
        
        # 5. Visualization
        col_vis, col_plots = st.columns([1, 1])
        with col_vis:
            self._render_star_graph(G, selected_node, filtered_predecessors, data, t_step)
        with col_plots:
            self._render_dynamics_plots(G, selected_node, filtered_predecessors, data, t_step, total_steps)

    def _render_hud(self, node_info, data, node_id, t_step, n_pre, n_post):
        current_v = data["v_mem"][node_id, t_step]
        is_spiking = data["spikes"][node_id, t_step]
        neuron_params = data.get("neuron_voltage_params", {}).get(node_id, {})
        v_rest = neuron_params.get("v_rest_physical")
        v_th = neuron_params.get("v_threshold_physical")
        cell_class = neuron_params.get("cell_class", node_info.get("type", "?"))

        st.markdown(f"### 📊 Neuron State (t={t_step})")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: st.metric("Type", f"{cell_class}")
        with c2: st.metric("Membrane", "⚡ SPIKE!" if is_spiking else f"{current_v:.2f} mV", delta="Spiking" if is_spiking else None, delta_color="inverse")
        with c3:
            if v_rest is None or v_th is None:
                st.metric("Rest/Th", "N/A")
            else:
                st.metric("Rest/Th", f"{v_rest:.2f}/{v_th:.2f}")
        with c4: st.metric("Total Inputs", n_pre)
        with c5: st.metric("Total Outputs", n_post)
        st.markdown("---")

    def _render_star_graph(self, G, center, pre_nodes, data, t_step):
        """
        Modified Star Graph:
        1. Dynamic Radius: Prevents overlap by expanding circle based on node count.
        2. Small Nodes: Base size reduced to 5.
        3. Edge Weights: Displayed on edges.
        """
        st.subheader(f"🕸️ Star Connectivity (Filtered)")

        net = Network(height="650px", width="100%", bgcolor="#1E1E1E", font_color="white")
        
        # --- 1. Dynamic Geometry Calculation ---
        num_pre = len(pre_nodes)
        # 关键逻辑：根据节点数量计算所需最小周长
        # 假设每个节点需要 15px 的空间 (5px本体 + 10px间隙)
        min_circumference = max(num_pre * 15, 1500) 
        # 反推半径 r = C / 2pi
        radius = min_circumference / (2 * math.pi)
        
        # 限制最小半径，防止只有1个节点时挤在一起
        radius = max(300, radius)

        # --- 2. Add Center Node ---
        v_curr = data["v_mem"][center, t_step]
        is_spike = data["spikes"][center, t_step]
        n_type = G.nodes[center].get('type', 'Unknown')
        
        if is_spike:
            color = {'background': '#FFFF00', 'border': '#FFFFFF'}
            lbl = "SPIKE!"
        else:
            color = {'background': '#00FF00', 'border': '#FFFFFF'} 
            lbl = f"CENTER\n{v_curr:.1f}mV"
            
        # 中心节点稍微大一点 (size=15)，作为视觉锚点
        net.add_node(center, label=lbl, title=f"Type: {n_type}", 
                     x=0, y=0, color=color, borderWidth=2, size=15,
                     physics=False)

        # --- 3. Add Pre Nodes (Circle) ---
        for i, nid in enumerate(pre_nodes):
            angle = 2 * math.pi * i / (num_pre + 1e-9)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            
            v_curr = data["v_mem"][nid, t_step]
            is_spike = data["spikes"][nid, t_step]
            n_ei = G.nodes[nid].get('ei', 'E')
            
            if is_spike:
                color = {'background': '#FFFF00', 'border': '#FFFFFF'}
                border = 2
                lbl = "⚡"
            else:
                base = "#ff4d4d" if n_ei == 'E' else "#4d4dff"
                color = {'background': base, 'border': base}
                border = 0 # 平静状态下去掉边框，减小视觉体积
                lbl = f"{nid}" # 只显示ID，减少文字干扰

            # 【关键】：size=5，非常小，保证放大后也不糊成一团
            net.add_node(nid, label=lbl, title=f"ID: {nid}\nType: {G.nodes[nid].get('type')}\nV: {v_curr:.2f}mV", 
                         x=x, y=y, color=color, borderWidth=border, size=5,
                         physics=False)
            
            # --- 4. Add Edge with Weights ---
            if G.has_edge(nid, center):
                w = G[nid][center]['weight']
                
                if is_spike:
                    edge_col = "#FFFF00"
                    width = 3
                else:
                    edge_col = "#ff4d4d" if n_ei == 'E' else "#4d4dff"
                    width = 1 # 细线
                
                # 配置连线标签 (权重)
                # align='middle' 是目前最稳妥的方式
                # 我们把背景设为黑色，字体设小，这样即使线很密，也能看清字
                net.add_edge(nid, center, 
                             label=f"{w:.2f}", # 显示两位小数权重
                             title=f"Weight: {w:.4f}",
                             color=edge_col, 
                             width=width, 
                             dashes=not is_spike, 
                             arrows='to',
                             font={
                                 'color': 'white', 
                                 'size': 10, # 字体调小
                                 'align': 'middle', # Vis.js 只能 middle/top/bottom，无法自动贴近source
                                 'background': '#1E1E1E', # 与背景同色，制造“镂空”效果
                                 'strokeWidth': 0
                             })

        # Render
        net.save_graph("temp_net.html")
        with open("temp_net.html", 'r', encoding='utf-8') as f:
            components.html(f.read(), height=660)

    def _render_dynamics_plots(self, G, center, inputs, data, t_step, total_steps):
        # 保持之前的绘图逻辑不变
        st.subheader("📈 Filtered Dynamics")
        ts_v = data["v_mem"][center, :]
        ts_psc = data["psc"][center, :]
        ts_epsc = data["epsc"][center, :]
        ts_ipsc = data["ipsc"][center, :]
        neuron_params = data.get("neuron_voltage_params", {}).get(center, {})
        v_th = neuron_params.get("v_threshold_physical")
        
        max_raster = 50
        if len(inputs) > max_raster:
            display_inputs = inputs[:max_raster]
            st.caption(f"Note: Raster plot limited to first {max_raster} neurons.")
        else:
            display_inputs = inputs
            
        raster_ids = [center] + display_inputs
        
        plt.style.use('dark_background')
        fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True, gridspec_kw={'height_ratios': [3, 2, 2, 2]})
        
        # Plot 0: Raster
        ax0 = axes[0]
        ax0.set_title(f"Spike Raster (Top: Center, Below: Inputs)", color='white', fontsize=10)
        y_labels = []
        for i, nid in enumerate(raster_ids):
            spk_t = np.where(data["spikes"][nid, :])[0]
            if nid == center:
                col, sz, lbl = '#FFFF00', 80, "CENTER"
            else:
                n_type = G.nodes[nid].get('type', '?')
                n_ei = G.nodes[nid].get('ei', 'E')
                col = '#ff4d4d' if n_ei == 'E' else '#4d4dff'
                sz, lbl = 20, f"{n_type}"
            y_labels.append(lbl)
            if len(spk_t) > 0: ax0.scatter(spk_t, [i]*len(spk_t), color=col, s=sz, marker='|')
        
        ax0.set_yticks(range(len(raster_ids)))
        ax0.set_yticklabels(y_labels, fontsize=6)
        ax0.set_ylim(-0.5, len(raster_ids)-0.5)
        ax0.invert_yaxis()
        
        # Plot 1: V_mem
        ax1 = axes[1]
        ax1.plot(ts_v, color='cyan', lw=1.5)
        if v_th is not None:
            ax1.axhline(v_th, color='white', ls='--')
        ax1.set_ylabel("mV")
        
        # Plot 2: Currents
        ax2 = axes[2]
        ax2.plot(ts_epsc, color='#ff4d4d', alpha=0.8)
        ax2.plot(ts_ipsc, color='#4d4dff', alpha=0.8)
        ax2.set_ylabel("nA")
        
        # Plot 3: Net
        ax3 = axes[3]
        ax3.plot(ts_psc, color='white', lw=1)
        ax3.fill_between(range(total_steps), ts_psc, 0, where=(ts_psc>=0), color='red', alpha=0.2)
        ax3.fill_between(range(total_steps), ts_psc, 0, where=(ts_psc<0), color='blue', alpha=0.2)
        ax3.set_ylabel("nA")

        for ax in axes:
            ax.axvline(t_step, color='white', ls='--')
            ax.grid(True, ls=':', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)

if __name__ == "__main__":
    # You can instantiate with default paths if you prefer
    # app = SNNVisualizer(network_path="...", simulation_path="...")
    # app = SNNVisualizer(network_path = "/home/liuxingyu/mice_unnamed_torch_dev/output_networks/x_2000_2200_z_2200_2400_Total_4166_HybridGu_eemu1_eesig1.0_eimu1_eisig1.0_iimu1_iisig1.0_ie1.0",
    #     simulation_path = "/home/liuxingyu/mice_unnamed_torch_dev/outputs/2026-01-18/14-03-50/eemu1_eesig1.0_eimu1_eisig1.0_iimu1_iisig1.0_ie1.0/simulation_output"
    # ) 
    app = SNNVisualizer(network_path = "/home/liuxingyu/btorch-examples/mice_v1/tutorial_outputs/2026-04-18/11-24-51/tutorial_assets/connectome",
        simulation_path = "/home/liuxingyu/btorch-examples/mice_v1/tutorial_outputs/2026-04-18/11-24-51/simulation_output"
    ) 
    app.run()
