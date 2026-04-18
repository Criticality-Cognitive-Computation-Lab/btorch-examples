"""Population-level connectivity graph (Fig.1C-style) utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from .connectivity import sparseconn_to_coo


@dataclass(frozen=True)
class PopulationEdge:
    src: str
    dst: str
    probability: float
    mean_weight: float


def _pop_stats(block: np.ndarray) -> tuple[float, float]:
    if block.size == 0:
        return 0.0, 0.0
    nz = block != 0.0
    p = float(nz.mean())
    m = float(block[nz].mean()) if nz.any() else 0.0
    return p, m


def build_population_graph(
    model: torch.nn.Module,
) -> tuple[nx.DiGraph, list[PopulationEdge]]:
    """Create a population graph from an instantiated GLIF RSNN model."""
    recurrent = sparseconn_to_coo(model.recurrent_conn).toarray()
    n_e = model.n_e
    n_i = model.n_i
    n_a = min(max(model.n_adapt, 0), model.n_neuron)
    adapt_mask = np.zeros(model.n_neuron, dtype=bool)
    adapt_mask[:n_a] = True

    # recurrent stats
    ee = recurrent[:n_e, :n_e]
    ei = recurrent[:n_e, n_e:]
    ie = recurrent[n_e:, :n_e]
    ii = recurrent[n_e:, n_e:]

    input_w = model.input_linear.weight.detach().cpu().numpy().T
    out_w = model.output_linear.weight.detach().cpu().numpy()

    G = nx.DiGraph()
    G.add_nodes_from(["X", "E", "I", "A", "Y"])
    edges: list[PopulationEdge] = []

    def add(src: str, dst: str, p: float, w: float) -> None:
        edges.append(PopulationEdge(src=src, dst=dst, probability=p, mean_weight=w))
        G.add_edge(src, dst, probability=p, mean_weight=w)

    # X -> E/I
    p_xe = float(np.mean(input_w[:, :n_e] != 0.0))
    w_xe = float(np.mean(input_w[:, :n_e]))
    p_xi = float(np.mean(input_w[:, n_e:] != 0.0)) if n_i > 0 else 0.0
    w_xi = float(np.mean(input_w[:, n_e:])) if n_i > 0 else 0.0
    add("X", "E", p_xe, w_xe)
    if n_i > 0:
        add("X", "I", p_xi, w_xi)

    # recurrent E/I blocks
    p, w = _pop_stats(ee)
    add("E", "E", p, w)
    if n_i > 0:
        p, w = _pop_stats(ei)
        add("E", "I", p, w)
        p, w = _pop_stats(ie)
        add("I", "E", p, w)
        p, w = _pop_stats(ii)
        add("I", "I", p, w)

    # Adaptation population links (membership + effective asc amplitude)
    if n_a > 0:
        asc = model.neuron.asc_amps.detach().cpu().numpy().reshape(model.n_neuron, -1)
        mean_asc = float(asc[adapt_mask].mean())
        add("E", "A", float(adapt_mask[:n_e].mean()), mean_asc)
        if n_i > 0:
            add("I", "A", float(adapt_mask[n_e:].mean()), mean_asc)
        add("A", "E", float(adapt_mask[:n_e].mean()), mean_asc)
        if n_i > 0:
            add("A", "I", float(adapt_mask[n_e:].mean()), mean_asc)

    # E/I/A -> Y
    p_ey = float(np.mean(out_w[:, :n_e] != 0.0))
    w_ey = float(np.mean(out_w[:, :n_e]))
    add("E", "Y", p_ey, w_ey)
    if n_i > 0:
        p_iy = float(np.mean(out_w[:, n_e:] != 0.0))
        w_iy = float(np.mean(out_w[:, n_e:]))
        add("I", "Y", p_iy, w_iy)
    if n_a > 0:
        p_ay = float(np.mean(out_w[:, adapt_mask] != 0.0))
        w_ay = float(np.mean(out_w[:, adapt_mask]))
        add("A", "Y", p_ay, w_ay)

    return G, edges


def plot_population_graph(model: torch.nn.Module, save_path: str | Path) -> Path:
    """Render and save deterministic population-level graph."""
    graph, _ = build_population_graph(model)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    pos = {
        "X": (-1.2, 0.0),
        "E": (0.0, 0.55),
        "I": (0.0, -0.55),
        "A": (0.9, 0.0),
        "Y": (2.0, 0.0),
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=1600,
        node_color=["#d9edf7", "#dff0d8", "#f2dede", "#fcf8e3", "#d9edf7"],
        edgecolors="#222222",
        linewidths=1.2,
        ax=ax,
    )
    nx.draw_networkx_labels(graph, pos, font_size=12, font_weight="bold", ax=ax)

    edge_colors = []
    widths = []
    labels = {}
    for u, v, d in graph.edges(data=True):
        w = d["mean_weight"]
        p = d["probability"]
        edge_colors.append("#1a9850" if w >= 0 else "#d73027")
        widths.append(1.2 + 4.0 * p)
        labels[(u, v)] = f"p={p:.2f}\nmu={w:+.3g}"

    nx.draw_networkx_edges(
        graph,
        pos,
        edge_color=edge_colors,
        width=widths,
        alpha=0.9,
        arrows=True,
        arrowsize=16,
        connectionstyle="arc3,rad=0.08",
        ax=ax,
    )
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, font_size=8, ax=ax)
    ax.set_title("Population Connectivity (Fig.1C style)")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path
