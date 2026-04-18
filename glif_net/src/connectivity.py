"""Connectivity utilities for GLIF RSNN models.

This module mirrors the E/I sparse construction pattern used in btorch tests,
but keeps production code self-contained for tutorials/experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse
import torch


@dataclass(frozen=True)
class ConnectivitySpec:
    """Structured configuration for E/I sparse connectivity generation."""

    density: float = 1.0
    i_e_ratio: float = 100.0
    e_to_e_mean: float = 4.0e-3
    e_to_e_std: float = 1.9e-3
    e_i_mean: float = 5.0e-2
    i_i_mean: float = 25e-4


def _apply_density(
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    density: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if density >= 1.0:
        return rows, cols, vals
    keep = int(rows.size * density)
    if keep <= 0:
        return rows[:0], cols[:0], vals[:0]
    idx = rng.choice(rows.size, keep, replace=False)
    return rows[idx], cols[idx], vals[idx]


def build_sparse_mat(
    n_e_neurons: int,
    n_i_neurons: int,
    split: bool = False,
    density: float = 1.0,
    i_e_ratio: float = 100.0,
    e_to_e_mean: float = 4.0e-3,
    e_to_e_std: float = 1.9e-3,
    e_i_mean: float = 5.0e-2,
    i_i_mean: float = 25e-4,
    seed: int | None = None,
) -> (
    tuple[scipy.sparse.coo_array, np.ndarray, np.ndarray]
    | tuple[
        scipy.sparse.coo_array,
        scipy.sparse.coo_array,
        np.ndarray,
        np.ndarray,
    ]
):
    """Build E/I structured sparse matrix with Dale-compatible signs.

    Matrix shape is (n_src, n_dst): rows are pre-synaptic neurons.
    E rows are non-negative, I rows are non-positive.
    """
    if not 0.0 <= density <= 1.0:
        raise ValueError("density must be between 0 and 1")
    if n_e_neurons <= 0 or n_i_neurons < 0:
        raise ValueError("invalid neuron counts")

    rng = np.random.default_rng(seed)
    n_neurons = n_e_neurons + n_i_neurons
    e_idx = np.arange(n_e_neurons)
    i_idx = np.arange(n_e_neurons, n_neurons)

    all_rows: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    all_vals: list[np.ndarray] = []

    e_rows_list: list[np.ndarray] = []
    e_cols_list: list[np.ndarray] = []
    e_vals_list: list[np.ndarray] = []
    i_rows_list: list[np.ndarray] = []
    i_cols_list: list[np.ndarray] = []
    i_vals_list: list[np.ndarray] = []

    # E -> E (log-normal)
    m = np.log(e_to_e_mean**2 / np.sqrt(e_to_e_std**2 + e_to_e_mean**2))
    s = np.sqrt(np.log(1.0 + (e_to_e_std**2 / e_to_e_mean**2)))
    e_to_e_weights = np.exp(rng.normal(loc=m, scale=s, size=(n_e_neurons, n_e_neurons)))

    e_rows, e_cols = np.meshgrid(e_idx, e_idx, indexing="ij")
    rows = e_rows.flatten()
    cols = e_cols.flatten()
    vals = e_to_e_weights.flatten()
    rows, cols, vals = _apply_density(rows, cols, vals, density, rng)
    all_rows.append(rows)
    all_cols.append(cols)
    all_vals.append(vals)
    e_rows_list.append(rows)
    e_cols_list.append(cols)
    e_vals_list.append(vals)

    # I -> E (strong negative)
    mean_e_weights = e_to_e_weights.mean(axis=0)
    mean_i_weights = mean_e_weights * i_e_ratio
    for e_neuron in e_idx:
        vals = np.abs(
            rng.normal(
                loc=mean_i_weights[e_neuron],
                scale=max(mean_i_weights[e_neuron] * 0.25, 1e-12),
                size=n_i_neurons,
            )
        )
        rows = i_idx.copy()
        cols = np.full(n_i_neurons, e_neuron)
        rows, cols, vals = _apply_density(rows, cols, vals, density, rng)
        all_rows.append(rows)
        all_cols.append(cols)
        all_vals.append(-vals)
        i_rows_list.append(rows)
        i_cols_list.append(cols)
        i_vals_list.append(vals)

    # E -> I (homogeneous positive)
    e_rows, i_cols = np.meshgrid(e_idx, i_idx, indexing="ij")
    rows = e_rows.flatten()
    cols = i_cols.flatten()
    vals = np.full(rows.size, e_i_mean, dtype=np.float64)
    rows, cols, vals = _apply_density(rows, cols, vals, density, rng)
    all_rows.append(rows)
    all_cols.append(cols)
    all_vals.append(vals)
    e_rows_list.append(rows)
    e_cols_list.append(cols)
    e_vals_list.append(vals)

    # I -> I (homogeneous negative)
    i_rows, i_cols = np.meshgrid(i_idx, i_idx, indexing="ij")
    rows = i_rows.flatten()
    cols = i_cols.flatten()
    vals = np.full(rows.size, i_i_mean, dtype=np.float64)
    rows, cols, vals = _apply_density(rows, cols, vals, density, rng)
    all_rows.append(rows)
    all_cols.append(cols)
    all_vals.append(-vals)
    i_rows_list.append(rows)
    i_cols_list.append(cols)
    i_vals_list.append(vals)

    full_matrix = scipy.sparse.coo_array(
        (
            np.concatenate(all_vals),
            (np.concatenate(all_rows), np.concatenate(all_cols)),
        ),
        shape=(n_neurons, n_neurons),
    )

    if not split:
        return full_matrix, e_idx, i_idx

    e_matrix = scipy.sparse.coo_array(
        (
            np.concatenate(e_vals_list),
            (np.concatenate(e_rows_list), np.concatenate(e_cols_list)),
        ),
        shape=(n_neurons, n_neurons),
    )
    i_matrix = scipy.sparse.coo_array(
        (
            np.concatenate(i_vals_list),
            (np.concatenate(i_rows_list), np.concatenate(i_cols_list)),
        ),
        shape=(n_neurons, n_neurons),
    )
    return e_matrix, i_matrix, e_idx, i_idx


def sparseconn_dale_violations(
    sparse_conn: torch.nn.Module,
    n_e_neurons: int,
) -> dict[str, float]:
    """Return count/fraction of Dale-law sign violations."""
    if not hasattr(sparse_conn, "magnitude") or not hasattr(sparse_conn, "indices"):
        return {"count": 0.0, "fraction": 0.0}

    src_idx = sparse_conn.indices[1]
    w = sparse_conn.magnitude.detach()
    excitatory = src_idx < n_e_neurons
    bad_e = (w[excitatory] < 0).sum().item()
    bad_i = (w[~excitatory] > 0).sum().item()
    bad = float(bad_e + bad_i)
    return {
        "count": bad,
        "fraction": bad / float(max(w.numel(), 1)),
    }


def sparseconn_to_coo(sparse_conn: torch.nn.Module) -> scipy.sparse.coo_array:
    """Export current sparse-connection weights as scipy COO."""
    if not hasattr(sparse_conn, "indices") or not hasattr(sparse_conn, "magnitude"):
        raise TypeError("sparse_conn must expose indices and magnitude")
    out_features = int(sparse_conn.out_features)
    in_features = int(sparse_conn.in_features)
    # stored layout is transposed, convert back to (src, dst)
    row = sparse_conn.indices[1].detach().cpu().numpy()
    col = sparse_conn.indices[0].detach().cpu().numpy()
    val = sparse_conn.magnitude.detach().cpu().numpy()
    return scipy.sparse.coo_array((val, (row, col)), shape=(in_features, out_features))
