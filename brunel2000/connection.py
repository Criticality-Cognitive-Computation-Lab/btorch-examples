"""Sparse E/I connectivity builders for Model A and Model B."""

import numpy as np
import pandas as pd
import scipy.sparse
from btorch.connectome.connection import make_hetersynapse_conn


def _sample_without_self(
    rng: np.random.Generator,
    start: int,
    stop: int,
    size: int,
    post: int,
) -> np.ndarray:
    candidates = np.arange(start, stop, dtype=np.int64)
    if start <= post < stop:
        candidates = candidates[candidates != post]
    return rng.choice(candidates, size=size, replace=False)


def build_model_a_conn(
    n_e: int, n_i: int, c_e: int, c_i: int, j: float, g: float, dt_ms: float, seed: int
) -> scipy.sparse.coo_array:
    """Build sparse recurrent weight matrix for Model A.

    Each neuron receives c_e random E inputs at +J and c_i random I inputs at
    -g*J. No self-connections.
    """
    rng = np.random.default_rng(seed)
    n_neuron = n_e + n_i
    rows, cols, data = [], [], []

    for post in range(n_neuron):
        # E inputs
        pre_e = _sample_without_self(rng, 0, n_e, c_e, post)
        for pre in pre_e:
            rows.append(pre)
            cols.append(post)
            data.append(j)

        # I inputs
        pre_i = _sample_without_self(rng, n_e, n_neuron, c_i, post)
        for pre in pre_i:
            rows.append(pre)
            cols.append(post)
            data.append(-g * j)

    return scipy.sparse.coo_array((data, (rows, cols)), shape=(n_neuron, n_neuron))


def build_model_b_conn(
    n_e: int,
    n_i: int,
    c_e: int,
    c_i: int,
    j_e: float,
    j_i: float,
    g_e: float,
    g_i: float,
    d_ee_ms: float,
    d_ei_ms: float,
    d_ie_ms: float,
    d_ii_ms: float,
    dt_ms: float,
    seed: int,
):
    """Build sparse recurrent connectivity for Model B with heterogeneous delays.

    Returns:
        conn_d: delay-expanded sparse matrix
        receptor_idx: DataFrame mapping receptor_index to pre/post types
        delays_ms: 1-D array of delays per original non-zero entry (ms)
        n_delay_bins: number of delay bins used
    """
    rng = np.random.default_rng(seed)
    n_neuron = n_e + n_i

    # Build neurons DataFrame
    neurons_df = pd.DataFrame(
        {
            "simple_id": range(n_neuron),
            "EI": ["E"] * n_e + ["I"] * n_i,
        }
    )

    max_delays = {
        ("E", "E"): d_ee_ms,
        ("E", "I"): d_ei_ms,
        ("I", "E"): d_ie_ms,
        ("I", "I"): d_ii_ms,
    }

    rows, cols, weights, delays_ms = [], [], [], []

    for post in range(n_neuron):
        post_type = "E" if post < n_e else "I"

        # E inputs: E->E has J_E, E->I has J_I (paper Model B)
        pre_e = _sample_without_self(rng, 0, n_e, c_e, post)
        for pre in pre_e:
            e_weight = j_e if post_type == "E" else j_i
            rows.append(pre)
            cols.append(post)
            weights.append(e_weight)
            max_d = max_delays[("E", post_type)]
            delays_ms.append(rng.uniform(0.0, max_d))

        # I inputs: I->E has -g_E*J_E, I->I has -g_I*J_I (paper Model B)
        pre_i = _sample_without_self(rng, n_e, n_neuron, c_i, post)
        for pre in pre_i:
            base_j = j_e if post_type == "E" else j_i
            g_val = g_e if post_type == "E" else g_i
            rows.append(pre)
            cols.append(post)
            weights.append(-g_val * base_j)
            max_d = max_delays[("I", post_type)]
            delays_ms.append(rng.uniform(0.0, max_d))

    delays_ms = np.array(delays_ms, dtype=np.float64)
    delay_steps = np.clip((delays_ms / dt_ms).astype(int), 0, None)

    max_delay_step = int(delay_steps.max()) if len(delay_steps) > 0 else 0
    n_delay_bins = max_delay_step + 1

    connections_df = pd.DataFrame(
        {
            "pre_simple_id": rows,
            "post_simple_id": cols,
            "syn_count": weights,
            "delay_steps": delay_steps,
        }
    )

    conn_d, receptor_idx = make_hetersynapse_conn(
        neurons_df,
        connections_df,
        receptor_type_col="EI",
        receptor_type_mode="neuron",
        delay_col="delay_steps",
        n_delay_bins=n_delay_bins,
    )

    return conn_d, receptor_idx, delays_ms, n_delay_bins
