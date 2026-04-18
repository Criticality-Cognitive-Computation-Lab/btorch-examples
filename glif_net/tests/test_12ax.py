from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from glif_net.experiments.task_12ax import (  # noqa: E402
    Config,
    aggregate_symbol_logits,
    compute_targets,
    encode_poisson_episode,
    generate_batch,
    generate_symbol_sequence,
)


def test_12ax_fsm_outputs_expected_r_positions():
    seq = list("1AX2BY1ABX2BZY")
    y = compute_targets(seq)
    # R on exact AX under context 1 and BY under context 2
    r_positions = [i for i, v in enumerate(y.tolist()) if v == 1]
    assert r_positions == [2, 5, 13]


def test_12ax_generator_returns_valid_symbols_and_length():
    rng = np.random.default_rng(0)
    seq = generate_symbol_sequence(90, rng)
    assert len(seq) == 90
    assert set(seq).issubset(set("12ABCXYZ"))


def test_12ax_batch_shapes_and_symbol_window_reduce():
    cfg = Config(profile="quick", batch_size=2, episode_symbols=20, symbol_ms=15)
    rng = np.random.default_rng(1)
    device = torch.device("cpu")
    x, y = generate_batch(cfg, rng, device)
    assert x.shape == (cfg.episode_symbols * cfg.symbol_ms, 2, 40)
    assert y.shape == (cfg.episode_symbols, 2)

    logits = torch.randn(x.shape[0], x.shape[1], 2)
    reduced = aggregate_symbol_logits(logits, cfg.symbol_ms)
    assert reduced.shape == (cfg.episode_symbols, 2, 2)


def test_poisson_encoder_only_activates_symbol_subset():
    seq = ["1", "A", "X"]
    x = encode_poisson_episode(
        seq,
        symbol_ms=10,
        n_per_symbol=5,
        rate_hz=200.0,
        dt=1.0,
        device=torch.device("cpu"),
    )
    assert x.shape == (30, 40)
    # first symbol is "1", active subset should be first 5 inputs only
    assert torch.all(x[:10, 5:] == 0)
