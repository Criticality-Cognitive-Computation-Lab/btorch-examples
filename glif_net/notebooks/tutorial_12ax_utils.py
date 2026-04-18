"""Utility helpers for the 12AX beginner tutorial notebook.

This module intentionally keeps repetitive data and plotting utilities out of the
notebook so cells can focus on btorch-specific learning content.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

SYMBOLS = ("1", "2", "A", "B", "C", "X", "Y", "Z")
SYM_TO_IDX = {s: i for i, s in enumerate(SYMBOLS)}
L_CLASS = 0
R_CLASS = 1


def _sample_chunk(rng: np.random.Generator) -> list[str]:
    # [12][ABCXYZ]{1,10} ((A[CZ]{0,6}X|B[CZ]{0,6}Y)|([ABC][XYZ])){1,2}
    chunk: list[str] = [str(rng.choice(["1", "2"]))]
    n_noise = int(rng.integers(1, 11))
    chunk.extend(rng.choice(list("ABCXYZ"), size=n_noise).tolist())
    n_blocks = int(rng.integers(1, 3))
    for _ in range(n_blocks):
        if rng.random() < 0.5:
            if rng.random() < 0.5:
                middle_n = int(rng.integers(0, 7))
                middle = rng.choice(list("CZ"), size=middle_n).tolist()
                chunk.extend(["A", *middle, "X"])
            else:
                middle_n = int(rng.integers(0, 7))
                middle = rng.choice(list("CZ"), size=middle_n).tolist()
                chunk.extend(["B", *middle, "Y"])
        else:
            chunk.append(str(rng.choice(list("ABC"))))
            chunk.append(str(rng.choice(list("XYZ"))))
    return chunk


def generate_symbol_sequence(length: int, rng: np.random.Generator) -> list[str]:
    seq: list[str] = []
    while len(seq) < length:
        seq.extend(_sample_chunk(rng))
    return seq[:length]


def compute_targets(sequence: list[str]) -> torch.Tensor:
    """Compute L/R targets using the 12AX finite-state rule."""
    context = "1"
    pending: str | None = None
    y = torch.full((len(sequence),), L_CLASS, dtype=torch.long)

    for i, sym in enumerate(sequence):
        if sym in ("1", "2"):
            context = sym
            pending = None
            y[i] = L_CLASS
        elif sym in ("A", "B"):
            if (context == "1" and sym == "A") or (context == "2" and sym == "B"):
                pending = sym
            else:
                pending = None
            y[i] = L_CLASS
        elif sym in ("X", "Y"):
            if (context == "1" and pending == "A" and sym == "X") or (
                context == "2" and pending == "B" and sym == "Y"
            ):
                y[i] = R_CLASS
            else:
                y[i] = L_CLASS
            pending = None
        else:
            y[i] = L_CLASS

    return y


def encode_poisson_episode(
    sequence: list[str],
    symbol_ms: int,
    n_per_symbol: int,
    rate_hz: float,
    dt: float,
    device: torch.device,
) -> torch.Tensor:
    """Encode one symbol episode into Poisson spikes with shape (T, input_dim)."""
    input_dim = len(SYMBOLS) * n_per_symbol
    T = len(sequence) * symbol_ms
    x = torch.zeros((T, input_dim), device=device)
    p = min(max(rate_hz * dt / 1000.0, 0.0), 1.0)

    for s_idx, sym in enumerate(sequence):
        t0 = s_idx * symbol_ms
        t1 = t0 + symbol_ms
        i0 = SYM_TO_IDX[sym] * n_per_symbol
        i1 = i0 + n_per_symbol
        spikes = torch.rand((symbol_ms, n_per_symbol), device=device) < p
        x[t0:t1, i0:i1] = spikes.to(x.dtype)

    return x


def generate_episode_tensors(
    episode_symbols: int,
    symbol_ms: int,
    input_neurons_per_symbol: int,
    poisson_rate_hz: float,
    dt: float,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Sample one episode and return symbols, encoded spikes, and symbol targets."""
    sequence = generate_symbol_sequence(episode_symbols, rng)
    x = encode_poisson_episode(
        sequence,
        symbol_ms=symbol_ms,
        n_per_symbol=input_neurons_per_symbol,
        rate_hz=poisson_rate_hz,
        dt=dt,
        device=device,
    )
    y = compute_targets(sequence).to(device)
    return sequence, x, y


def generate_batch(
    cfg: object,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a batch with x:(T,B,input_dim), y:(S,B)."""
    input_dim = len(SYMBOLS) * int(cfg.input_neurons_per_symbol)
    T = int(cfg.episode_symbols) * int(cfg.symbol_ms)
    B = int(cfg.batch_size)

    x = torch.zeros((T, B, input_dim), device=device)
    y = torch.zeros((int(cfg.episode_symbols), B), dtype=torch.long, device=device)

    for b in range(B):
        _, xb, yb = generate_episode_tensors(
            episode_symbols=int(cfg.episode_symbols),
            symbol_ms=int(cfg.symbol_ms),
            input_neurons_per_symbol=int(cfg.input_neurons_per_symbol),
            poisson_rate_hz=float(cfg.poisson_rate_hz),
            dt=float(cfg.dt),
            rng=rng,
            device=device,
        )
        x[:, b] = xb
        y[:, b] = yb

    return x, y


def aggregate_symbol_logits(
    sequence_logits: torch.Tensor, symbol_ms: int
) -> torch.Tensor:
    """Convert timestep logits (T,B,C) to symbol logits (S,B,C) by window mean."""
    if sequence_logits.ndim != 3:
        raise ValueError("sequence_logits must be (T, B, C)")
    T, _, _ = sequence_logits.shape
    if T % symbol_ms != 0:
        raise ValueError("T must be divisible by symbol_ms")
    S = T // symbol_ms
    return sequence_logits.view(
        S, symbol_ms, sequence_logits.shape[1], sequence_logits.shape[2]
    ).mean(dim=1)


def plot_symbol_target_timeline(
    sequence: list[str],
    targets: torch.Tensor,
    max_symbols: int = 100,
) -> None:
    """Plot sampled symbol stream and corresponding L/R targets."""
    n = min(max_symbols, len(sequence))
    symbols = sequence[:n]
    target_np = targets[:n].detach().cpu().numpy()

    x = np.arange(n)
    symbol_vals = np.array([SYM_TO_IDX[s] for s in symbols], dtype=np.int64)

    fig, axes = plt.subplots(
        2, 1, figsize=(13, 5), sharex=True, constrained_layout=True
    )
    axes[0].step(x, symbol_vals, where="mid", color="#176087", linewidth=1.6)
    axes[0].set_yticks(range(len(SYMBOLS)), SYMBOLS)
    axes[0].set_ylabel("Symbol")
    axes[0].set_title("Sampled 12AX Symbol Stream")
    axes[0].grid(alpha=0.25)

    axes[1].step(x, target_np, where="mid", color="#A33F3F", linewidth=1.8)
    axes[1].set_yticks([0, 1], ["L", "R"])
    axes[1].set_xlabel("Symbol index")
    axes[1].set_ylabel("Target")
    axes[1].set_title("Target Sequence (R only on context-matching AX / BY)")
    axes[1].grid(alpha=0.25)
    plt.show()


def plot_training_curves(metrics_df: pd.DataFrame) -> None:
    """Plot compact training/eval curves for loss and accuracies."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    axes[0].plot(
        metrics_df["iter"],
        metrics_df["train_loss"],
        label="train_loss",
        color="#176087",
    )
    if "eval_loss" in metrics_df:
        axes[0].plot(
            metrics_df["iter"],
            metrics_df["eval_loss"],
            label="eval_loss",
            color="#8B5A2B",
        )
    axes[0].set_title("Loss")
    axes[0].set_xlabel("iteration")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(
        metrics_df["iter"],
        metrics_df["train_symbol_acc"],
        label="train_symbol_acc",
        color="#176087",
    )
    if "eval_symbol_acc" in metrics_df:
        axes[1].plot(
            metrics_df["iter"],
            metrics_df["eval_symbol_acc"],
            label="eval_symbol_acc",
            color="#8B5A2B",
        )
    axes[1].set_title("Symbol Accuracy")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylim(0.0, 1.01)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    axes[2].plot(
        metrics_df["iter"],
        metrics_df["train_episode_success"],
        label="train_episode_success",
        color="#176087",
    )
    if "eval_episode_success" in metrics_df:
        axes[2].plot(
            metrics_df["iter"],
            metrics_df["eval_episode_success"],
            label="eval_episode_success",
            color="#8B5A2B",
        )
    axes[2].set_title("Episode Success")
    axes[2].set_xlabel("iteration")
    axes[2].set_ylim(0.0, 1.01)
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc="best")

    plt.show()


def plot_scale_history(scale_values: list[float]) -> None:
    """Plot trajectory of learnable input scale."""
    fig, ax = plt.subplots(figsize=(7.5, 3.5), constrained_layout=True)
    ax.plot(
        np.arange(1, len(scale_values) + 1),
        scale_values,
        color="#2D7D46",
        linewidth=2.0,
    )
    ax.set_title("Learnable input_scale trajectory")
    ax.set_xlabel("iteration")
    ax.set_ylabel("scale value")
    ax.grid(alpha=0.25)
    plt.show()


def plot_spike_raster(
    spikes: torch.Tensor,
    max_time: int = 400,
    max_neurons: int = 60,
    batch_index: int = 0,
) -> None:
    """Plot a lightweight spike raster subset from spikes:(T,B,N)."""
    spikes_cpu = spikes.detach().cpu()
    T = min(max_time, spikes_cpu.shape[0])
    N = min(max_neurons, spikes_cpu.shape[2])

    s = spikes_cpu[:T, batch_index, :N].T
    rows, cols = torch.where(s > 0)

    fig, ax = plt.subplots(figsize=(11, 3.8), constrained_layout=True)
    ax.scatter(cols.numpy(), rows.numpy(), s=4, color="#1D4966")
    ax.set_title(f"Spike raster (batch={batch_index}, neurons 0..{N - 1}, t<={T})")
    ax.set_xlabel("time step")
    ax.set_ylabel("neuron index")
    ax.grid(alpha=0.15)
    plt.show()
