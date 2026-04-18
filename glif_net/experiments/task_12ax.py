"""12AX experiment with GLIF RSNN (BPTT + Dale constraint)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from glif_net.src.model import SingleLayerGLIFRSNN

SYMBOLS = ("1", "2", "A", "B", "C", "X", "Y", "Z")
SYM_TO_IDX = {s: i for i, s in enumerate(SYMBOLS)}
L_CLASS = 0
R_CLASS = 1


@dataclass
class Config:
    # Run setup
    seed: int = 7
    device: str = "cuda"
    profile: str = "quick"  # quick | paper
    output_dir: str = "./outputs/12ax"

    # Task
    episode_symbols: int = 90
    symbol_ms: int = 500
    input_neurons_per_symbol: int = 5
    poisson_rate_hz: float = 200.0
    dt: float = 1.0

    # Model
    n_neuron: int = 200
    n_adapt: int = 100
    n_e_ratio: float = 0.8
    tau_mem: float = 20.0
    tau_syn: float = 5.0
    tau_ref: float = 5.0
    v_threshold: float = -30.0
    v_reset: float = -60.0
    asc_amp: float = -0.00425
    tau_adapt_min: float = 1.0
    tau_adapt_max: float = 13500.0

    # Optimization
    iterations: int = 10_000
    batch_size: int = 20
    lr: float = 1e-3
    max_grad_norm: float = 1.0

    # Regularization
    rate_reg_coeff: float = 15.0
    target_rate_hz: float = 10.0

    # Eval
    eval_every: int = 200
    eval_batches: int = 10


def parse_config() -> Config:
    cli = OmegaConf.from_cli()
    cfg = OmegaConf.merge(OmegaConf.structured(Config()), cli)
    out = OmegaConf.to_object(cfg)
    assert isinstance(out, Config)
    if out.profile == "quick":
        out.symbol_ms = min(out.symbol_ms, 60)
        out.iterations = min(out.iterations, 60)
        out.batch_size = min(out.batch_size, 4)
        out.eval_every = min(out.eval_every, 20)
        out.eval_batches = min(out.eval_batches, 2)
    return out


def set_seed(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


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
    """Compute L/R targets for 12AX finite-state rule."""
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
    """Encode one episode into Poisson spikes: (T, input_dim)."""
    n_symbols = len(SYMBOLS)
    input_dim = n_symbols * n_per_symbol
    T = len(sequence) * symbol_ms
    x = torch.zeros((T, input_dim), device=device)
    p = min(max(rate_hz * dt / 1000.0, 0.0), 1.0)
    for s_idx, sym in enumerate(sequence):
        t0 = s_idx * symbol_ms
        t1 = t0 + symbol_ms
        i0 = SYM_TO_IDX[sym] * n_per_symbol
        i1 = i0 + n_per_symbol
        x[t0:t1, i0:i1] = (torch.rand((symbol_ms, n_per_symbol), device=device) < p).to(
            x.dtype
        )
    return x


def generate_batch(
    cfg: Config, rng: np.random.Generator, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    input_dim = len(SYMBOLS) * cfg.input_neurons_per_symbol
    T = cfg.episode_symbols * cfg.symbol_ms
    x = torch.zeros((T, cfg.batch_size, input_dim), device=device)
    y = torch.zeros(
        (cfg.episode_symbols, cfg.batch_size), dtype=torch.long, device=device
    )
    for b in range(cfg.batch_size):
        seq = generate_symbol_sequence(cfg.episode_symbols, rng)
        x[:, b] = encode_poisson_episode(
            seq,
            symbol_ms=cfg.symbol_ms,
            n_per_symbol=cfg.input_neurons_per_symbol,
            rate_hz=cfg.poisson_rate_hz,
            dt=cfg.dt,
            device=device,
        )
        y[:, b] = compute_targets(seq).to(device)
    return x, y


def aggregate_symbol_logits(
    sequence_logits: torch.Tensor, symbol_ms: int
) -> torch.Tensor:
    # sequence_logits: (T, B, 2) -> (S, B, 2)
    T, B, C = sequence_logits.shape
    S = T // symbol_ms
    return sequence_logits.view(S, symbol_ms, B, C).mean(dim=1)


def build_model(cfg: Config, device: torch.device) -> SingleLayerGLIFRSNN:
    input_dim = len(SYMBOLS) * cfg.input_neurons_per_symbol
    model = SingleLayerGLIFRSNN(
        input_dim=input_dim,
        output_dim=2,
        n_neuron=cfg.n_neuron,
        n_adapt=cfg.n_adapt,
        n_e_ratio=cfg.n_e_ratio,
        tau=cfg.tau_mem,
        tau_syn=cfg.tau_syn,
        tau_ref=cfg.tau_ref,
        v_threshold=cfg.v_threshold,
        v_reset=cfg.v_reset,
        asc_amp=cfg.asc_amp,
        tau_adapt=700.0,
        tau_adapt_min=cfg.tau_adapt_min,
        tau_adapt_max=cfg.tau_adapt_max,
        dt=cfg.dt,
        grad_checkpoint=True,
        chunk_size=1000,
    ).to(device)
    model.init_voltage()
    return model


def compute_rate_regularization(
    spikes: torch.Tensor, dt: float, target_rate_hz: float
) -> torch.Tensor:
    # spikes: (T, B, N)
    T, B, _ = spikes.shape
    duration_sec = T * dt / 1000.0
    rates = spikes.sum(dim=(0, 1)) / max(B * duration_sec, 1e-12)
    return ((rates - target_rate_hz) ** 2).mean()


@torch.no_grad()
def evaluate(
    model: SingleLayerGLIFRSNN,
    cfg: Config,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses = []
    sym_acc = []
    ep_acc = []
    for _ in range(cfg.eval_batches):
        x, y = generate_batch(cfg, rng, device)
        seq_logits, states = model(x, return_sequence=True)
        symbol_logits = aggregate_symbol_logits(seq_logits, cfg.symbol_ms)
        ce = F.cross_entropy(symbol_logits.reshape(-1, 2), y.reshape(-1))
        rate_reg = compute_rate_regularization(
            states["spikes"], cfg.dt, cfg.target_rate_hz
        )
        loss = ce + cfg.rate_reg_coeff * rate_reg
        pred = symbol_logits.argmax(dim=-1)
        losses.append(float(loss.item()))
        sym_acc.append(float((pred == y).float().mean().item()))
        ep_acc.append(float((pred == y).all(dim=0).float().mean().item()))
    return {
        "loss": float(np.mean(losses)),
        "symbol_acc": float(np.mean(sym_acc)),
        "episode_success": float(np.mean(ep_acc)),
    }


def train(cfg: Config) -> Path:
    rng = set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    run_dir = (
        Path(cfg.output_dir) / f"12ax_{datetime.now():%Y%m%d_%H%M%S}_seed{cfg.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    rows: list[dict[str, float | int]] = []
    for it in range(1, cfg.iterations + 1):
        model.train()
        x, y = generate_batch(cfg, rng, device)

        optimizer.zero_grad()
        seq_logits, states = model(x, return_sequence=True)
        symbol_logits = aggregate_symbol_logits(seq_logits, cfg.symbol_ms)

        loss_ce = F.cross_entropy(symbol_logits.reshape(-1, 2), y.reshape(-1))
        loss_rate = compute_rate_regularization(
            states["spikes"], cfg.dt, cfg.target_rate_hz
        )
        loss = loss_ce + cfg.rate_reg_coeff * loss_rate

        loss.backward()
        if cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()
        model.apply_dale_projection()

        pred = symbol_logits.argmax(dim=-1)
        train_sym_acc = float((pred == y).float().mean().item())
        train_ep_succ = float((pred == y).all(dim=0).float().mean().item())

        row: dict[str, float | int] = {
            "iter": it,
            "train_loss": float(loss.item()),
            "train_ce": float(loss_ce.item()),
            "train_rate": float(loss_rate.item()),
            "train_symbol_acc": train_sym_acc,
            "train_episode_success": train_ep_succ,
            "dale_violation_fraction": float(model.dale_violations()["fraction"]),
        }

        if it % cfg.eval_every == 0 or it == cfg.iterations:
            eval_stats = evaluate(model, cfg, rng, device)
            row.update(
                {
                    "eval_loss": eval_stats["loss"],
                    "eval_symbol_acc": eval_stats["symbol_acc"],
                    "eval_episode_success": eval_stats["episode_success"],
                }
            )
            print(
                f"iter={it:05d} train_sym={train_sym_acc:.4f} "
                f"eval_sym={eval_stats['symbol_acc']:.4f} eval_ep={eval_stats['episode_success']:.4f}"
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "metrics.csv", index=False)

    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    torch.save({"model_state_dict": model.state_dict()}, run_dir / "model_final.pt")
    return run_dir


def main() -> None:
    cfg = parse_config()
    out_dir = train(cfg)
    print(f"Run complete: {out_dir}")


if __name__ == "__main__":
    main()
