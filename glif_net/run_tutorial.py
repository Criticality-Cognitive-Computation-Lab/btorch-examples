#!/usr/bin/env python
"""
Unified entry point for all GLIF RSNN tutorials.

Usage:
    python -m glif_net.run_tutorial task=seqmnist n_neuron=220
    python -m glif_net.run_tutorial task=shd n_adapt=100
    python -m glif_net.run_tutorial task=speech_command lr=0.001
"""

import sys
import importlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf

from glif_net.src.model import SingleLayerGLIFRSNN
from glif_net.src.loss import CombinedLoss, LossConfig
from glif_net.src.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
)
from glif_net.src.calibration import calibrate_io_scales
from glif_net.src.viz import compute_firing_rate_stats, log_tensorboard_viz


@dataclass
class Config:
    """Configuration with dataclass defaults. CLI overrides only."""

    # Task selection
    task: str = "seqmnist"  # speech_command | shd | seqmnist | poisson_mnist

    # Dataset paths
    data_dir: str = "./data"

    # Network architecture
    n_neuron: int = 256
    n_e_ratio: float = 0.8
    n_adapt: int = 0  # 0 = all adapt, -1 = half, N = first N neurons
    asc_amp: float = -0.2
    tau_adapt: float = 700.0

    # Neuron parameters
    tau_mem: float = 20.0
    tau_syn: float = 5.0
    v_threshold: float = -45.0
    v_reset: float = -60.0
    tau_ref: float = 5.0

    # Input/Output
    input_scale: float = 1.0
    output_scale: float = 1.0
    response_window: float = 0.8
    readout_tau: float = 20.0

    # Training
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    lr_decay_every: int = 2500
    lr_decay_factor: float = 0.8
    weight_decay: float = 0.0
    optimizer: str = "adam"
    max_grad_norm: float = 1.0

    # Simulation
    dt: float = 1.0
    T: int | None = None

    # Device
    device: str = "cuda"

    # Loss weights
    ce_weight: float = 1.0
    voltage_weight: float = 0.01
    rate_weight: float = 0.1
    rate_target_min: float = 2.0
    rate_target_max: float = 30.0

    # Calibration
    calibrate: bool = True
    calibrate_batches: int = 10

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 10
    resume: bool = False
    resume_from: str | None = None

    # Logging
    use_tensorboard: bool = True
    log_dir: str = "./runs"
    log_every: int = 50

    # Visualization
    viz_every: int = 5
    num_viz_samples: int = 3

    # Task-specific (set by task modules)
    n_input_neurons: int = 80
    poisson_rate: float = 100.0
    n_mfcc: int = 20


# Task registry
TASK_MODULES = {
    "speech_command": "glif_net.tutorials.speech_command",
    "shd": "glif_net.tutorials.shd",
    "seqmnist": "glif_net.tutorials.seqmnist",
    "poisson_mnist": "glif_net.tutorials.poisson_mnist",
}


def get_merged_config() -> Config:
    """
    Load config with task-specific defaults applied.

    Priority: CLI > Task Defaults > Base Defaults
    """
    # First parse just the task from CLI
    cli = OmegaConf.from_cli()
    task = cli.get("task", "seqmnist")

    # Get task defaults
    task_module = importlib.import_module(TASK_MODULES[task])
    task_defaults = task_module.get_task_defaults()

    # Build config: base -> task defaults -> CLI
    base_cfg = OmegaConf.structured(Config())
    task_cfg = OmegaConf.create(task_defaults)
    cli_cfg = cli

    merged = OmegaConf.unsafe_merge(base_cfg, task_cfg, cli_cfg)
    return OmegaConf.to_object(merged)


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: CombinedLoss,
    device: torch.device,
    config: Config,
    epoch: int,
    writer: SummaryWriter | None,
) -> tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (x, target) in enumerate(train_loader):
        x = x.to(device)  # (T, batch, input_dim) or (batch, T, input_dim)
        target = target.to(device)

        # Handle different input shapes
        if x.ndim == 3 and x.shape[0] != config.T:
            # Input is (batch, T, input_dim), transpose to (T, batch, input_dim)
            x = x.transpose(0, 1)

        optimizer.zero_grad()

        # Forward pass
        output, states = model(x)

        # Compute loss
        T = x.shape[0]
        loss, loss_dict = loss_fn(output, target, states, T)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        optimizer.step()

        # Stats
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        # Logging
        if writer is not None and batch_idx % config.log_every == 0:
            global_step = epoch * len(train_loader) + batch_idx
            for key, value in loss_dict.items():
                writer.add_scalar(f"loss/{key}", value, global_step)

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    loss_fn: CombinedLoss,
    device: torch.device,
    config: Config,
) -> tuple[float, float, dict]:
    """Evaluate on test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_spikes = []

    for x, target in test_loader:
        x = x.to(device)
        target = target.to(device)

        # Handle different input shapes
        if x.ndim == 3 and x.shape[0] != config.T:
            x = x.transpose(0, 1)

        # Forward pass
        output, states = model(x)

        # Compute loss
        T = x.shape[0]
        loss, _ = loss_fn(output, target, states, T)

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        all_spikes.append(states["spikes"])

    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total

    # Compute firing rate stats
    all_spikes = torch.cat([s.cpu() for s in all_spikes], dim=1)
    stats = compute_firing_rate_stats(all_spikes, config.dt)

    return avg_loss, accuracy, stats


def main():
    # 1. Load merged config
    config = get_merged_config()
    print(f"Task: {config.task}")
    print(
        f"Config: n_neuron={config.n_neuron}, n_adapt={config.n_adapt}, asc_amp={config.asc_amp}"
    )

    # 2. Setup device, logging directories
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    log_dir = Path(config.log_dir) / f"{config.task}_{datetime.now():%Y%m%d_%H%M%S}"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Log dir: {log_dir}")

    # 3. Get dataloaders from task module
    task_module = importlib.import_module(TASK_MODULES[config.task])
    train_loader, test_loader, input_dim, output_dim, T = task_module.get_dataloaders(
        config
    )

    # Override T if task provides it
    if T is not None:
        config.T = T

    print(f"Input dim: {input_dim}, Output dim: {output_dim}, T: {config.T}")

    # 4. Build model
    model = SingleLayerGLIFRSNN(
        input_dim=input_dim,
        output_dim=output_dim,
        n_neuron=config.n_neuron,
        n_e_ratio=config.n_e_ratio,
        n_adapt=config.n_adapt,
        asc_amp=config.asc_amp,
        tau_adapt=config.tau_adapt,
        tau_mem=config.tau_mem,
        tau_syn=config.tau_syn,
        v_threshold=config.v_threshold,
        v_reset=config.v_reset,
        tau_ref=config.tau_ref,
        input_scale=config.input_scale,
        output_scale=config.output_scale,
        response_window=config.response_window,
        readout_tau=config.readout_tau,
        dt=config.dt,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 5. Calibrate
    if config.calibrate:
        print("Calibrating I/O scales...")
        calibrate_io_scales(model, train_loader, device, config.calibrate_batches)
        print(f"  Input scale: {model.input_scale.scale.item():.4f}")
        print(f"  Output scale: {model.output_scale.scale.item():.4f}")

    # 6. Optimizer and scheduler
    if config.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
    elif config.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )

    # StepLR scheduler (decay every N steps)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.lr_decay_every // len(train_loader)
        + 1,  # Convert steps to epochs
        gamma=config.lr_decay_factor,
    )

    # 7. Resume if requested
    start_epoch = 0
    best_acc = 0.0
    if config.resume:
        if config.resume_from:
            checkpoint_path = Path(config.resume_from)
        else:
            checkpoint_path = find_latest_checkpoint(config.checkpoint_dir, config.task)

        if checkpoint_path and checkpoint_path.exists():
            print(f"Resuming from checkpoint: {checkpoint_path}")
            start_epoch, best_acc = load_checkpoint(
                checkpoint_path, model, optimizer, device
            )
            start_epoch += 1
            print(f"  Starting from epoch {start_epoch}, best_acc={best_acc:.2f}%")

    # 8. Training loop
    writer = SummaryWriter(log_dir) if config.use_tensorboard else None
    loss_config = LossConfig(
        ce_weight=config.ce_weight,
        voltage_weight=config.voltage_weight,
        rate_weight=config.rate_weight,
        rate_target_min=config.rate_target_min,
        rate_target_max=config.rate_target_max,
        v_reset=config.v_reset,
    )
    loss_fn = CombinedLoss(loss_config, config.dt)

    print(f"\nStarting training for {config.epochs} epochs...")

    for epoch in range(start_epoch, config.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, loss_fn, device, config, epoch, writer
        )
        test_loss, test_acc, stats = evaluate(
            model, test_loader, loss_fn, device, config
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Logging
        print(
            f"Epoch {epoch + 1}/{config.epochs}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.2f}%, "
            f"lr={current_lr:.6f}, rate={stats['mean_rate_hz']:.2f}Hz"
        )

        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/acc", train_acc, epoch)
            writer.add_scalar("test/loss", test_loss, epoch)
            writer.add_scalar("test/acc", test_acc, epoch)
            writer.add_scalar("train/lr", current_lr, epoch)
            writer.add_scalar("stats/firing_rate_hz", stats["mean_rate_hz"], epoch)
            writer.add_scalar(
                "scales/input_scale", model.input_scale.scale.item(), epoch
            )
            writer.add_scalar(
                "scales/output_scale", model.output_scale.scale.item(), epoch
            )

        # Checkpointing
        if (epoch + 1) % config.save_every == 0 or test_acc > best_acc:
            is_best = test_acc > best_acc
            if is_best:
                best_acc = test_acc

            checkpoint_path = checkpoint_dir / f"{config.task}_epoch{epoch + 1:03d}.pth"
            save_checkpoint(model, optimizer, epoch, best_acc, checkpoint_path)

            if is_best:
                best_path = checkpoint_dir / f"{config.task}_best.pth"
                save_checkpoint(model, optimizer, epoch, best_acc, best_path)

        # Visualization
        if config.use_tensorboard and (epoch + 1) % config.viz_every == 0:
            # Get a sample batch for visualization
            x, _ = next(iter(test_loader))
            if x.ndim == 3 and x.shape[0] != config.T:
                x = x.transpose(0, 1)
            x = x.to(device)

            with torch.no_grad():
                _, states = model(x)

            log_tensorboard_viz(writer, states["spikes"], states["voltage"], epoch)

    # 9. Final evaluation
    final_loss, final_acc, final_stats = evaluate(
        model, test_loader, loss_fn, device, config
    )
    print(f"\nFinal test accuracy: {final_acc:.2f}%")
    print(f"Final firing rate: {final_stats['mean_rate_hz']:.2f} Hz")

    if writer is not None:
        writer.close()

    # Save final checkpoint
    final_path = checkpoint_dir / f"{config.task}_final.pth"
    save_checkpoint(model, optimizer, config.epochs, final_acc, final_path)
    print(f"\nTraining complete. Checkpoint saved to {final_path}")


if __name__ == "__main__":
    main()
