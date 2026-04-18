"""Checkpoint utilities for saving/loading model state."""

import torch
from pathlib import Path
from btorch.models import functional


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_acc: float,
    path: str | Path,
) -> None:
    """
    Save checkpoint with model state, optimizer state, and memory snapshots.

    Memory reset values are stored separately since dynamic buffers are excluded
    from state_dict().
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
        "memories_rv": functional.named_memory_reset_values(model),
        "memory_values": functional.named_memory_values(model),
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cuda",
    restore_memory_values: bool = False,
) -> tuple[int, float]:
    """
    Load checkpoint and restore model state.

    Returns:
        epoch: Starting epoch
        best_acc: Best accuracy achieved
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Filter out dynamic state keys (already excluded from state_dict, but be safe)
    state_dict = checkpoint["model_state_dict"]
    dynamic_keys = functional.named_memory_values(model).keys()
    for key in dynamic_keys:
        state_dict.pop(key, None)

    # Load model weights
    model.load_state_dict(state_dict, strict=False)

    # Restore memory reset values
    if "memories_rv" in checkpoint:
        functional.set_memory_reset_values(model, checkpoint["memories_rv"])

    # Optional exact runtime-state restoration
    if restore_memory_values and "memory_values" in checkpoint:
        functional.set_memory_values(model, checkpoint["memory_values"])

    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    best_acc = checkpoint.get("best_acc", 0.0)

    return epoch, best_acc


def find_latest_checkpoint(checkpoint_dir: str | Path, task: str) -> Path | None:
    """Find the latest checkpoint for a task."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob(f"{task}_*.pth"))
    if not checkpoints:
        return None

    # Sort by modification time
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0]
