"""
Checkpoint utilities for KhatriVoice.
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


def save_checkpoint(
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    loss: float,
    config: Any,
    is_best: bool = False,
    max_checkpoints: int = 5,
) -> None:
    """
    Save a training checkpoint.

    Args:
        checkpoint_dir: Directory to save checkpoint
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Learning rate scheduler
        step: Current training step
        loss: Current loss value
        config: Training configuration
        is_best: Whether this is the best checkpoint so far
        max_checkpoints: Maximum number of checkpoints to keep
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "config": config.to_dict() if hasattr(config, "to_dict") else vars(config),
        "timestamp": datetime.now().isoformat(),
    }

    # Save checkpoint
    checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

    # Save latest
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)

    # Save best if needed
    if is_best:
        best_path = checkpoint_dir / "checkpoint_best.pt"
        torch.save(checkpoint, best_path)
        print(f"Saved best checkpoint (loss: {loss:.4f})")

    # Clean up old checkpoints
    cleanup_old_checkpoints(checkpoint_dir, max_checkpoints)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Load a training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load tensors to

    Returns:
        Dictionary with checkpoint metadata
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state if provided
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state if provided
    if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "step": checkpoint.get("step", 0),
        "loss": checkpoint.get("loss", float("inf")),
        "config": checkpoint.get("config"),
        "timestamp": checkpoint.get("timestamp"),
    }


def cleanup_old_checkpoints(checkpoint_dir: Path, max_checkpoints: int) -> None:
    """
    Remove old checkpoints to save disk space.

    Args:
        checkpoint_dir: Directory containing checkpoints
        max_checkpoints: Maximum number of checkpoints to keep
    """
    # Get all step checkpoints
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_step_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )

    # Remove old ones
    while len(checkpoints) > max_checkpoints:
        old_checkpoint = checkpoints.pop(0)
        old_checkpoint.unlink()
        print(f"Removed old checkpoint: {old_checkpoint}")
