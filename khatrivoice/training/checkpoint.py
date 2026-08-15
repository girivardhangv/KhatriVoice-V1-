"""
Checkpoint management for KhatriVoice training.

This module handles saving and loading model checkpoints.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
from datetime import datetime


class CheckpointManager:
    """
    Manages model checkpoints during training.

    Handles:
    - Saving checkpoints at regular intervals
    - Loading checkpoints to resume training
    - Keeping track of best checkpoints
    - Cleaning up old checkpoints

    Attributes:
        checkpoint_dir: Directory to save checkpoints
        max_checkpoints: Maximum number of checkpoints to keep
        save_best: Whether to save the best checkpoint separately
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        max_checkpoints: int = 5,
        save_best: bool = True,
    ) -> None:
        """
        Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of recent checkpoints to keep
            save_best: Whether to save the best checkpoint separately
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best

        # Create directory if needed
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Track best loss
        self.best_loss: Optional[float] = None

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        step: int,
        epoch: int,
        loss: float,
        config: Any,
        extra_state: Optional[Dict] = None,
        tokenizer_path: Optional[str] = None,
    ) -> Path:
        """
        Save a checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            step: Current training step
            epoch: Current epoch
            loss: Current loss value
            config: Model configuration
            extra_state: Additional state to save
            tokenizer_path: Path to saved tokenizer

        Returns:
            Path to saved checkpoint
        """
        # Build checkpoint
        checkpoint = {
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "config": config.to_dict() if hasattr(config, "to_dict") else config,
            "timestamp": datetime.now().isoformat(),
        }

        # Include tokenizer path for inference
        if tokenizer_path:
            checkpoint["tokenizer_path"] = tokenizer_path
        elif hasattr(self, "tokenizer_path"):
            checkpoint["tokenizer_path"] = self.tokenizer_path

        if extra_state:
            checkpoint["extra_state"] = extra_state

        # Save checkpoint with step number
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)

        # Save best checkpoint if needed
        if self.save_best:
            if self.best_loss is None or loss < self.best_loss:
                self.best_loss = loss
                best_path = self.checkpoint_dir / "checkpoint_best.pt"
                torch.save(checkpoint, best_path)

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        return checkpoint_path

    def load(
        self,
        checkpoint_path: Optional[str | Path] = None,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint (uses latest if None)
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            device: Device to load to

        Returns:
            Checkpoint dictionary
        """
        # Determine checkpoint path
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "checkpoint_latest.pt"
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model state
        if model is not None:
            model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            if checkpoint["scheduler_state_dict"] is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_checkpoints."""
        # Get all step checkpoints
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )

        # Remove old checkpoints
        while len(checkpoints) > self.max_checkpoints:
            oldest = checkpoints.pop(0)
            oldest.unlink()

    def list_checkpoints(self) -> List[Path]:
        """List all available checkpoints."""
        return sorted(self.checkpoint_dir.glob("checkpoint_step_*.pt"))

    def has_checkpoint(self) -> bool:
        """Check if any checkpoint exists."""
        return (self.checkpoint_dir / "checkpoint_latest.pt").exists()

    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint if it exists."""
        best_path = self.checkpoint_dir / "checkpoint_best.pt"
        return best_path if best_path.exists() else None

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint if it exists."""
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        return latest_path if latest_path.exists() else None


def save_checkpoint(
    filepath: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    config: Optional[Any] = None,
    step: int = 0,
    loss: float = 0.0,
) -> None:
    """
    Simple utility to save a checkpoint.

    Args:
        filepath: Path to save checkpoint
        model: Model to save
        optimizer: Optional optimizer state
        config: Optional configuration
        step: Training step
        loss: Loss value
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if config is not None:
        checkpoint["config"] = config.to_dict() if hasattr(config, "to_dict") else config

    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str | Path,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Simple utility to load a checkpoint.

    Args:
        filepath: Path to checkpoint
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        device: Device to load to

    Returns:
        Checkpoint dictionary
    """
    filepath = Path(filepath)
    checkpoint = torch.load(filepath, map_location=device)

    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
