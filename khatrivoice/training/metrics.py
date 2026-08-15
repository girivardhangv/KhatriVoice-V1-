"""
Training metrics for KhatriVoice.

This module provides metrics tracking for training and evaluation.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F


@dataclass
class TrainingMetrics:
    """
    Container for training metrics.

    Tracks loss, perplexity, and various statistics during training.

    Attributes:
        step: Current training step
        epoch: Current epoch
        loss: Running loss value
        total_tokens: Total tokens processed
        num_batches: Number of batches processed
        losses: List of recent losses (for sliding window)
        learning_rate: Current learning rate
        grad_norm: Last gradient norm
    """

    step: int = 0
    epoch: int = 0
    loss: float = 0.0
    total_tokens: int = 0
    num_batches: int = 0
    losses: List[float] = field(default_factory=list)
    learning_rate: float = 0.0
    grad_norm: float = 0.0

    # For smoothing window
    smoothing_window: int = 100

    def update(
        self,
        loss: float,
        batch_size: int,
        seq_len: int,
        learning_rate: float = 0.0,
        grad_norm: float = 0.0,
    ) -> None:
        """
        Update metrics with new batch.

        Args:
            loss: Batch loss
            batch_size: Batch size
            seq_len: Sequence length
            learning_rate: Current learning rate
            grad_norm: Gradient norm
        """
        self.loss = loss
        self.step += 1
        self.num_batches += 1
        self.total_tokens += batch_size * seq_len
        self.learning_rate = learning_rate
        self.grad_norm = grad_norm

        # Track recent losses
        self.losses.append(loss)
        if len(self.losses) > self.smoothing_window:
            self.losses.pop(0)

    def reset_epoch(self) -> None:
        """Reset epoch-level metrics."""
        self.num_batches = 0
        self.loss = 0.0

    def get_average_loss(self) -> float:
        """Get smoothed average loss."""
        if not self.losses:
            return 0.0
        return sum(self.losses) / len(self.losses)

    def get_perplexity(self) -> float:
        """Get perplexity from average loss."""
        avg_loss = self.get_average_loss()
        return float(torch.exp(torch.tensor(avg_loss)))

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "step": self.step,
            "epoch": self.epoch,
            "loss": self.loss,
            "avg_loss": self.get_average_loss(),
            "perplexity": self.get_perplexity(),
            "total_tokens": self.total_tokens,
            "learning_rate": self.learning_rate,
            "grad_norm": self.grad_norm,
        }


class MetricsTracker:
    """
    Tracks and aggregates training/validation metrics.

    Maintains separate metrics for training and validation,
    and provides methods for logging and reporting.
    """

    def __init__(self, smoothing_window: int = 100) -> None:
        """
        Initialize the metrics tracker.

        Args:
            smoothing_window: Window size for smoothing loss
        """
        self.train_metrics = TrainingMetrics(smoothing_window=smoothing_window)
        self.val_metrics = TrainingMetrics(smoothing_window=smoothing_window)
        self._history: List[Dict] = []

    def update_train(
        self,
        loss: float,
        batch_size: int,
        seq_len: int,
        learning_rate: float = 0.0,
        grad_norm: float = 0.0,
    ) -> None:
        """Update training metrics."""
        self.train_metrics.update(
            loss=loss,
            batch_size=batch_size,
            seq_len=seq_len,
            learning_rate=learning_rate,
            grad_norm=grad_norm,
        )

    def update_val(
        self,
        loss: float,
        batch_size: int,
        seq_len: int,
    ) -> None:
        """Update validation metrics."""
        self.val_metrics.update(
            loss=loss,
            batch_size=batch_size,
            seq_len=seq_len,
        )

    def reset_epoch(self) -> None:
        """Reset epoch-level metrics."""
        self.train_metrics.reset_epoch()
        self.val_metrics.reset_epoch()

    def log_step(self) -> Dict:
        """Get current step metrics dict."""
        data = {
            "train": self.train_metrics.to_dict(),
            "val": self.val_metrics.to_dict(),
        }
        self._history.append(data)
        return data

    def get_best_val_loss(self) -> float:
        """Get best validation loss from history."""
        if not self._history:
            return float("inf")
        return min(h["val"]["loss"] for h in self._history)

    def get_train_perplexity(self) -> float:
        """Get current training perplexity."""
        return self.train_metrics.get_perplexity()

    def get_val_perplexity(self) -> float:
        """Get current validation perplexity."""
        return self.val_metrics.get_perplexity()

    def get_summary(self) -> str:
        """Get summary string."""
        train = self.train_metrics.to_dict()
        val = self.val_metrics.to_dict()

        return (
            f"Step {train['step']} | "
            f"Train Loss: {train['avg_loss']:.4f} | "
            f"Train PPL: {train['perplexity']:.2f} | "
            f"LR: {train['learning_rate']:.2e} | "
            f"Grad Norm: {train['grad_norm']:.2f}"
        )


@torch.no_grad()
def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute cross-entropy loss.

    Args:
        logits: Model logits [batch, seq_len, vocab_size]
        labels: Target labels [batch, seq_len]
        ignore_index: Index to ignore in loss

    Returns:
        Loss tensor (scalar)
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Shift for causal LM
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)

    # Compute loss
    loss = F.cross_entropy(
        shift_logits,
        shift_labels,
        ignore_index=ignore_index,
        reduction="mean",
    )

    return loss


@torch.no_grad()
def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from loss.

    Args:
        loss: Cross-entropy loss value

    Returns:
        Perplexity value
    """
    return float(torch.exp(torch.tensor(loss)))


@torch.no_grad()
def compute_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """
    Compute token-level accuracy.

    Args:
        logits: Model logits [batch, seq_len, vocab_size]
        labels: Target labels [batch, seq_len]
        ignore_index: Index to ignore

    Returns:
        Accuracy value between 0 and 1
    """
    # Get predictions
    predictions = logits.argmax(dim=-1)

    # Shift for causal LM
    predictions = predictions[..., :-1]
    labels = labels[..., 1:]

    # Create mask for valid tokens
    mask = labels != ignore_index

    # Compute accuracy
    correct = (predictions == labels) & mask
    accuracy = correct.sum().float() / mask.sum().float()

    return accuracy.item()
