"""
Training metrics tracking for KhatriVoice.
"""

from typing import Dict, List
from collections import deque


class MetricsTracker:
    """
    Track training metrics.

    Args:
        window_size: Size of moving average window
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.losses: deque = deque(maxlen=window_size)
        self.grad_norms: deque = deque(maxlen=window_size)
        self.learning_rates: deque = deque(maxlen=window_size)

    def add_loss(self, loss: float) -> None:
        """Add a loss value."""
        self.losses.append(loss)

    def add_grad_norm(self, grad_norm: float) -> None:
        """Add a gradient norm value."""
        self.grad_norms.append(grad_norm)

    def add_learning_rate(self, lr: float) -> None:
        """Add a learning rate value."""
        self.learning_rates.append(lr)

    def get_avg_loss(self) -> float:
        """Get average loss over window."""
        if not self.losses:
            return 0.0
        return sum(self.losses) / len(self.losses)

    def get_avg_grad_norm(self) -> float:
        """Get average gradient norm over window."""
        if not self.grad_norms:
            return 0.0
        return sum(self.grad_norms) / len(self.grad_norms)

    def get_current_lr(self) -> float:
        """Get current learning rate."""
        if not self.learning_rates:
            return 0.0
        return self.learning_rates[-1]

    def get_summary(self) -> Dict[str, float]:
        """Get summary of all metrics."""
        return {
            "avg_loss": self.get_avg_loss(),
            "avg_grad_norm": self.get_avg_grad_norm(),
            "current_lr": self.get_current_lr(),
            "last_loss": self.losses[-1] if self.losses else 0.0,
        }

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self.losses.clear()
        self.grad_norms.clear()
        self.learning_rates.clear()
