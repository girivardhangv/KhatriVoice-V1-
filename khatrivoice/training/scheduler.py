"""
Learning rate schedulers for KhatriVoice training.

This module provides custom learning rate schedulers.
"""

from typing import Optional
import math
from torch.optim.lr_scheduler import _LRScheduler


class WarmupScheduler(_LRScheduler):
    """
    Warmup learning rate scheduler.

    Linearly increases learning rate during warmup, then maintains
    or decays based on the decay type.

    Attributes:
        optimizer: Optimizer to schedule
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr: Minimum learning rate
        decay_type: Type of decay after warmup ('linear', 'cosine', 'constant')
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        decay_type: str = "cosine",
        last_epoch: int = -1,
    ) -> None:
        """
        Initialize the scheduler.

        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            min_lr: Minimum learning rate
            decay_type: Type of decay after warmup
            last_epoch: Last epoch index
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.decay_type = decay_type

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list:
        """Get learning rates for current step."""
        step = self._step_count

        if step < self.warmup_steps:
            # Linear warmup
            warmup_factor = step / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        # After warmup
        if self.decay_type == "constant":
            return self.base_lrs

        # Calculate decay
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )

        if self.decay_type == "linear":
            decay_factor = max(self.min_lr / self.base_lrs[0],
                              1.0 - progress)
        elif self.decay_type == "cosine":
            decay_factor = max(
                self.min_lr / self.base_lrs[0],
                0.5 * (1.0 + math.cos(math.pi * progress))
            )
        else:
            decay_factor = 1.0

        return [base_lr * decay_factor for base_lr in self.base_lrs]


class CosineAnnealingWarmupScheduler(_LRScheduler):
    """
    Cosine annealing with linear warmup.

    This is the standard learning rate schedule for language models:
    1. Linear warmup phase
    2. Cosine annealing decay
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Initialize the scheduler.

        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            min_lr: Minimum learning rate
            last_epoch: Last epoch index
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list:
        """Get learning rates for current step."""
        step = self._step_count

        if step < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * step / max(1, self.warmup_steps)
                for base_lr in self.base_lrs
            ]

        # Cosine annealing
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )

        return [
            self.min_lr + 0.5 * (base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
            for base_lr in self.base_lrs
        ]
