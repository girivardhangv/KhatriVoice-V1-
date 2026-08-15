"""
Learning rate scheduler utilities for KhatriVoice.
"""

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional


def create_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr: float = 0.0,
) -> LambdaLR:
    """
    Create a learning rate scheduler with linear warmup and decay.

    Args:
        optimizer: Optimizer to schedule
        warmup_steps: Number of warmup steps
        max_steps: Total training steps
        min_lr: Minimum learning rate

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        # Warmup
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Decay
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(min_lr, float(1.0 - progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler


def create_cosine_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr: float = 0.0,
) -> LambdaLR:
    """
    Create a cosine annealing learning rate scheduler with warmup.

    Args:
        optimizer: Optimizer to schedule
        warmup_steps: Number of warmup steps
        max_steps: Total training steps
        min_lr: Minimum learning rate ratio

    Returns:
        LambdaLR scheduler
    """
    import math

    def lr_lambda(current_step: int) -> float:
        # Warmup
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler
