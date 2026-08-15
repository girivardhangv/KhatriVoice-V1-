"""
Optimizer utilities for KhatriVoice training.

This module provides optimizer creation and configuration.
"""

from typing import Optional, List
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
) -> AdamW:
    """
    Create AdamW optimizer for model training.

    Args:
        model: KhatriVoice model
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        betas: Adam betas
        eps: Epsilon for numerical stability

    Returns:
        AdamW optimizer
    """
    # Separate parameters that should and shouldn't have weight decay
    no_decay = ["bias", "layernorm", "norm"]

    optimizer_grouped_parameters = []
    params_no_decay = []
    params_with_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if parameter should have no decay
        should_no_decay = any(nd in name.lower() for nd in no_decay)

        if should_no_decay:
            params_no_decay.append(param)
        else:
            params_with_decay.append(param)

    # Create parameter groups
    optimizer_grouped_parameters = [
        {
            "params": params_with_decay,
            "weight_decay": weight_decay,
        },
        {
            "params": params_no_decay,
            "weight_decay": 0.0,
        },
    ]

    # Handle empty groups
    if not params_with_decay and not params_no_decay:
        optimizer_grouped_parameters = [
            {
                "params": model.parameters(),
                "weight_decay": weight_decay,
            }
        ]
    elif not params_with_decay:
        optimizer_grouped_parameters = [
            {
                "params": params_no_decay,
                "weight_decay": 0.0,
            }
        ]
    elif not params_no_decay:
        optimizer_grouped_parameters = [
            {
                "params": params_with_decay,
                "weight_decay": weight_decay,
            }
        ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=betas,
        eps=eps,
    )

    return optimizer


def create_linear_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """
    Create a linear warmup + linear decay scheduler.

    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate ratio at end of training

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        # Linear warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Linear decay
        progress = (current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )

        # From 1.0 to min_lr_ratio
        return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))

    return LambdaLR(optimizer, lr_lambda)


def create_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """
    Create a linear warmup + cosine decay scheduler.

    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate ratio at end of training

    Returns:
        LambdaLR scheduler
    """
    import math

    def lr_lambda(current_step: int) -> float:
        # Linear warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay
        progress = (current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )

        # Cosine from 1.0 to min_lr_ratio
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)
