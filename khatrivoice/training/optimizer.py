"""
Optimizer utilities for KhatriVoice.
"""

import torch
from torch.optim import AdamW
from typing import Optional


def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.95,
    max_grad_norm: float = 1.0,
) -> AdamW:
    """
    Create an AdamW optimizer with weight decay.

    Args:
        model: Model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        beta1: Adam beta1
        beta2: Adam beta2
        max_grad_norm: Maximum gradient norm for clipping

    Returns:
        AdamW optimizer
    """
    # Separate parameters that should and shouldn't have weight decay
    no_decay = ["bias", "layer_norm.weight", "layernorm.weight", "norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(beta1, beta2),
    )

    return optimizer


def clip_gradients(
    model: torch.nn.Module,
    max_grad_norm: float,
) -> float:
    """
    Clip gradients by global norm.

    Args:
        model: Model with gradients
        max_grad_norm: Maximum gradient norm

    Returns:
        Total gradient norm before clipping
    """
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_grad_norm,
    )
