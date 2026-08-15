"""
Seed utilities for KhatriVoice.

Provides deterministic behavior for reproducibility across
random number generation, PyTorch, and NumPy.
"""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    This function sets the seed for Python's random module, NumPy,
    and PyTorch (both CPU and CUDA if available).

    Args:
        seed: The random seed to use

    Note:
        For full reproducibility in PyTorch CUDA, additional settings
        may be needed (torch.backends.cudnn.deterministic, etc.)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set seed for all CUDA devices
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_deterministic_mode(enabled: bool = True) -> None:
    """
    Enable or disable deterministic mode for PyTorch.

    This makes PyTorch operations deterministic at the potential cost
    of performance. Useful for reproduction but should be disabled
    for training speed.

    Args:
        enabled: Whether to enable deterministic mode

    Note:
        CUDA deterministic operations may be slower.
        Some operations may throw errors in deterministic mode.
    """
    if enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_random_generator(
    seed: Optional[int] = None,
    device: str | torch.device = "cpu",
) -> torch.Generator:
    """
    Create a PyTorch random generator with a specified seed.

    Args:
        seed: The random seed to use (None for random seed)
        device: Device for the generator

    Returns:
        torch.Generator instance
    """
    if seed is None:
        seed = torch.initial_seed()

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator
