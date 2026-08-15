"""
Normalization layers for KhatriVoice.

This module implements RMSNorm, the normalization layer used in
many modern language models like LLaMA.
"""

import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    RMSNorm is a simplified version of LayerNorm that:
    - Does not compute the mean
    - Normalizes using only the root mean square
    - Is more computationally efficient
    - Works well for language models

    Reference: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)

    Attributes:
        hidden_size: Size of the input
        eps: Small constant for numerical stability
        weight: Learnable scale parameter
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        """
        Initialize RMSNorm.

        Args:
            hidden_size: Size of the input tensor
            eps: Small constant to avoid division by zero
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.eps = eps

        # Learnable scale parameter (gamma)
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [..., hidden_size]

        Returns:
            Normalized tensor with same shape
        """
        return self._norm(x)

    def _norm(self, x: Tensor) -> Tensor:
        """
        Apply RMS normalization.

        Args:
            x: Input tensor

        Returns:
            Normalized tensor
        """
        # Compute variance (mean of squares)
        variance = x.pow(2).mean(dim=-1, keepdim=True)

        # Normalize
        x_normed = x * torch.rsqrt(variance + self.eps)

        # Scale
        return self.weight * x_normed

    def __repr__(self) -> str:
        return f"RMSNorm(hidden_size={self.hidden_size}, eps={self.eps})"


class LayerNorm(nn.Module):
    """
    Standard Layer Normalization.

    Included for comparison with RMSNorm. KhatriVoice primarily uses
    RMSNorm as it's more efficient and commonly used in modern LLMs.

    Attributes:
        hidden_size: Size of the input
        eps: Small constant for numerical stability
        weight: Learnable scale parameter (gamma)
        bias: Learnable shift parameter (beta)
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
    ) -> None:
        """
        Initialize LayerNorm.

        Args:
            hidden_size: Size of the input tensor
            eps: Small constant to avoid division by zero
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [..., hidden_size]

        Returns:
            Normalized tensor with same shape
        """
        # Compute mean and variance
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize
        x_normed = (x - mean) / torch.sqrt(variance + self.eps)

        # Scale and shift
        return self.weight * x_normed + self.bias

    def __repr__(self) -> str:
        return f"LayerNorm(hidden_size={self.hidden_size}, eps={self.eps})"


class GroupedRMSNorm(nn.Module):
    """
    Grouped RMSNorm for grouped query attention.

    Normalizes groups of heads independently. This is useful
    when using GQA where multiple query heads share the same key/value.

    Attributes:
        num_groups: Number of groups
        hidden_size: Total hidden size
        eps: Small constant for numerical stability
    """

    def __init__(
        self,
        hidden_size: int,
        num_groups: int,
        eps: float = 1e-6,
    ) -> None:
        """
        Initialize GroupedRMSNorm.

        Args:
            hidden_size: Total hidden size
            num_groups: Number of groups
            eps: Small constant for numerical stability
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.eps = eps

        assert hidden_size % num_groups == 0, \
            "hidden_size must be divisible by num_groups"

        self.group_size = hidden_size // num_groups
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [..., hidden_size]

        Returns:
            Normalized tensor
        """
        # Reshape into groups
        x_reshaped = x.view(*x.shape[:-1], self.num_groups, self.group_size)

        # Compute variance per group
        variance = x_reshaped.pow(2).mean(dim=-1, keepdim=True)

        # Normalize
        x_normed = x_reshaped * torch.rsqrt(variance + self.eps)

        # Reshape back
        x_normed = x_normed.view(*x.shape)

        # Scale
        return self.weight * x_normed

    def __repr__(self) -> str:
        return (
            f"GroupedRMSNorm("
            f"hidden_size={self.hidden_size}, "
            f"num_groups={self.num_groups})"
        )
