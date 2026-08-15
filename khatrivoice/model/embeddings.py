"""
Embedding layers and normalization for KhatriVoice.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TokenEmbedding(nn.Module):
    """
    Token embedding layer with optional position embeddings.

    Args:
        vocab_size: Size of the vocabulary
        hidden_size: Dimension of the embeddings
        max_position: Maximum sequence length for positional embeddings
        dropout: Dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for token embedding.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)

        Returns:
            Embedded tokens of shape (batch_size, seq_len, hidden_size)
        """
        # Get token embeddings
        embeddings = self.token_embedding(input_ids)

        # Apply dropout
        embeddings = self.dropout(embeddings)

        return embeddings


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    A simpler alternative to LayerNorm that normalizes using RMS without
    centering and scaling bias.

    Args:
        hidden_size: Dimension of the input
        eps: Small constant for numerical stability
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMS normalization.

        Args:
            x: Input tensor of shape (..., hidden_size)

        Returns:
            Normalized tensor of same shape
        """
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # Normalize and scale
        x_normed = x / rms
        return self.weight * x_normed


class LayerNorm(nn.Module):
    """
    Standard Layer Normalization.

    Args:
        hidden_size: Dimension of the input
        eps: Small constant for numerical stability
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for layer normalization.

        Args:
            x: Input tensor of shape (..., hidden_size)

        Returns:
            Normalized tensor of same shape
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / (std + self.eps) + self.bias
