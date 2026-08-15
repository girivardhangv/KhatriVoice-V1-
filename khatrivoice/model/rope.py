"""
Rotary Position Embeddings (RoPE) for KhatriVoice.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Precompute the frequency tensor for RoPE.

    Args:
        dim: Dimension of the embeddings
        max_seq_len: Maximum sequence length
        theta: Base frequency
        device: Device to create tensor on

    Returns:
        Complex tensor of shape (max_seq_len, dim // 2)
    """
    # Calculate frequencies
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))

    # Create position indices
    t = torch.arange(max_seq_len, device=device)

    # Outer product
    freqs = torch.outer(t, freqs)

    # Convert to complex form
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        xq: Query tensor of shape (batch, seq_len, num_heads, head_dim)
        xk: Key tensor of shape (batch, seq_len, num_kv_heads, head_dim)
        freqs_cis: Precomputed frequencies of shape (seq_len, head_dim // 2)

    Returns:
        Tuple of rotated query and key tensors
    """
    # Reshape for complex multiplication
    xq_r = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_r = xk.float().reshape(*xk.shape[:-1], -1, 2)

    # Convert to complex
    xq_c = torch.view_as_complex(xq_r)
    xk_c = torch.view_as_complex(xk_r)

    # Get the right slice of freqs_cis
    seq_len = xq.shape[1]
    freqs_cis = freqs_cis[:seq_len]

    # Reshape freqs_cis for broadcasting
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim // 2)

    # Apply rotation
    xq_out = torch.view_as_real(xq_c * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_c * freqs_cis).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding module.

    Precomputes and caches the rotation frequencies for efficient position encoding.

    Args:
        head_dim: Dimension of each attention head
        max_seq_len: Maximum sequence length
        theta: Base frequency for RoPE
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute frequencies (will be moved to correct device in forward)
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(head_dim, max_seq_len, theta),
            persistent=False,
        )

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings.

        Args:
            xq: Query tensor of shape (batch, seq_len, num_heads, head_dim)
            xk: Key tensor of shape (batch, seq_len, num_kv_heads, head_dim)

        Returns:
            Tuple of rotated query and key tensors
        """
        return apply_rotary_emb(xq, xk, self.freqs_cis)
