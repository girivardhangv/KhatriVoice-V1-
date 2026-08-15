"""
Rotary Position Embeddings (RoPE) for KhatriVoice.

RoPE encodes position information by rotating the query and key vectors.
This approach has several advantages:
- Produces explicit relative position dependency
- Provides better length extrapolation than learned position embeddings
- Is computationally efficient

Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
"""

from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tensor:
    """
    Precompute the frequency tensor for RoPE.

    Args:
        dim: Dimension of the query/key (head_dim)
        max_seq_len: Maximum sequence length
        theta: Base frequency (default: 10000)

    Returns:
        Complex tensor of shape [max_seq_len, dim // 2]
    """
    # Compute frequencies for each dimension
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    # Create position indices
    t = torch.arange(max_seq_len)

    # Compute outer product: [max_seq_len, dim // 2]
    freqs = torch.outer(t, freqs)

    # Convert to complex: [max_seq_len, dim // 2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis


def apply_rotary_emb(
    x: Tensor,
    freqs_cis: Tensor,
) -> Tensor:
    """
    Apply rotary embeddings to input tensor.

    Args:
        x: Input tensor [batch, seq_len, n_heads, head_dim]
        freqs_cis: Frequency tensor [seq_len, head_dim // 2]

    Returns:
        Tensor with rotary embeddings applied
    """
    batch_size, seq_len, n_heads, head_dim = x.shape

    # Reshape for complex multiplication
    # [batch, seq_len, n_heads, head_dim] -> [batch, seq_len, n_heads, head_dim // 2]
    x_reshape = x.float().reshape(batch_size, seq_len, n_heads, head_dim // 2, 2)
    x_complex = torch.view_as_complex(x_reshape)

    # Get the right slice of frequencies
    freqs_cis = freqs_cis[:seq_len]

    # Apply rotation
    # [batch, seq_len, n_heads, head_dim // 2] * [seq_len, head_dim // 2]
    x_rotated = x_complex * freqs_cis.unsqueeze(0).unsqueeze(2)

    # Convert back to real
    # [batch, seq_len, n_heads, head_dim // 2] -> [batch, seq_len, n_heads, head_dim]
    x_out = torch.view_as_real(x_rotated).flatten(-2)

    return x_out.type_as(x)


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for KhatriVoice.

    This implementation uses the standard RoPE formulation where
    position information is encoded through rotations of the
    query and key vectors.

    Attributes:
        dim: Dimension of each attention head
        max_seq_len: Maximum sequence length supported
        theta: Base frequency (default: 10000.0)
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
    ) -> None:
        """
        Initialize RoPE.

        Args:
            dim: Dimension of each attention head (head_dim)
            max_seq_len: Maximum sequence length
            theta: Base frequency for position encoding
        """
        super().__init__()

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute frequencies
        freqs_cis = precompute_freqs_cis(dim, max_seq_len, theta)

        # Register as buffer (not a parameter)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        position_ids: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply rotary embeddings to queries and keys.

        Args:
            q: Query tensor [batch, seq_len, n_heads, head_dim]
            k: Key tensor [batch, seq_len, n_heads, head_dim]
            position_ids: Optional position IDs [batch, seq_len]

        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        batch_size, seq_len, n_heads, head_dim = q.shape

        # Get position-specific frequencies if position_ids provided
        if position_ids is not None:
            # Gather frequencies based on position IDs
            freqs = self.freqs_cis[position_ids]  # [batch, seq_len, dim // 2] complex
        else:
            # Use default frequencies
            freqs = self.freqs_cis[:seq_len]  # [seq_len, dim // 2] complex

        # Apply rotary embeddings
        q_rotated = self._apply_rotary(q, freqs)
        k_rotated = self._apply_rotary(k, freqs)

        return q_rotated, k_rotated

    def _apply_rotary(
        self,
        x: Tensor,
        freqs: Tensor,
    ) -> Tensor:
        """
        Apply rotary embeddings to a tensor.

        Args:
            x: Input tensor [batch, seq_len, n_heads, head_dim]
            freqs: Frequency tensor

        Returns:
            Rotated tensor
        """
        batch_size, seq_len, n_heads, head_dim = x.shape

        # Ensure head_dim is even
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"

        # Split into real and imaginary parts
        x_reshape = x.reshape(batch_size, seq_len, n_heads, head_dim // 2, 2)
        x_complex = torch.view_as_complex(x_reshape)

        # Broadcast frequencies for batch and heads
        if freqs.dim() == 3:
            # [batch, seq_len, dim // 2]
            freqs = freqs.unsqueeze(2)  # [batch, seq_len, 1, dim // 2]
        else:
            # [seq_len, dim // 2]
            freqs = freqs.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim // 2]

        # Apply rotation
        x_rotated = x_complex * freqs

        # Convert back to real
        x_out = torch.view_as_real(x_rotated).flatten(-2)

        return x_out.type_as(x)

    def extend_position_ids(
        self,
        position_ids: Tensor,
    ) -> Tensor:
        """
        Extend position IDs for longer sequences.

        Args:
            position_ids: Original position IDs

        Returns:
            Extended position IDs
        """
        # Get current max position
        max_pos = position_ids.max().item()

        # Extend if needed
        if max_pos >= self.max_seq_len:
            new_max_seq_len = max_pos + 1

            # Recompute frequencies
            freqs_cis = precompute_freqs_cis(self.dim, new_max_seq_len, self.theta)

            # Update buffer
            self.register_buffer("freqs_cis", freqs_cis, persistent=False)
            self.max_seq_len = new_max_seq_len

        return position_ids

    def __repr__(self) -> str:
        return (
            f"RotaryPositionEmbedding("
            f"dim={self.dim}, "
            f"max_seq_len={self.max_seq_len}, "
            f"theta={self.theta})"
        )


def compute_rope_freqs(
    head_dim: int,
    seq_len: int,
    theta: float = 10000.0,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """
    Compute RoPE frequencies for a given sequence length.

    This is a convenience function for computing frequencies on-the-fly.

    Args:
        head_dim: Dimension of each attention head
        seq_len: Sequence length
        theta: Base frequency
        device: Device to create tensors on

    Returns:
        Frequency tensor [seq_len, head_dim // 2] as complex
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis
