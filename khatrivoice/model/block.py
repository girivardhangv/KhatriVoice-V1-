"""
Transformer block for KhatriVoice.

This module implements a single transformer block with:
- Pre-normalization (RMSNorm)
- Causal self-attention
- Residual connections
- SwiGLU MLP
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor

from khatrivoice.model.attention import CausalSelfAttention
from khatrivoice.model.mlp import SwiGLU
from khatrivoice.model.normalization import RMSNorm


class TransformerBlock(nn.Module):
    """
    Single transformer block for KhatriVoice.

    This implements a decoder-only transformer block with:
    - RMSNorm for pre-normalization
    - Causal self-attention with RoPE
    - Residual connections
    - SwiGLU feed-forward network

    Architecture:
        x = x + attention(norm(x))
        x = x + ffn(norm(x))

    The block uses "pre-norm" architecture where normalization
    is applied before each sub-layer.

    Attributes:
        hidden_size: Hidden dimension
        num_heads: Number of attention heads
        num_kv_heads: Number of key/value heads
        intermediate_size: MLP intermediate dimension
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-6,
    ) -> None:
        """
        Initialize transformer block.

        Args:
            hidden_size: Hidden dimension
            num_heads: Number of attention heads
            num_kv_heads: Number of key/value heads (for GQA)
            intermediate_size: MLP intermediate dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            rope_theta: Base frequency for RoPE
            norm_eps: Epsilon for RMSNorm
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.intermediate_size = intermediate_size

        # Pre-normalization layers
        self.input_layernorm = RMSNorm(hidden_size, eps=norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=norm_eps)

        # Self-attention
        self.self_attn = CausalSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            rope_theta=rope_theta,
        )

        # MLP (SwiGLU)
        self.mlp = SwiGLU(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Forward pass.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask [batch, seq_len]
            position_ids: Optional position IDs [batch, seq_len]
            past_key_value: Optional cached KV for generation
            use_cache: Whether to return cache

        Returns:
            Tuple of (hidden_states, cache)
        """
        residual = hidden_states

        # Pre-norm for attention
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention
        hidden_states, past_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        # Residual connection
        hidden_states = residual + hidden_states

        # Pre-norm for MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP
        hidden_states = self.mlp(hidden_states)

        # Residual connection
        hidden_states = residual + hidden_states

        return hidden_states, past_key_value

    def __repr__(self) -> str:
        return (
            f"TransformerBlock("
            f"hidden_size={self.hidden_size}, "
            f"num_heads={self.num_heads}, "
            f"intermediate_size={self.intermediate_size})"
        )
