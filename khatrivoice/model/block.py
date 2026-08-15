"""
Transformer block for KhatriVoice.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from khatrivoice.model.attention import Attention
from khatrivoice.model.mlp import MLP
from khatrivoice.model.embeddings import RMSNorm


class TransformerBlock(nn.Module):
    """
    Single transformer block with attention and MLP.

    Uses pre-norm architecture with RMSNorm following LLaMA.

    Args:
        hidden_size: Dimension of hidden representations
        num_heads: Number of attention heads
        num_kv_heads: Number of key-value heads for GQA
        intermediate_size: Dimension of MLP intermediate layer
        max_seq_len: Maximum sequence length
        rope_theta: Base frequency for RoPE
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        max_seq_len: int = 512,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Attention
        self.self_attn = Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            dropout=dropout,
        )

        # MLP
        self.mlp = MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout,
        )

        # Layer norms (pre-norm architecture)
        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for transformer block.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            past_key_value: Optional cached key-value states
            use_cache: Whether to return cached key-value states

        Returns:
            Tuple of (output tensor, optional cached key-value)
        """
        residual = hidden_states

        # Pre-norm for attention
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention
        hidden_states, _, present_key_value = self.self_attn(
            hidden_states=hidden_states,
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

        return hidden_states, present_key_value
