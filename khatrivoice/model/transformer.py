"""
Transformer backbone for KhatriVoice.

This module implements the full transformer stack of blocks.
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
from torch import Tensor

from khatrivoice.model.block import TransformerBlock
from khatrivoice.model.normalization import RMSNorm


class Transformer(nn.Module):
    """
    Transformer stack for KhatriVoice.

    This is the main transformer backbone consisting of:
    - Stack of transformer blocks
    - Final layer normalization

    The embedding layer and output head are separate modules.

    Attributes:
        hidden_size: Hidden dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        num_kv_heads: Number of key/value heads
        intermediate_size: MLP intermediate dimension
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-6,
    ) -> None:
        """
        Initialize transformer.

        Args:
            hidden_size: Hidden dimension
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads
            num_kv_heads: Number of key/value heads
            intermediate_size: MLP intermediate dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            rope_theta: Base frequency for RoPE
            norm_eps: Epsilon for RMSNorm
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len

        # Stack of transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                intermediate_size=intermediate_size,
                max_seq_len=max_seq_len,
                dropout=dropout,
                rope_theta=rope_theta,
                norm_eps=norm_eps,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.norm = RMSNorm(hidden_size, eps=norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[List[Tuple[Tensor, Tensor]]]]:
        """
        Forward pass.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask [batch, seq_len]
            position_ids: Optional position IDs [batch, seq_len]
            past_key_values: Optional list of cached KV tuples
            use_cache: Whether to return caches

        Returns:
            Tuple of (hidden_states, past_key_values)
        """
        # Initialize past_key_values if needed
        if past_key_values is None:
            past_key_values = [None] * self.num_layers

        # Track new caches
        new_past_key_values: List[Tuple[Tensor, Tensor]] = []

        # Apply transformer blocks
        for i, layer in enumerate(self.layers):
            hidden_states, past_kv = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[i],
                use_cache=use_cache,
            )

            if use_cache:
                new_past_key_values.append(past_kv)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        return hidden_states, new_past_key_values if use_cache else None

    def __repr__(self) -> str:
        return (
            f"Transformer("
            f"hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, "
            f"num_heads={self.num_heads})"
        )
