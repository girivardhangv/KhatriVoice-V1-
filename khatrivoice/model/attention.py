"""
Attention mechanisms for KhatriVoice.

This module implements:
- Causal self-attention
- Grouped Query Attention (GQA)
- Rotary Position Embeddings integration
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from khatrivoice.model.rope import RotaryPositionEmbedding


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with optional Grouped Query Attention (GQA).

    This implements the core attention mechanism for KhatriVoice:
    - Uses RoPE for position encoding
    - Supports Grouped Query Attention for efficiency
    - Includes causal masking for autoregressive generation

    Attributes:
        hidden_size: Hidden dimension
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads (for GQA)
        head_dim: Dimension per head
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        bias: bool = False,
        rope_theta: float = 10000.0,
    ) -> None:
        """
        Initialize attention.

        Args:
            hidden_size: Hidden dimension
            num_heads: Number of query heads
            num_kv_heads: Number of key/value heads (<= num_heads)
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            bias: Whether to use bias in projections
            rope_theta: Base frequency for RoPE
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0, \
            "hidden_size must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, \
            "num_heads must be divisible by num_kv_heads (for GQA)"

        self.num_groups = num_heads // num_kv_heads
        self.kv_dim = num_kv_heads * self.head_dim

        # Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.kv_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.kv_dim, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        # RoPE
        self.rope = RotaryPositionEmbedding(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            theta=rope_theta,
        )

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).bool(),
            persistent=False,
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize projection weights."""
        # Use smaller init for stability
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)

        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.o_proj.bias)

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
            past_key_value: Optional cached (key, value) for generation
            use_cache: Whether to return cache for generation

        Returns:
            Tuple of (output, cache) where cache is optional
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        # Q: [batch, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # K: [batch, seq_len, num_kv_heads, head_dim]
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        # V: [batch, seq_len, num_kv_heads, head_dim]
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply RoPE to Q and K
        q, k = self.rope(q, k, position_ids)

        # Handle KV cache for generation
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        # Update cache
        past_key_value = (k, v) if use_cache else None

        # Get current sequence length (including cached tokens)
        kv_seq_len = k.shape[1]

        # For GQA, expand K and V to match Q's number of heads
        # Repeat each KV head to match num_heads / num_kv_heads
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=2)
            v = v.repeat_interleave(self.num_groups, dim=2)

        # Transpose for attention: [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        # [batch, num_heads, seq_len, kv_seq_len]
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask
        # Ensure mask is on same device as attention weights
        causal_mask = self.causal_mask[:seq_len, :kv_seq_len]
        causal_mask = causal_mask.to(attn_weights.device)
        attn_weights = attn_weights.masked_fill(
            ~causal_mask.bool(),
            float("-inf"),
        )

        # Apply additional attention mask if provided
        # attention_mask is already shaped [batch, 1, 1, seq_len] from the model
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(
                ~attention_mask.bool(),
                float("-inf"),
            )

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        # Apply dropout
        attn_weights = self.attn_dropout(attn_weights)

        # Compute output
        # [batch, num_heads, seq_len, head_dim]
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back: [batch, seq_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        # Output projection
        output = self.o_proj(attn_output)

        # Apply residual dropout
        output = self.resid_dropout(output)

        return output, past_key_value

    def __repr__(self) -> str:
        return (
            f"CausalSelfAttention("
            f"hidden_size={self.hidden_size}, "
            f"num_heads={self.num_heads}, "
            f"num_kv_heads={self.num_kv_heads})"
        )


class Attention(nn.Module):
    """
    Attention wrapper for compatibility.

    This is a thin wrapper around CausalSelfAttention for cleaner interfaces.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
    ) -> None:
        """Initialize attention."""
        super().__init__()
        self.self_attn = CausalSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            rope_theta=rope_theta,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """Forward pass."""
        return self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
