"""
Attention mechanisms for KhatriVoice transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from khatrivoice.model.rope import RotaryPositionEmbedding


class Attention(nn.Module):
    """
    Multi-head attention with Grouped Query Attention (GQA) support.

    GQA uses fewer key-value heads than query heads to reduce memory
    bandwidth requirements while maintaining quality.

    Args:
        hidden_size: Dimension of the hidden representations
        num_heads: Number of query attention heads
        num_kv_heads: Number of key-value attention heads
        max_seq_len: Maximum sequence length
        rope_theta: Base frequency for RoPE
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int = 512,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.num_groups = num_heads // num_kv_heads

        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        # RoPE
        self.rope = RotaryPositionEmbedding(
            head_dim=self.head_dim,
            max_seq_len=max_seq_len,
            theta=rope_theta,
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Scaling
        self.scale = 1.0 / (self.head_dim ** 0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for multi-head attention.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Optional mask of shape (batch, 1, seq_len, seq_len)
            position_ids: Optional position IDs (unused, RoPE handles positions)
            past_key_value: Optional cached key-value states
            use_cache: Whether to return cached key-value states

        Returns:
            Tuple of (output tensor, None, optional cached key-value)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to query, key, value
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply RoPE
        q, k = self.rope(q, k)

        # Handle cached key-value
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        # Cache key-value if needed
        present_key_value = (k, v) if use_cache else None

        # Get the actual sequence length after concatenation with cache
        kv_seq_len = k.shape[1]

        # Expand k and v for GQA (repeat each kv head for num_groups query heads)
        if self.num_groups > 1:
            k = k.unsqueeze(3).expand(-1, -1, -1, self.num_groups, -1)
            k = k.reshape(batch_size, kv_seq_len, self.num_heads, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, self.num_groups, -1)
            v = v.reshape(batch_size, kv_seq_len, self.num_heads, self.head_dim)

        # Transpose for attention: (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Create causal mask
        query_len = q.shape[2]

        # Apply causal mask
        if query_len > 1 or kv_seq_len > 1:
            # Create causal mask where each query position can only attend to keys at or before it
            causal_mask = torch.triu(
                torch.ones(query_len, kv_seq_len, device=hidden_states.device, dtype=torch.bool),
                diagonal=kv_seq_len - query_len + 1,
            )
            attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Apply attention mask if provided
        if attention_mask is not None:
            # Reshape attention mask for broadcasting
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)

            # Convert to additive mask (0 for positions to attend, -inf for positions to ignore)
            attention_mask = (1.0 - attention_mask.float()) * -10000.0

            attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back: (batch, seq_len, num_heads * head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        # Output projection
        output = self.o_proj(attn_output)

        return output, None, present_key_value
