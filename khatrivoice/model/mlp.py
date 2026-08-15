"""
MLP layers for KhatriVoice.

This module implements the SwiGLU feed-forward network,
which is used in many modern language models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network.

    This implements the SwiGLU activation function combined with
    a gated linear unit, which is used in models like LLaMA.

    SwiGLU(x) = Swish(W_gate @ x) * (W_up @ x)

    Where the Swish function is SiLU (Sigmoid Linear Unit).

    Reference: "GLU Variants Improve Transformer" (Shazeer, 2020)

    Attributes:
        hidden_size: Input/output dimension
        intermediate_size: Dimension of the intermediate (up) projection
        bias: Whether to use bias in projections
        gate_proj: Gate projection (applies SwiGLU activation)
        up_proj: Up projection (no activation)
        down_proj: Down projection (back to hidden_size)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        dropout: float = 0.0,
    ) -> None:
        """
        Initialize SwiGLU MLP.

        Args:
            hidden_size: Input/output dimension
            intermediate_size: Dimension of intermediate projection
            bias: Whether to use bias
            dropout: Dropout probability
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.bias = bias

        # Gate projection: hidden -> intermediate
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)

        # Up projection: hidden -> intermediate
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)

        # Down projection: intermediate -> hidden
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize projection weights."""
        # Use smaller init for stability
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.xavier_uniform_(self.down_proj.weight)

        if self.bias:
            nn.init.zeros_(self.gate_proj.bias)
            nn.init.zeros_(self.up_proj.bias)
            nn.init.zeros_(self.down_proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        # Gate + SwiGLU activation: SiLU(W_gate @ x)
        gate = F.silu(self.gate_proj(x))

        # Up projection: W_up @ x
        up = self.up_proj(x)

        # Combine: (SiLU(W_gate @ x)) * (W_up @ x)
        hidden = gate * up

        # Down projection: W_down @ hidden
        output = self.down_proj(hidden)

        # Apply dropout
        output = self.dropout(output)

        return output

    def __repr__(self) -> str:
        return (
            f"SwiGLU("
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size})"
        )


class MLP(nn.Module):
    """
    Standard MLP with configurable activation.

    This is a simpler alternative to SwiGLU for cases where
    a traditional feed-forward network is preferred.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        """
        Initialize MLP.

        Args:
            hidden_size: Input/output dimension
            intermediate_size: Dimension of intermediate projection
            bias: Whether to use bias
            dropout: Dropout probability
            activation: Activation function ('gelu', 'relu', 'silu')
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.bias = bias

        # Up projection
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)

        # Down projection
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Activation
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize projection weights."""
        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.xavier_uniform_(self.down_proj.weight)

        if self.bias:
            nn.init.zeros_(self.up_proj.bias)
            nn.init.zeros_(self.down_proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        x = self.up_proj(x)
        x = self.act(x)
        x = self.down_proj(x)
        x = self.dropout(x)
        return x

    def __repr__(self) -> str:
        return f"MLP(hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size})"


class FeedForwardNetwork(nn.Module):
    """
    Feed-forward network using SwiGLU.

    Alias for SwiGLU for semantic clarity.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        dropout: float = 0.0,
    ) -> None:
        """Initialize FFN."""
        super().__init__()
        self.mlp = SwiGLU(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=bias,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.mlp(x)

    def __repr__(self) -> str:
        return repr(self.mlp)
