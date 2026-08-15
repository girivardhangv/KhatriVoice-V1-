"""
MLP (Feed-Forward Network) for KhatriVoice transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    SwiGLU-style MLP with gated activation.

    This implementation uses a gated linear unit with SiLU activation,
    following the LLaMA architecture.

    Args:
        hidden_size: Input and output dimension
        intermediate_size: Dimension of the intermediate layer
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MLP.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)

        Returns:
            Output tensor of shape (batch, seq_len, hidden_size)
        """
        # Gated activation: gate * up (element-wise)
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up

        # Down projection
        x = self.down_proj(x)
        x = self.dropout(x)

        return x


class StandardMLP(nn.Module):
    """
    Standard MLP with GELU activation.

    Args:
        hidden_size: Input and output dimension
        intermediate_size: Dimension of the intermediate layer
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for standard MLP.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)

        Returns:
            Output tensor of shape (batch, seq_len, hidden_size)
        """
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
