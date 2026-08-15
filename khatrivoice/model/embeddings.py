"""
Token Embeddings for KhatriVoice.

This module implements the token embedding layer for the language model.
"""

import torch
import torch.nn as nn
from torch import Tensor


class TokenEmbedding(nn.Module):
    """
    Token embedding layer for KhatriVoice.

    Converts token IDs to dense embeddings. This is the first layer
    in the transformer that maps discrete token IDs to continuous vectors.

    Attributes:
        num_embeddings: Vocabulary size
        embedding_dim: Dimension of the embeddings
        padding_idx: Index for padding token (not updated during training)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = 0,
    ) -> None:
        """
        Initialize the token embedding layer.

        Args:
            num_embeddings: Size of the vocabulary
            embedding_dim: Dimension of the embeddings
            padding_idx: Index for padding token
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # Create embedding table
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize embedding weights using normal distribution."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        # Zero out padding embedding
        if self.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.padding_idx].fill_(0)

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]

        Returns:
            Embeddings [batch_size, seq_len, embedding_dim]
        """
        return self.embedding(input_ids)

    def get_output_norm(self) -> float:
        """Get the norm of embedding weights (for debugging)."""
        return self.embedding.weight.norm().item()

    def __repr__(self) -> str:
        return (
            f"TokenEmbedding("
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim})"
        )


class Embedding(nn.Module):
    """
    Combined embedding module for the transformer.

    Optionally includes:
    - Token embeddings
    - Position embeddings (learned, or None if using RoPE)

    For KhatriVoice, position embeddings are optional since we use RoPE.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int,
        dropout: float = 0.0,
        padding_token_id: int = 0,
        use_position_embeddings: bool = False,
    ) -> None:
        """
        Initialize embeddings.

        Args:
            vocab_size: Vocabulary size
            hidden_size: Hidden dimension
            max_position_embeddings: Maximum sequence length
            dropout: Dropout probability
            padding_token_id: Padding token ID
            use_position_embeddings: Whether to use learned position embeddings
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.use_position_embeddings = use_position_embeddings

        # Token embeddings
        self.token_embedding = TokenEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=padding_token_id,
        )

        # Position embeddings (optional, not used with RoPE)
        if use_position_embeddings:
            self.position_embedding = nn.Embedding(
                num_embeddings=max_position_embeddings,
                embedding_dim=hidden_size,
            )
            self._init_position_embeddings()
        else:
            self.position_embedding = None

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Register position indices buffer
        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand(1, -1),
            persistent=False,
        )

    def _init_position_embeddings(self) -> None:
        """Initialize position embeddings."""
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.01)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            position_ids: Optional position IDs [batch_size, seq_len]

        Returns:
            Hidden states [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len = input_ids.shape

        # Get token embeddings
        hidden_states = self.token_embedding(input_ids)

        # Add position embeddings if enabled
        if self.use_position_embeddings and self.position_embedding is not None:
            if position_ids is None:
                position_ids = self.position_ids[:, :seq_len]
            position_embeddings = self.position_embedding(position_ids)
            hidden_states = hidden_states + position_embeddings

        # Apply dropout
        hidden_states = self.dropout(hidden_states)

        return hidden_states

    def __repr__(self) -> str:
        return (
            f"Embedding("
            f"vocab_size={self.vocab_size}, "
            f"hidden_size={self.hidden_size}, "
            f"use_position_embeddings={self.use_position_embeddings})"
        )
