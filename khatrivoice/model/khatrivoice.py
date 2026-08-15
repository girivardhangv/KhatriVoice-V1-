"""
KhatriVoice Language Model.

This module implements the complete KhatriVoice model:
- Token embeddings
- Transformer backbone
- Language model output head

All parameters are initialized from random. No pretrained weights.
"""

import math
from typing import Optional, Tuple, List, Dict, Any
import torch
import torch.nn as nn
from torch import Tensor

from khatrivoice.config.model_config import KhatriVoiceConfig
from khatrivoice.model.embeddings import Embedding
from khatrivoice.model.transformer import Transformer


class KhatriVoice(nn.Module):
    """
    KhatriVoice: A decoder-only autoregressive Transformer language model.

    This model is designed for the SSK Khatri language and implements:
    - Token embeddings with padding support
    - Rotary Position Embeddings (RoPE)
    - Causal self-attention with Grouped Query Attention (GQA)
    - SwiGLU feed-forward networks
    - RMSNorm pre-normalization
    - Residual connections

    All parameters are randomly initialized. No pretrained weights are used.

    Attributes:
        config: Model configuration
        embedding: Token embedding layer
        transformer: Transformer backbone
        lm_head: Language model output head
    """

    def __init__(self, config: KhatriVoiceConfig) -> None:
        """
        Initialize KhatriVoice model.

        Args:
            config: KhatriVoiceConfig with model parameters
        """
        super().__init__()

        self.config = config

        # Token embeddings (no position embeddings, using RoPE)
        self.embedding = Embedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            max_position_embeddings=config.max_sequence_length,
            dropout=config.dropout,
            use_position_embeddings=False,  # Using RoPE
        )

        # Transformer backbone
        self.transformer = Transformer(
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_kv_heads,
            intermediate_size=config.intermediate_size,
            max_seq_len=config.max_sequence_length,
            dropout=config.dropout,
            rope_theta=config.rope_theta,
        )

        # Language model head (linear projection to vocabulary)
        # Weight tying: share weights with embedding layer
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        # Tie weights between embedding and output head
        self.lm_head.weight = self.embedding.token_embedding.embedding.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for all parameters."""
        # Already initialized in sub-modules, but we can add custom init here
        pass

    def get_input_embeddings(self) -> nn.Embedding:
        """Get the input embedding layer."""
        return self.embedding.token_embedding.embedding

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Set the input embedding layer."""
        self.embedding.token_embedding.embedding = value

    def get_output_embeddings(self) -> nn.Linear:
        """Get the output embedding (lm_head)."""
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        """Set the output embedding (lm_head)."""
        self.lm_head = value

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            past_key_values: Cached KV for generation
            use_cache: Whether to return cache
            labels: Labels for loss calculation [batch_size, seq_len]

        Returns:
            Dictionary with 'logits', 'loss' (if labels provided), and optionally 'past_key_values'
        """
        batch_size, seq_len = input_ids.shape

        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Get embeddings
        hidden_states = self.embedding(input_ids)

        # Ensure attention mask has the right shape
        if attention_mask is not None:
            # attention_mask: [batch, seq_len] -> [batch, 1, 1, seq_len] for broadcasting
            attention_mask = attention_mask[:, None, None, :]

        # Apply transformer
        hidden_states, new_past_key_values = self.transformer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        # Compute logits
        logits = self.lm_head(hidden_states)

        # Prepare output
        output = {"logits": logits}

        # Compute loss if labels provided
        if labels is not None:
            loss = self._compute_loss(logits, labels)
            output["loss"] = loss

        # Add cache if requested
        if use_cache:
            output["past_key_values"] = new_past_key_values

        return output

    def _compute_loss(
        self,
        logits: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """
        Compute cross-entropy loss.

        Args:
            logits: Model logits [batch, seq_len, vocab_size]
            labels: Target labels [batch, seq_len]

        Returns:
            Loss scalar
        """
        # Shift logits and labels for causal LM
        # Predict next token: logits[i] predicts labels[i+1]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten
        batch_size, seq_len, vocab_size = shift_logits.shape
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)

        # Compute cross-entropy loss
        # -100 labels are ignored (padding)
        loss = nn.functional.cross_entropy(
            shift_logits,
            shift_labels,
            ignore_index=-100,
            reduction="mean",
        )

        return loss

    def prepare_inputs_for_generation(
        self,
        input_ids: Tensor,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        attention_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Prepare inputs for generation.

        Args:
            input_ids: Token IDs
            past_key_values: Cached KV
            attention_mask: Attention mask

        Returns:
            Dictionary of inputs for forward pass
        """
        # If we have cache, only need last token
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": True,
        }

    def count_parameters(self) -> Dict[str, int]:
        """
        Count model parameters.

        Returns:
            Dictionary with parameter counts by category
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        embedding_params = sum(
            p.numel() for p in self.embedding.parameters()
        )
        attention_params = sum(
            p.numel() for layer in self.transformer.layers
            for p in layer.self_attn.parameters()
        )
        mlp_params = sum(
            p.numel() for layer in self.transformer.layers
            for p in layer.mlp.parameters()
        )
        norm_params = sum(
            p.numel() for layer in self.transformer.layers
            for name, p in layer.named_parameters()
            if "layernorm" in name or "norm" in name
        )
        norm_params += sum(p.numel() for p in self.transformer.norm.parameters())

        # Output head is tied, so don't double count
        output_params = 0  # Tied with embeddings

        return {
            "total": total,
            "trainable": trainable,
            "embedding": embedding_params,
            "attention": attention_params,
            "mlp": mlp_params,
            "normalization": norm_params,
            "output_head": output_params,
        }

    def print_parameter_summary(self) -> None:
        """Print a summary of model parameters."""
        params = self.count_parameters()

        print("=" * 50)
        print("KhatriVoice Parameter Summary")
        print("=" * 50)
        print(f"Total parameters: {params['total']:,}")
        print(f"Trainable parameters: {params['trainable']:,}")
        print()
        print("By component:")
        print(f"  Embedding: {params['embedding']:,}")
        print(f"  Attention: {params['attention']:,}")
        print(f"  MLP: {params['mlp']:,}")
        print(f"  Normalization: {params['normalization']:,}")
        print(f"  Output head (tied): {params['output_head']:,}")
        print("=" * 50)

    def __repr__(self) -> str:
        return (
            f"KhatriVoice(\n"
            f"  vocab_size={self.config.vocab_size},\n"
            f"  hidden_size={self.config.hidden_size},\n"
            f"  num_layers={self.config.num_layers},\n"
            f"  num_heads={self.config.num_heads},\n"
            f"  num_kv_heads={self.config.num_kv_heads},\n"
            f"  intermediate_size={self.config.intermediate_size},\n"
            f"  max_seq_len={self.config.max_sequence_length},\n"
            f")"
        )


def create_model(config: KhatriVoiceConfig) -> KhatriVoice:
    """
    Create a KhatriVoice model from configuration.

    Args:
        config: Model configuration

    Returns:
        KhatriVoice model instance
    """
    return KhatriVoice(config)


def load_model(
    checkpoint_path: str,
    config: Optional[KhatriVoiceConfig] = None,
    device: str = "auto",
) -> KhatriVoice:
    """
    Load a KhatriVoice model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config: Optional config (loaded from checkpoint if not provided)
        device: Device to load model on

    Returns:
        KhatriVoice model instance
    """
    from khatrivoice.utils.device import get_device

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Get config
    if config is None:
        config_dict = checkpoint.get("config", {})
        config = KhatriVoiceConfig.from_dict(config_dict)

    # Create model
    model = KhatriVoice(config)

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])

    # Move to device
    device = get_device(device)
    model = model.to(device)

    return model
