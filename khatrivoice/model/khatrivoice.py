"""
KhatriVoice Language Model.

A transformer-based language model using LLaMA-style architecture
with RoPE, GQA, and SwiGLU MLP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union

from khatrivoice.config.model_config import KhatriVoiceConfig
from khatrivoice.model.embeddings import TokenEmbedding, RMSNorm
from khatrivoice.model.block import TransformerBlock


class KhatriVoice(nn.Module):
    """
    KhatriVoice Language Model.

    A causal transformer language model with:
    - Rotary Position Embeddings (RoPE)
    - Grouped Query Attention (GQA)
    - SwiGLU MLP activation
    - RMS Layer Normalization

    Args:
        config: KhatriVoiceConfig with model architecture parameters
    """

    def __init__(self, config: KhatriVoiceConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            max_position=config.max_sequence_length,
            dropout=config.dropout,
        )

        # Transformer layers
        self.transformer = nn.ModuleDict({
            "layers": nn.ModuleList([
                TransformerBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_kv_heads,
                    intermediate_size=config.intermediate_size,
                    max_seq_len=config.max_sequence_length,
                    rope_theta=config.rope_theta,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ])
        })

        # Final layer norm
        self.norm = RMSNorm(config.hidden_size)

        # Output projection (tied with input embeddings)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights between embedding and output
        self.lm_head.weight = self.embedding.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with small random values."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass for KhatriVoice.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len)
            attention_mask: Optional attention mask
            position_ids: Optional position IDs (unused, RoPE handles positions)
            past_key_values: Optional list of cached key-value tuples
            use_cache: Whether to return cached key-value states
            labels: Optional labels for computing loss

        Returns:
            Tuple of (logits, optional loss, optional cached key-values)
        """
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        hidden_states = self.embedding(input_ids)

        # Initialize past key values if not provided
        if past_key_values is None:
            past_key_values = [None] * self.config.num_layers

        # Track present key values for caching
        present_key_values = [] if use_cache else None

        # Pass through transformer layers
        for idx, layer in enumerate(self.transformer["layers"]):
            past_kv = past_key_values[idx] if past_key_values else None

            hidden_states, present_kv = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                use_cache=use_cache,
            )

            if use_cache:
                present_key_values.append(present_kv)

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        # Output projection
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten for cross entropy
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss, present_key_values

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Starting token IDs of shape (batch, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Optional top-k filtering
            top_p: Optional nucleus sampling threshold
            do_sample: Whether to sample or take argmax
            eos_token_id: Optional end-of-sequence token ID
            pad_token_id: Optional padding token ID

        Returns:
            Generated token IDs including input
        """
        self.eval()

        # Initialize
        generated = input_ids.clone()
        past_key_values = None

        for _ in range(max_new_tokens):
            # Get predictions
            logits, _, past_key_values = self.forward(
                generated[:, -self.config.max_sequence_length:],
                past_key_values=past_key_values,
                use_cache=True,
            )

            # Get next token logits
            next_token_logits = logits[:, -1, :]

            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample or take argmax
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to generated
            generated = torch.cat([generated, next_token], dim=-1)

            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

    def get_num_params(self) -> int:
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
