"""
KhatriVoice Model Configuration.

This module defines the configuration dataclass for KhatriVoice model architecture,
training parameters, and all hyperparameters needed for the language model.
"""

from dataclasses import dataclass, asdict
from typing import Optional
import yaml
from pathlib import Path


@dataclass
class KhatriVoiceConfig:
    """
    Configuration for KhatriVoice model architecture and training.

    This configuration defines all parameters needed to instantiate and train
    the KhatriVoice language model from scratch.

    Attributes:
        # Model Architecture
        vocab_size: Size of the tokenizer vocabulary
        hidden_size: Dimension of the hidden representations
        num_layers: Number of transformer blocks
        num_attention_heads: Number of attention heads for queries
        num_kv_heads: Number of attention heads for keys/values (for GQA)
        intermediate_size: Dimension of the MLP intermediate layer
        max_sequence_length: Maximum context window length
        rope_theta: Base frequency for Rotary Position Embeddings
        dropout: Dropout probability for regularization

        # Training Parameters
        batch_size: Training batch size
        gradient_accumulation_steps: Steps to accumulate gradients before update
        learning_rate: Initial learning rate
        weight_decay: Weight decay for AdamW
        beta1: AdamW beta1 parameter
        beta2: AdamW beta2 parameter
        max_grad_norm: Maximum gradient norm for clipping
        warmup_steps: Number of warmup steps for LR scheduler
        max_steps: Maximum training steps
        eval_steps: Steps between evaluations
        save_steps: Steps between checkpoint saves
        log_steps: Steps between logging

        # Data Parameters
        seed: Random seed for reproducibility
        data_path: Path to training data
        tokenizer_path: Path to tokenizer vocabulary

        # Checkpointing
        checkpoint_dir: Directory to save checkpoints
        resume_from: Path to resume training from checkpoint

        # Device
        device: Device to use (auto/cpu/cuda)
    """

    # Model Architecture
    vocab_size: int = 32000
    hidden_size: int = 512
    num_layers: int = 8
    num_attention_heads: int = 8
    num_kv_heads: int = 8
    intermediate_size: int = 2048
    max_sequence_length: int = 512
    rope_theta: float = 10000.0
    dropout: float = 0.1

    # Training Parameters
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    max_steps: int = 10000
    eval_steps: int = 500
    save_steps: int = 1000
    log_steps: int = 10

    # Data Parameters
    seed: int = 42
    data_path: str = "data/processed"
    tokenizer_path: str = "data/processed/tokenizer"

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    resume_from: Optional[str] = None

    # Device
    device: str = "auto"

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        self._validate()

    def _validate(self) -> None:
        """Validate that all parameters have sensible values."""
        # Model architecture validations
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.hidden_size > 0, "hidden_size must be positive"
        assert self.num_layers > 0, "num_layers must be positive"
        assert self.num_attention_heads > 0, "num_attention_heads must be positive"
        assert self.num_kv_heads > 0, "num_kv_heads must be positive"
        assert self.intermediate_size > 0, "intermediate_size must be positive"
        assert self.max_sequence_length > 0, "max_sequence_length must be positive"
        assert self.rope_theta > 0, "rope_theta must be positive"
        assert 0.0 <= self.dropout < 1.0, "dropout must be in [0, 1)"

        # Attention dimension validation
        assert self.hidden_size % self.num_attention_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"
        assert self.num_attention_heads % self.num_kv_heads == 0, \
            f"num_attention_heads ({self.num_attention_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"

        # Training parameter validations
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.gradient_accumulation_steps > 0, "gradient_accumulation_steps must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.weight_decay >= 0, "weight_decay must be non-negative"
        assert 0 < self.beta1 < 1, "beta1 must be in (0, 1)"
        assert 0 < self.beta2 < 1, "beta2 must be in (0, 1)"
        assert self.max_grad_norm > 0, "max_grad_norm must be positive"
        assert self.warmup_steps >= 0, "warmup_steps must be non-negative"
        assert self.max_steps > 0, "max_steps must be positive"

    @classmethod
    def load(cls, path: str) -> "KhatriVoiceConfig":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            KhatriVoiceConfig instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def save(self, path: str) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Path to save configuration file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return asdict(self)

    def __str__(self) -> str:
        """Return string representation of configuration."""
        lines = ["KhatriVoiceConfig:"]
        for key, value in asdict(self).items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)
