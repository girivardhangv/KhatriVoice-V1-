"""
KhatriVoice Model Configuration.

This module defines the configuration dataclass for KhatriVoice model architecture,
training parameters, and all hyperparameters needed for the language model.
"""

from dataclasses import dataclass
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

        # Training parameter validations
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.gradient_accumulation_steps > 0, "gradient_accumulation_steps must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.weight_decay >= 0, "weight_decay must be non-negative"
        assert self.max_grad_norm > 0, "max_grad_norm must be positive"

        # Head dimension validation
        assert self.hidden_size % self.num_attention_heads == 0, \
            "hidden_size must be divisible by num_attention_heads"
        assert self.num_attention_heads % self.num_kv_heads == 0, \
            "num_attention_heads must be divisible by num_kv_heads (for GQA)"

    @property
    def head_dim(self) -> int:
        """Calculate the dimension per attention head."""
        return self.hidden_size // self.num_attention_heads

    @property
    def total_parameters(self) -> int:
        """Estimate total model parameters (approximate)."""
        # Embedding parameters
        embed_params = self.vocab_size * self.hidden_size

        # Per-layer parameters
        # - Attention: Q, K, V projections + output projection
        #   Q: hidden_size * hidden_size
        #   K: hidden_size * (num_kv_heads * head_dim) = hidden_size * (hidden_size * num_kv_heads / num_attention_heads)
        #   V: same as K
        #   Output: hidden_size * hidden_size
        kv_dim = self.num_kv_heads * self.head_dim
        attn_params = (
            self.hidden_size * self.hidden_size +  # Q projection
            self.hidden_size * kv_dim +            # K projection
            self.hidden_size * kv_dim +            # V projection
            self.hidden_size * self.hidden_size    # Output projection
        )

        # - MLP: Two linear layers
        mlp_params = (
            self.hidden_size * self.intermediate_size +  # Gate projection
            self.hidden_size * self.intermediate_size +  # Up projection
            self.intermediate_size * self.hidden_size    # Down projection
        )

        # - LayerNorm: 2 per layer (before attention, before MLP)
        ln_params = 2 * self.hidden_size

        # Total per layer
        layer_params = attn_params + mlp_params + ln_params

        # Total for all layers
        all_layer_params = self.num_layers * layer_params

        # Final layer norm + output head (tied with embeddings)
        final_ln_params = self.hidden_size

        # Total (embeddings are tied, so no separate output head)
        total = embed_params + all_layer_params + final_ln_params

        return total

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            # Model Architecture
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "intermediate_size": self.intermediate_size,
            "max_sequence_length": self.max_sequence_length,
            "rope_theta": self.rope_theta,
            "dropout": self.dropout,
            # Training Parameters
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "max_grad_norm": self.max_grad_norm,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "log_steps": self.log_steps,
            # Data Parameters
            "seed": self.seed,
            "data_path": self.data_path,
            "tokenizer_path": self.tokenizer_path,
            # Checkpointing
            "checkpoint_dir": self.checkpoint_dir,
            "resume_from": self.resume_from,
            # Device
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "KhatriVoiceConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def save(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str | Path) -> "KhatriVoiceConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def __str__(self) -> str:
        """Return a formatted string representation."""
        lines = [
            "KhatriVoice Configuration",
            "=" * 50,
            "",
            "Model Architecture:",
            f"  vocab_size: {self.vocab_size:,}",
            f"  hidden_size: {self.hidden_size}",
            f"  num_layers: {self.num_layers}",
            f"  num_attention_heads: {self.num_attention_heads}",
            f"  num_kv_heads: {self.num_kv_heads}",
            f"  head_dim: {self.head_dim}",
            f"  intermediate_size: {self.intermediate_size}",
            f"  max_sequence_length: {self.max_sequence_length}",
            f"  rope_theta: {self.rope_theta}",
            f"  dropout: {self.dropout}",
            "",
            "Estimated Parameters:",
            f"  total: {self.total_parameters:,}",
            f"  (~{self.total_parameters / 1e6:.2f}M)",
            "",
            "Training Parameters:",
            f"  batch_size: {self.batch_size}",
            f"  gradient_accumulation_steps: {self.gradient_accumulation_steps}",
            f"  learning_rate: {self.learning_rate}",
            f"  weight_decay: {self.weight_decay}",
            f"  max_grad_norm: {self.max_grad_norm}",
            f"  warmup_steps: {self.warmup_steps}",
            f"  max_steps: {self.max_steps}",
            "",
            "Data Parameters:",
            f"  seed: {self.seed}",
            f"  data_path: {self.data_path}",
            f"  tokenizer_path: {self.tokenizer_path}",
            "",
            "Checkpointing:",
            f"  checkpoint_dir: {self.checkpoint_dir}",
            f"  resume_from: {self.resume_from}",
            "",
            f"  device: {self.device}",
        ]
        return "\n".join(lines)


def get_tiny_config() -> KhatriVoiceConfig:
    """
    Get the tiny configuration for CPU testing.

    This configuration is designed to be small enough (~100K parameters)
    to run on CPU and verify the training pipeline works correctly.
    """
    return KhatriVoiceConfig(
        # Tiny model architecture (~100K-1M parameters)
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=4,
        intermediate_size=256,
        max_sequence_length=128,
        rope_theta=10000.0,
        dropout=0.0,
        # Training
        batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_steps=10,
        max_steps=100,
        eval_steps=20,
        save_steps=50,
        log_steps=5,
        # Data
        seed=42,
        data_path="data/processed",
        tokenizer_path="data/processed/tokenizer",
        # Checkpointing
        checkpoint_dir="checkpoints",
        resume_from=None,
        device="auto",
    )


def get_small_config() -> KhatriVoiceConfig:
    """
    Get the small configuration for development.

    This configuration is suitable for quick GPU training experiments.
    """
    return KhatriVoiceConfig(
        # Small model architecture (~10M parameters)
        vocab_size=8000,
        hidden_size=256,
        num_layers=6,
        num_attention_heads=8,
        num_kv_heads=4,
        intermediate_size=1024,
        max_sequence_length=512,
        rope_theta=10000.0,
        dropout=0.1,
        # Training
        batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_steps=100,
        max_steps=10000,
        eval_steps=500,
        save_steps=1000,
        log_steps=10,
        # Data
        seed=42,
        data_path="data/processed",
        tokenizer_path="data/processed/tokenizer",
        # Checkpointing
        checkpoint_dir="checkpoints",
        resume_from=None,
        device="auto",
    )


def get_base_config() -> KhatriVoiceConfig:
    """
    Get the base configuration for full training.

    This configuration is suitable for serious training on GPU infrastructure.
    """
    return KhatriVoiceConfig(
        # Base model architecture (~100M parameters)
        vocab_size=32000,
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        num_kv_heads=6,
        intermediate_size=3072,
        max_sequence_length=1024,
        rope_theta=10000.0,
        dropout=0.1,
        # Training
        batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_steps=1000,
        max_steps=100000,
        eval_steps=1000,
        save_steps=5000,
        log_steps=10,
        # Data
        seed=42,
        data_path="data/processed",
        tokenizer_path="data/processed/tokenizer",
        # Checkpointing
        checkpoint_dir="checkpoints",
        resume_from=None,
        device="auto",
    )
