#!/usr/bin/env python3
"""
Train KhatriVoice language model.

Usage:
    python scripts/train.py --config configs/cooking_v2_test.yaml
    python scripts/train.py --config configs/cooking_v2.yaml
    python scripts/train.py --config configs/cooking_v2.yaml --resume checkpoints/cooking_v2/checkpoint_latest.pt
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from khatrivoice.config.model_config import KhatriVoiceConfig
from khatrivoice.model.khatrivoice import KhatriVoice
from khatrivoice.tokenizer.tokenizer import KhatriTokenizer
from khatrivoice.data.dataset import KhatriDataset, ConversationDataset
from khatrivoice.data.collator import DataCollator
from khatrivoice.data.preprocessing import (
    load_text_file,
    split_train_val_test,
)
from khatrivoice.training.trainer import Trainer
from khatrivoice.utils.seed import set_seed
from khatrivoice.utils.logging import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train KhatriVoice")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cooking_v2_test.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to train on (auto, cpu, cuda)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    return parser.parse_args()


def load_text_dataset(data_path: str, min_samples: int = 10) -> list:
    """
    Load text dataset from a file or directory.

    Args:
        data_path: Path to text file or directory
        min_samples: Minimum number of samples to ensure for training

    Returns:
        List of text samples
    """
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    if path.is_file():
        # Load single text file
        text = load_text_file(path, clean=False)
        if not text:
            raise ValueError(f"File is empty: {data_path}")

        # Split into lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        # If too few lines, split into sentences
        if len(lines) < min_samples:
            from khatrivoice.data.preprocessing import split_sentences
            sentences = []
            for line in lines:
                sentences.extend(split_sentences(line))
            if sentences:
                lines = sentences

        # If still too few, repeat the content
        if len(lines) < min_samples:
            original_lines = lines.copy()
            while len(lines) < min_samples:
                lines.extend(original_lines)

        return lines
    else:
        # Load all text files from directory
        from khatrivoice.data.preprocessing import load_text_files
        texts = load_text_files(path, pattern="*.txt", encoding="utf-8")
        if not texts:
            raise ValueError(f"No text files found in: {data_path}")
        return texts


def main():
    """Main training function."""
    args = parse_args()

    # Setup logging
    setup_logging()

    # Load configuration
    config = KhatriVoiceConfig.load(args.config)

    # Override seed from command line if provided
    if args.seed is not None:
        config.seed = args.seed

    # Set seed
    set_seed(config.seed)

    print(f"\nLoaded configuration from {args.config}")
    print(config)

    # Load dataset
    print("\n" + "=" * 60)
    print("Loading Dataset")
    print("=" * 60)

    print(f"Loading dataset from: {config.data_path}")
    texts = load_text_dataset(config.data_path)

    if not texts:
        raise ValueError("No training texts loaded")

    print(f"  Total samples: {len(texts)}")

    # Create tokenizer
    print("\n" + "=" * 60)
    print("Creating Tokenizer")
    print("=" * 60)

    tokenizer = KhatriTokenizer(lowercase=True)

    # Train tokenizer on the corpus
    print(f"Training tokenizer on corpus...")
    tokenizer.train(texts, vocab_size=config.vocab_size)
    print(f"  Vocabulary size: {tokenizer.vocab_size}")

    # Update model config to match tokenizer's vocab size
    if tokenizer.vocab_size != config.vocab_size:
        print(f"  Updating config.vocab_size from {config.vocab_size} to {tokenizer.vocab_size}")
        config.vocab_size = tokenizer.vocab_size

    # Split dataset (train 80%, val 20%)
    train_texts, val_texts, test_texts = split_train_val_test(
        texts,
        train_ratio=0.8,
        val_ratio=0.2,
        test_ratio=0.0,
        seed=config.seed,
    )
    print(f"  Training samples: {len(train_texts)}")
    print(f"  Validation samples: {len(val_texts)}")

    # Create datasets
    print("\n" + "=" * 60)
    print("Creating Datasets")
    print("=" * 60)

    train_dataset = KhatriDataset(
        tokenizer=tokenizer,
        texts=train_texts,
        max_length=config.max_sequence_length,
    )

    val_dataset = KhatriDataset(
        tokenizer=tokenizer,
        texts=val_texts,
        max_length=config.max_sequence_length,
    )

    print(f"  Train dataset: {len(train_dataset)} samples")
    print(f"  Val dataset: {len(val_dataset)} samples")

    # Create data loaders
    collator = DataCollator(
        pad_token_id=tokenizer.pad_id,
        ignore_index=-100,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    ) if val_texts else None

    # Create model
    print("\n" + "=" * 60)
    print("Creating Model")
    print("=" * 60)

    model = KhatriVoice(config)
    num_params = model.get_num_params()
    num_trainable = model.get_num_trainable_params()

    print(f"  Model parameters: {num_params:,}")
    print(f"  Trainable parameters: {num_trainable:,}")
    print(f"  Model size: {num_params * 4 / 1024 / 1024:.2f} MB (fp32)")

    # Setup device
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Create trainer
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
        tokenizer=tokenizer,
        device=device,
    )

    # Resume from checkpoint if provided
    if args.resume:
        trainer.resume(args.resume)

    # Train
    trainer.train()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Checkpoints saved to: {config.checkpoint_dir}")
    print(f"  Tokenizer saved to: {config.checkpoint_dir}/tokenizer")


if __name__ == "__main__":
    main()
