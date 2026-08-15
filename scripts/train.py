#!/usr/bin/env python3
"""
Train KhatriVoice language model.

Usage:
    python scripts/train.py --config configs/tiny.yaml
    python scripts/train.py --config configs/tiny.yaml --resume checkpoints/checkpoint_latest.pt
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
from khatrivoice.data.dataset import KhatriDataset
from khatrivoice.data.collator import DataCollator
from khatrivoice.data.preprocessing import create_tiny_dataset, split_train_val_test
from khatrivoice.training.trainer import Trainer
from khatrivoice.utils.seed import set_seed
from khatrivoice.utils.logging import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train KhatriVoice")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tiny.yaml",
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
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Setup logging
    setup_logging()

    # Set seed
    set_seed(args.seed)

    # Load configuration
    config = KhatriVoiceConfig.load(args.config)
    print(f"\nLoaded configuration from {args.config}")
    print(config)

    # Create tokenizer and train on tiny dataset
    print("\n" + "=" * 60)
    print("Creating Tokenizer")
    print("=" * 60)

    tokenizer = KhatriTokenizer(lowercase=True)

    # Create tiny dataset for testing
    print("Creating tiny dataset for testing...")
    texts = create_tiny_dataset()
    print(f"  Total samples: {len(texts)}")

    # Train tokenizer
    tokenizer.train(texts, vocab_size=config.vocab_size)
    print(f"  Vocabulary size: {tokenizer.vocab_size}")

    # Split dataset (train 80%, val 10%, test 10%)
    train_texts, val_texts, _ = split_train_val_test(texts, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
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
    ) if val_texts else None

    print(f"Train dataset: {len(train_dataset)} samples")
    if val_dataset:
        print(f"Val dataset: {len(val_dataset)} samples")

    # Create data loaders
    collator = DataCollator(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    ) if val_dataset else None

    # Create model
    print("\n" + "=" * 60)
    print("Creating Model")
    print("=" * 60)

    model = KhatriVoice(config)
    model.print_parameter_summary()

    # Create trainer
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=args.device,
        checkpoint_dir="checkpoints",
        resume_from=args.resume,
    )

    # Train
    final_metrics = trainer.train()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final loss: {final_metrics['loss']:.4f}")
    print(f"Final perplexity: {final_metrics['perplexity']:.2f}")
    print(f"Checkpoints saved to: checkpoints/")


if __name__ == "__main__":
    main()
