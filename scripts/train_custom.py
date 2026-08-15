#!/usr/bin/env python3
"""
Train KhatriVoice on your custom data.

Usage:
    python scripts/train_custom.py --data path/to/your/data.txt --config configs/tiny.yaml
"""

import argparse
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from khatrivoice.config.model_config import KhatriVoiceConfig
from khatrivoice.model.khatrivoice import KhatriVoice
from khatrivoice.tokenizer.tokenizer import KhatriTokenizer
from khatrivoice.data.dataset import KhatriDataset
from khatrivoice.data.collator import DataCollator
from khatrivoice.data.preprocessing import load_text_files, split_train_val_test
from khatrivoice.training.trainer import Trainer
from khatrivoice.utils.seed import set_seed
from khatrivoice.utils.logging import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Train KhatriVoice on custom data")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training data (file or directory)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tiny.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=1000,
        help="Vocabulary size",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for model and tokenizer",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto, cpu, cuda)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup
    setup_logging()
    set_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("KhatriVoice Training")
    print("=" * 60)

    # Load configuration
    config = KhatriVoiceConfig.load(args.config)
    config.vocab_size = args.vocab_size
    print(f"\nConfiguration: {args.config}")
    print(config)

    # Load data
    data_path = Path(args.data)
    print(f"\nLoading data from: {data_path}")

    if data_path.is_file():
        # Single file
        with open(data_path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    elif data_path.is_dir():
        # Directory of files
        texts = load_text_files(data_path, pattern="*.txt")
    else:
        raise ValueError(f"Data path not found: {data_path}")

    print(f"Loaded {len(texts)} text samples")

    # Split data
    train_texts, val_texts, test_texts = split_train_val_test(
        texts,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )
    print(f"  Train: {len(train_texts)} samples")
    print(f"  Val: {len(val_texts)} samples")
    print(f"  Test: {len(test_texts)} samples")

    # Train tokenizer
    print("\n" + "=" * 60)
    print("Training Tokenizer")
    print("=" * 60)

    tokenizer = KhatriTokenizer(lowercase=True)
    tokenizer.train(texts, vocab_size=config.vocab_size)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Save tokenizer
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to: {tokenizer_path}")

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
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
    ) if val_dataset else None

    # Create model
    print("\n" + "=" * 60)
    print("Creating Model")
    print("=" * 60)

    model = KhatriVoice(config)
    model.print_parameter_summary()

    # Train
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    checkpoint_dir = output_dir / "checkpoints"

    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=args.device,
        checkpoint_dir=str(checkpoint_dir),
        resume_from=args.resume,
    )

    final_metrics = trainer.train()

    # Save final model
    print("\n" + "=" * 60)
    print("Saving Final Model")
    print("=" * 60)

    final_model_path = output_dir / "model_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
    }, final_model_path)
    print(f"Model saved to: {final_model_path}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final loss: {final_metrics['loss']:.4f}")
    print(f"Final perplexity: {final_metrics['perplexity']:.2f}")
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
