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
from khatrivoice.data.dataset import KhatriDataset, ConversationDataset
from khatrivoice.data.collator import DataCollator
from khatrivoice.data.preprocessing import (
    create_tiny_dataset,
    split_train_val_test,
    load_text_file,
    parse_conversation_file,
    conversations_to_training_data,
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
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--use-tiny-dataset",
        action="store_true",
        help="Use built-in tiny dataset for overfit testing (ignores data_path)",
    )
    parser.add_argument(
        "--conversation-mode",
        action="store_true",
        help="Treat data as User/AI conversation pairs with proper label masking",
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
        text = load_text_file(path, clean=False)  # Don't clean to preserve structure
        if not text:
            raise ValueError(f"File is empty: {data_path}")

        # Split into lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        # If too few lines, split into sentences or repeat
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

    # Load or create dataset
    print("\n" + "=" * 60)
    print("Loading Dataset")
    print("=" * 60)

    use_tiny = args.use_tiny_dataset

    if use_tiny:
        # Use built-in tiny dataset for testing
        print("Using built-in tiny dataset for overfit testing...")
        texts = create_tiny_dataset()
        print(f"  Dataset: built-in tiny dataset")
    elif Path(config.data_path).exists():
        # Load user-provided dataset
        print(f"Loading dataset from: {config.data_path}")
        texts = load_text_dataset(config.data_path)
    else:
        # Fallback to built-in dataset if config path doesn't exist
        print(f"Data path not found: {config.data_path}")
        print("Using built-in tiny dataset for overfit testing...")
        texts = create_tiny_dataset()
        print(f"  Dataset: built-in tiny dataset")

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

    # IMPORTANT: Update model config to match tokenizer's vocab size
    # This ensures embedding/output dimensions match the trained tokenizer
    if tokenizer.vocab_size != config.vocab_size:
        print(f"  Updating config.vocab_size from {config.vocab_size} to {tokenizer.vocab_size}")
        config.vocab_size = tokenizer.vocab_size

    # Split dataset (train 80%, val 20%, test 0%)
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

    # Use conversation dataset if in conversation mode
    if args.conversation_mode:
        print("Using ConversationDataset with labeled assistant responses")
        train_dataset = ConversationDataset(
            tokenizer=tokenizer,
            texts=train_texts,
            max_length=config.max_sequence_length,
            mask_user_tokens=True,
        )

        val_dataset = ConversationDataset(
            tokenizer=tokenizer,
            texts=val_texts,
            max_length=config.max_sequence_length,
            mask_user_tokens=True,
        ) if val_texts else None
    else:
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

    # Setup checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save tokenizer alongside checkpoints for inference
    tokenizer_save_path = checkpoint_dir / "tokenizer"
    tokenizer.save(tokenizer_save_path)
    print(f"\nTokenizer saved to: {tokenizer_save_path}")

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
        checkpoint_dir=str(checkpoint_dir),
        resume_from=args.resume,
    )

    # Train
    final_metrics = trainer.train()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final loss: {final_metrics['loss']:.4f}")
    print(f"Final perplexity: {final_metrics['perplexity']:.2f}")
    print(f"Checkpoints saved to: {checkpoint_dir}/")
    print(f"Tokenizer saved to: {tokenizer_save_path}")


if __name__ == "__main__":
    main()
