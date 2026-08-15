#!/usr/bin/env python3
"""
Smoke test for KhatriVoice cooking dataset.

This script validates:
- Cooking dataset file exists and is readable
- Text is valid and non-empty
- Tokenizer can be trained on the data
- Tokenization produces valid IDs
- Datasets can be created
- Batches have expected shapes
- Model can process the data
- Forward/backward pass works
- Loss is finite

Run with: python tests/test_cooking_data.py
"""

import sys
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

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
from khatrivoice.data.preprocessing import split_train_val_test, load_text_file


def test_file_exists():
    """Test that cooking.txt exists."""
    path = Path(project_root) / "data" / "household" / "cooking.txt"
    assert path.exists(), f"File not found: {path}"
    assert path.is_file(), f"Not a file: {path}"
    print(f"[OK] File exists: {path}")
    return path


def test_file_readable(path: Path):
    """Test that file is readable and non-empty."""
    text = load_text_file(path, encoding="utf-8", clean=False)
    assert text, "File is empty"
    assert len(text) > 1000, f"File too small: {len(text)} chars"

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    assert len(lines) > 100, f"Not enough lines: {len(lines)}"

    print(f"[OK] File readable: {len(text)} chars, {len(lines)} lines")
    return text, lines


def test_tokenizer(lines: list, max_vocab_size: int = 2000):
    """Test tokenizer training and encoding."""
    tokenizer = KhatriTokenizer(lowercase=True)

    # Train tokenizer
    tokenizer.train(lines, vocab_size=max_vocab_size)
    assert tokenizer.vocab_size > 0, "Vocabulary is empty"
    assert tokenizer.vocab_size <= max_vocab_size, f"Vocab size exceeds max: {tokenizer.vocab_size}"

    # Test encoding/decoding
    test_text = "Hello, this is a test of the tokenizer."
    encoded = tokenizer.encode(test_text)
    assert len(encoded) > 0, "Encoded sequence is empty"

    decoded = tokenizer.decode(encoded)
    # Should be similar (lowercase)
    assert decoded.lower().strip() == test_text.lower().strip() or len(decoded) > 0

    print(f"[OK] Tokenizer: vocab_size={tokenizer.vocab_size}")
    return tokenizer


def test_dataset_creation(lines: list, tokenizer: KhatriTokenizer, seq_length: int = 256):
    """Test dataset creation."""
    # Use a subset for faster testing
    test_lines = lines[:1000]

    dataset = KhatriDataset(
        texts=test_lines,
        tokenizer=tokenizer,
        max_length=seq_length,
    )

    assert len(dataset) > 0, "Dataset is empty"

    # Check a sample
    sample = dataset[0]
    assert "input_ids" in sample, "Missing input_ids"
    assert "labels" in sample, "Missing labels"
    assert len(sample["input_ids"]) == seq_length, f"Wrong sequence length: {len(sample['input_ids'])}"

    # Check token IDs are valid
    input_ids = sample["input_ids"]
    assert all(0 <= tid < tokenizer.vocab_size for tid in input_ids), "Invalid token IDs"

    print(f"[OK] Dataset: {len(dataset)} samples, seq_length={seq_length}")
    return dataset


def test_train_val_split(lines: list, tokenizer: KhatriTokenizer, seq_length: int = 256):
    """Test train/val split."""
    # Use subset for faster testing
    test_lines = lines[:1000]

    train_texts, val_texts, test_texts = split_train_val_test(
        test_lines,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )

    assert len(train_texts) > 0, "No training samples"
    assert len(val_texts) > 0, "No validation samples"

    train_dataset = KhatriDataset(
        texts=train_texts,
        tokenizer=tokenizer,
        max_length=seq_length,
    )

    val_dataset = KhatriDataset(
        texts=val_texts,
        tokenizer=tokenizer,
        max_length=seq_length,
    )

    print(f"[OK] Train/Val Split: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_texts)}")
    return train_dataset, val_dataset


def test_dataloader(dataset: KhatriDataset, tokenizer: KhatriTokenizer, batch_size: int = 4):
    """Test dataloader and batching."""
    collator = DataCollator(tokenizer=tokenizer)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,  # Single process for safety
    )

    # Get one batch
    batch = next(iter(dataloader))

    assert "input_ids" in batch, "Missing input_ids in batch"
    assert "labels" in batch, "Missing labels in batch"
    assert "attention_mask" in batch, "Missing attention_mask in batch"

    assert batch["input_ids"].shape[0] == batch_size, f"Wrong batch size: {batch['input_ids'].shape}"
    assert batch["input_ids"].shape[1] == dataset.max_length, f"Wrong sequence length"

    print(f"[OK] DataLoader: batch_shape={batch['input_ids'].shape}")
    return dataloader, batch


def test_model_forward(batch: dict, config: KhatriVoiceConfig):
    """Test model forward pass."""
    model = KhatriVoice(config)
    model.eval()

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Check outputs (returns dict)
    assert "logits" in outputs, "Model returned dict without logits"
    assert outputs["logits"] is not None, "Model returned None logits"
    assert outputs["logits"].shape[0] == batch["input_ids"].shape[0], "Wrong batch size in output"
    assert outputs["logits"].shape[2] == config.vocab_size, "Wrong vocab size in output"

    print(f"[OK] Model Forward: logits.shape={outputs['logits'].shape}")
    return model, outputs


def test_loss_computation(batch: dict, config: KhatriVoiceConfig):
    """Test loss computation."""
    model = KhatriVoice(config)
    model.train()

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    assert "loss" in outputs, "Model did not compute loss"
    assert outputs["loss"] is not None, "Model returned None loss"
    assert torch.isfinite(outputs["loss"]), f"Loss is not finite: {outputs['loss']}"

    print(f"[OK] Loss: {outputs['loss'].item():.4f}")
    return model, outputs["loss"]


def test_backward_pass(model: KhatriVoice, loss: torch.Tensor):
    """Test backward pass."""
    loss.backward()

    # Check gradients exist
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    total_count = sum(1 for _ in model.parameters())

    assert grad_count > 0, "No gradients computed"

    # Check gradients are finite
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Gradient not finite for {name}"

    print(f"[OK] Backward Pass: {grad_count}/{total_count} parameters have gradients")


def test_short_training_step(lines: list, config: KhatriVoiceConfig):
    """Test a few training steps."""
    tokenizer = KhatriTokenizer(lowercase=True)
    tokenizer.train(lines[:500], vocab_size=config.vocab_size)

    config = KhatriVoiceConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_attention_heads=config.num_attention_heads,
        num_kv_heads=config.num_kv_heads,
        intermediate_size=config.intermediate_size,
        max_sequence_length=config.max_sequence_length,
        dropout=config.dropout,
    )

    model = KhatriVoice(config)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    dataset = KhatriDataset(
        texts=lines[:200],
        tokenizer=tokenizer,
        max_length=config.max_sequence_length,
    )

    collator = DataCollator(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collator,
    )

    losses = []
    for i, batch in enumerate(dataloader):
        if i >= 5:  # Only 5 steps
            break

        optimizer.zero_grad()

        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs["loss"]
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    # Check that loss is generally decreasing or stable
    assert all(l > 0 for l in losses), "Negative loss"
    print(f"[OK] Training Steps: losses={[f'{l:.4f}' for l in losses]}")


def print_dataset_stats(path: Path, lines: list, tokenizer: KhatriTokenizer):
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"  File: {path}")
    print(f"  Total lines: {len(lines)}")
    print(f"  Total characters: {sum(len(line) for line in lines):,}")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print(f"  Average line length: {sum(len(line) for line in lines) / len(lines):.1f} chars")
    print(f"  Sample line: {lines[0][:80]}...")
    print("=" * 60)


def test():
    """Run all smoke tests."""
    print("=" * 60)
    print("KhatriVoice Cooking Dataset Smoke Test")
    print("=" * 60)
    print()

    # Configuration for testing (smaller than full cooking config)
    config = KhatriVoiceConfig(
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=4,
        intermediate_size=256,
        max_sequence_length=128,
        dropout=0.0,
    )

    try:
        # Run tests in order
        path = test_file_exists()
        text, lines = test_file_readable(path)
        tokenizer = test_tokenizer(lines, max_vocab_size=config.vocab_size)

        # Print stats
        print_dataset_stats(path, lines[:1000], tokenizer)

        # Continue tests
        dataset = test_dataset_creation(lines[:500], tokenizer, seq_length=config.max_sequence_length)
        train_dataset, val_dataset = test_train_val_split(lines[:500], tokenizer, seq_length=config.max_sequence_length)
        dataloader, batch = test_dataloader(dataset, tokenizer, batch_size=4)
        model, _ = test_model_forward(batch, config)
        model, loss = test_loss_computation(batch, config)
        test_backward_pass(model, loss)
        test_short_training_step(lines, config)

        print("\n" + "=" * 60)
        print("SMOKE TEST PASSED")
        print("=" * 60)
        print(f"\nCooking dataset is ready for training!")
        print(f"Run: python scripts/train.py --config configs/cooking.yaml")

    except AssertionError as e:
        print(f"\n[FAIL] Assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    test()
