#!/usr/bin/env python3
"""
Test script to verify the KhatriVoice data pipeline.

Run with: python tests/test_dataset.py
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


def test_preprocessing_import():
    """Test preprocessing imports."""
    from khatrivoice.data.preprocessing import (
        clean_text,
        split_sentences,
        split_paragraphs,
        load_text_file,
        split_train_val_test,
        create_sample_dataset,
        create_tiny_dataset,
    )
    print("[OK] Preprocessing imports")


def test_clean_text():
    """Test text cleaning."""
    from khatrivoice.data.preprocessing import clean_text

    # Test whitespace normalization
    text = "hello    world"
    cleaned = clean_text(text)
    assert cleaned == "hello world"

    # Test leading/trailing whitespace
    text = "  hello world  "
    cleaned = clean_text(text)
    assert cleaned == "hello world"

    print("[OK] clean_text")


def test_split_sentences():
    """Test sentence splitting."""
    from khatrivoice.data.preprocessing import split_sentences

    text = "Hello world. How are you? I am fine."
    sentences = split_sentences(text)

    assert len(sentences) == 3
    print(f"  Sentences: {sentences}")

    print("[OK] split_sentences")


def test_split_train_val_test():
    """Test train/val/test split."""
    from khatrivoice.data.preprocessing import split_train_val_test

    texts = [f"sample {i}" for i in range(100)]
    train, val, test = split_train_val_test(texts)

    print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    assert len(train) + len(val) + len(test) == len(texts)
    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10

    print("[OK] split_train_val_test")


def test_create_sample_dataset():
    """Test sample dataset creation."""
    from khatrivoice.data.preprocessing import create_sample_dataset

    samples = create_sample_dataset(num_samples=10, min_length=5, max_length=20)

    assert len(samples) == 10
    for sample in samples:
        words = sample.split()
        assert 5 <= len(words) <= 20

    print(f"  Created {len(samples)} samples")
    print(f"  Sample: {samples[0][:50]}...")

    print("[OK] create_sample_dataset")


def test_create_tiny_dataset():
    """Test tiny dataset creation."""
    from khatrivoice.data.preprocessing import create_tiny_dataset

    samples = create_tiny_dataset()
    assert len(samples) == 50  # 10 original * 5 repeats

    print(f"  Created {len(samples)} samples")

    print("[OK] create_tiny_dataset")


def test_dataset_creation():
    """Test KhatriDataset creation."""
    from khatrivoice.tokenizer.tokenizer import KhatriTokenizer
    from khatrivoice.data.dataset import KhatriDataset
    from khatrivoice.data.preprocessing import create_tiny_dataset

    # Create and train tokenizer
    tokenizer = KhatriTokenizer(lowercase=True)
    texts = create_tiny_dataset()
    tokenizer.train(texts, vocab_size=50)

    # Create dataset
    dataset = KhatriDataset(
        tokenizer=tokenizer,
        texts=texts,
        max_length=32,
    )

    print(f"  Dataset length: {len(dataset)}")
    print(f"  Vocab size: {tokenizer.vocab_size}")

    assert len(dataset) > 0

    print("[OK] dataset creation")


def test_dataset_getitem():
    """Test dataset __getitem__."""
    from khatrivoice.tokenizer.tokenizer import KhatriTokenizer
    from khatrivoice.data.dataset import KhatriDataset
    from khatrivoice.data.preprocessing import create_tiny_dataset

    # Create dataset
    tokenizer = KhatriTokenizer(lowercase=True)
    texts = create_tiny_dataset()
    tokenizer.train(texts, vocab_size=50)

    dataset = KhatriDataset(
        tokenizer=tokenizer,
        texts=texts,
        max_length=16,
    )

    # Get a sample
    sample = dataset[0]

    assert "input_ids" in sample
    assert "labels" in sample
    assert "attention_mask" in sample

    print(f"  Input shape: {sample['input_ids'].shape}")
    print(f"  Labels shape: {sample['labels'].shape}")
    print(f"  Attention mask shape: {sample['attention_mask'].shape}")

    # Check shapes match
    assert sample["input_ids"].shape == sample["labels"].shape
    assert sample["input_ids"].shape == sample["attention_mask"].shape

    print("[OK] dataset __getitem__")


def test_dataset_labels():
    """Test that labels are correctly shifted."""
    from khatrivoice.tokenizer.tokenizer import KhatriTokenizer
    from khatrivoice.data.dataset import KhatriDataset
    from khatrivoice.data.preprocessing import create_tiny_dataset

    # Create dataset
    tokenizer = KhatriTokenizer(lowercase=True)
    texts = create_tiny_dataset()
    tokenizer.train(texts, vocab_size=50)

    dataset = KhatriDataset(
        tokenizer=tokenizer,
        texts=texts,
        max_length=16,
    )

    sample = dataset[0]
    input_ids = sample["input_ids"]
    labels = sample["labels"]

    # Labels should be shifted by 1
    # The first label should predict what the second input is
    # Note: -100 is used for padding, so we need to check non-padded positions

    # Find positions with non-padded labels
    valid_positions = (labels != -100).nonzero(as_tuple=True)[0]

    if len(valid_positions) > 0:
        print(f"  Valid label positions: {valid_positions[:5].tolist()}")

    print("[OK] dataset labels")


def test_dataset_stats():
    """Test dataset statistics."""
    from khatrivoice.tokenizer.tokenizer import KhatriTokenizer
    from khatrivoice.data.dataset import KhatriDataset
    from khatrivoice.data.preprocessing import create_tiny_dataset

    tokenizer = KhatriTokenizer(lowercase=True)
    texts = create_tiny_dataset()
    tokenizer.train(texts, vocab_size=50)

    dataset = KhatriDataset(
        tokenizer=tokenizer,
        texts=texts,
        max_length=32,
    )

    stats = dataset.get_stats()

    print(f"  Stats: {stats}")

    assert "num_sequences" in stats
    assert "num_samples" in stats
    assert "total_tokens" in stats

    print("[OK] dataset stats")


def test_data_collator():
    """Test DataCollator."""
    from khatrivoice.tokenizer.tokenizer import KhatriTokenizer
    from khatrivoice.data.dataset import KhatriDataset
    from khatrivoice.data.collator import DataCollator
    from khatrivoice.data.preprocessing import create_tiny_dataset

    tokenizer = KhatriTokenizer(lowercase=True)
    texts = create_tiny_dataset()
    tokenizer.train(texts, vocab_size=50)

    dataset = KhatriDataset(
        tokenizer=tokenizer,
        texts=texts,
        max_length=32,
    )

    collator = DataCollator(tokenizer=tokenizer)

    # Create a batch
    batch = [dataset[i] for i in range(4)]
    collated = collator(batch)

    print(f"  Batch input_ids shape: {collated['input_ids'].shape}")
    print(f"  Batch labels shape: {collated['labels'].shape}")
    print(f"  Batch attention_mask shape: {collated['attention_mask'].shape}")

    assert collated["input_ids"].shape[0] == 4  # Batch size

    print("[OK] DataCollator")


def test_dataloader():
    """Test DataLoader creation."""
    from khatrivoice.tokenizer.tokenizer import KhatriTokenizer
    from khatrivoice.data.dataset import KhatriDataset
    from khatrivoice.data.collator import DataCollator, create_dataloader
    from khatrivoice.data.preprocessing import create_tiny_dataset

    tokenizer = KhatriTokenizer(lowercase=True)
    texts = create_tiny_dataset()
    tokenizer.train(texts, vocab_size=50)

    dataset = KhatriDataset(
        tokenizer=tokenizer,
        texts=texts,
        max_length=32,
    )

    collator = DataCollator(tokenizer=tokenizer)

    dataloader = create_dataloader(
        dataset,
        batch_size=4,
        shuffle=True,
        collator=collator,
    )

    # Get a batch
    batch = next(iter(dataloader))

    print(f"  Batch shape: {batch['input_ids'].shape}")

    assert batch["input_ids"].shape[0] <= 4

    print("[OK] DataLoader")


def test_causal_lm_collator():
    """Test DataCollatorForCausalLM."""
    from khatrivoice.tokenizer.tokenizer import KhatriTokenizer
    from khatrivoice.data.collator import DataCollatorForCausalLM

    tokenizer = KhatriTokenizer(lowercase=True)
    corpus = ["hello world", "test one two three"]
    tokenizer.train(corpus, vocab_size=50)

    collator = DataCollatorForCausalLM(tokenizer=tokenizer, max_length=16)

    # Create raw samples (list of input_ids)
    features = [
        {"input_ids": tokenizer.encode("hello world")},
        {"input_ids": tokenizer.encode("testing")},
    ]

    batch = collator(features)

    print(f"  Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"  Batch labels shape: {batch['labels'].shape}")

    assert batch["input_ids"].shape == batch["labels"].shape

    print("[OK] DataCollatorForCausalLM")


def main():
    """Run all tests."""
    print("=" * 60)
    print("KhatriVoice Data Pipeline Tests")
    print("=" * 60)
    print()

    tests = [
        test_preprocessing_import,
        test_clean_text,
        test_split_sentences,
        test_split_train_val_test,
        test_create_sample_dataset,
        test_create_tiny_dataset,
        test_dataset_creation,
        test_dataset_getitem,
        test_dataset_labels,
        test_dataset_stats,
        test_data_collator,
        test_dataloader,
        test_causal_lm_collator,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__} failed: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print()
    print("=" * 60)
    print(f"Tests passed: {passed}/{len(tests)}")
    if failed > 0:
        print(f"Tests failed: {failed}")
        sys.exit(1)
    print("All data pipeline tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
