"""
Text preprocessing utilities for KhatriVoice.

This module provides functions for preparing text data before tokenization
and dataset creation.
"""

import re
from typing import List, Optional
from pathlib import Path


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing.

    Args:
        text: Input text

    Returns:
        Cleaned text
    """
    # Replace multiple whitespace with single space
    text = re.sub(r"\s+", " ", text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences.

    This is a simple sentence splitter that works for most cases.
    For production use, consider using a proper sentence tokenizer.

    Args:
        text: Input text

    Returns:
        List of sentences
    """
    # Simple sentence boundary detection
    # Splits on . ! ? followed by space or end of string
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def split_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs.

    Args:
        text: Input text

    Returns:
        List of paragraphs
    """
    # Split on double newlines
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def load_text_file(
    filepath: str | Path,
    encoding: str = "utf-8",
    clean: bool = True,
) -> str:
    """
    Load text from a file.

    Args:
        filepath: Path to text file
        encoding: File encoding
        clean: Whether to clean the text

    Returns:
        Text content
    """
    filepath = Path(filepath)
    with open(filepath, "r", encoding=encoding) as f:
        text = f.read()

    if clean:
        text = clean_text(text)

    return text


def load_text_files(
    directory: str | Path,
    pattern: str = "*.txt",
    encoding: str = "utf-8",
) -> List[str]:
    """
    Load all text files from a directory.

    Args:
        directory: Directory path
        pattern: Glob pattern for files
        encoding: File encoding

    Returns:
        List of text contents
    """
    directory = Path(directory)
    files = sorted(directory.glob(pattern))

    texts: List[str] = []
    for filepath in files:
        text = load_text_file(filepath, encoding=encoding)
        if text:
            texts.append(text)

    return texts


def split_train_val_test(
    texts: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    shuffle: bool = True,
) -> tuple:
    """
    Split texts into train, validation, and test sets.

    Args:
        texts: List of texts
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        seed: Random seed
        shuffle: Whether to shuffle before splitting

    Returns:
        Tuple of (train_texts, val_texts, test_texts)
    """
    import random

    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1"

    # Shuffle if requested
    if shuffle:
        random.seed(seed)
        texts = texts.copy()
        random.shuffle(texts)

    # Calculate split indices
    n = len(texts)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_texts = texts[:train_end]
    val_texts = texts[train_end:val_end]
    test_texts = texts[val_end:]

    return train_texts, val_texts, test_texts


def create_sample_dataset(
    num_samples: int = 100,
    min_length: int = 10,
    max_length: int = 100,
    seed: int = 42,
) -> List[str]:
    """
    Create a sample dataset for testing.

    Generates synthetic text samples suitable for testing
    the data pipeline and model.

    Args:
        num_samples: Number of samples to generate
        min_length: Minimum length per sample (words)
        max_length: Maximum length per sample (words)
        seed: Random seed

    Returns:
        List of sample texts
    """
    import random

    random.seed(seed)

    # Simple vocabulary of words
    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "hello", "world", "this", "is", "a", "test", "of", "language",
        "model", "training", "data", "pipeline", "khatri", "voice",
        " Neural", "network", "transformer", "attention", "embedding",
        "hello", "again", "testing", "one", "two", "three", "four",
    ]

    samples: List[str] = []
    for _ in range(num_samples):
        length = random.randint(min_length, max_length)
        sample = " ".join(random.choice(words) for _ in range(length))
        samples.append(sample)

    return samples


def create_tiny_dataset() -> List[str]:
    """
    Create a tiny dataset for overfit testing.

    This creates a very small, repetitive dataset that the model
    should be able to memorize easily, useful for verifying
    the training pipeline works.

    Returns:
        List of sample texts
    """
    return [
        "hello world hello world hello world",
        "testing one two three testing one two three",
        "hello hello hello world world world",
        "one two three four five six seven eight",
        "the quick brown fox jumps over the lazy dog",
        "hello world testing one two three",
        "hello hello testing testing hello testing",
        "world world world hello hello hello",
        "one two one two one two one two",
        "hello world hello world hello world",
    ] * 5  # Repeat to have 50 samples
