"""
Data preprocessing utilities for KhatriVoice.
"""

import re
import random
from pathlib import Path
from typing import List, Tuple, Optional


def load_text_file(
    path: Path,
    encoding: str = "utf-8",
    clean: bool = True,
) -> str:
    """
    Load text from a single file.

    Args:
        path: Path to text file
        encoding: File encoding
        clean: Whether to clean the text

    Returns:
        Text content as string
    """
    path = Path(path)
    with open(path, "r", encoding=encoding) as f:
        text = f.read()

    if clean:
        text = clean_text(text)

    return text


def load_text_files(
    directory: Path,
    pattern: str = "*.txt",
    encoding: str = "utf-8",
    clean: bool = True,
) -> List[str]:
    """
    Load all text files from a directory.

    Args:
        directory: Directory containing text files
        pattern: Glob pattern for file selection
        encoding: File encoding
        clean: Whether to clean the text

    Returns:
        List of text contents
    """
    directory = Path(directory)
    texts = []
    for path in directory.glob(pattern):
        text = load_text_file(path, encoding=encoding, clean=clean)
        texts.append(text)
    return texts


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing.

    Args:
        text: Input text

    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences.

    Args:
        text: Input text

    Returns:
        List of sentences
    """
    # Simple sentence splitting
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def split_train_val_test(
    texts: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split texts into train, validation, and test sets.

    Args:
        texts: List of text samples
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_texts, val_texts, test_texts)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    random.seed(seed)
    texts = texts.copy()
    random.shuffle(texts)

    n = len(texts)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_texts = texts[:train_end]
    val_texts = texts[train_end:val_end]
    test_texts = texts[val_end:]

    return train_texts, val_texts, test_texts


def create_tiny_dataset() -> List[str]:
    """
    Create a tiny dataset for testing.

    Returns:
        List of text samples
    """
    return [
        "Hello world!",
        "This is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating.",
        "Natural language processing enables computers to understand human language.",
        "Transformers have revolutionized NLP.",
        "Attention is all you need.",
        "Deep learning models can learn complex patterns.",
        "Training neural networks requires careful optimization.",
        "Gradient descent is a fundamental optimization algorithm.",
    ]


def parse_conversation_file(path: Path, encoding: str = "utf-8") -> List[dict]:
    """
    Parse a conversation file with User/Assistant pairs.

    Args:
        path: Path to conversation file
        encoding: File encoding

    Returns:
        List of conversation dictionaries
    """
    text = load_text_file(path, encoding=encoding, clean=False)
    lines = text.strip().split("\n")

    conversations = []
    current_conv = {"user": "", "assistant": ""}

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("User:"):
            if current_conv["user"] or current_conv["assistant"]:
                conversations.append(current_conv)
                current_conv = {"user": "", "assistant": ""}
            current_conv["user"] = line[6:].strip()
        elif line.startswith("Assistant:"):
            current_conv["assistant"] = line[11:].strip()

    # Don't forget the last conversation
    if current_conv["user"] or current_conv["assistant"]:
        conversations.append(current_conv)

    return conversations


def conversations_to_training_data(conversations: List[dict]) -> List[str]:
    """
    Convert conversations to training text samples.

    Args:
        conversations: List of conversation dictionaries

    Returns:
        List of training text samples
    """
    texts = []
    for conv in conversations:
        text = f"User: {conv['user']} Assistant: {conv['assistant']}"
        texts.append(text)
    return texts
