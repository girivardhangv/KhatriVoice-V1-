"""
Text preprocessing utilities for KhatriVoice.

This module provides functions for preparing text data before tokenization
and dataset creation.
"""

import re
from typing import List, Optional, Dict
from pathlib import Path


# Special tokens for conversation formatting
# These contain invisible Unicode markers and must match vocabulary.py exactly
USER_TOKEN = ""  # Contains special Unicode LTR markers
ASSISTANT_TOKEN = ""  # Contains special Unicode LTR markers
END_TOKEN = "<|end|>"


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
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) >= 1e-6:
        # If ratios don't sum to 1, normalize them
        if total_ratio > 0:
            train_ratio /= total_ratio
            val_ratio /= total_ratio
            test_ratio /= total_ratio

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

    # Ensure we always have training data
    if not train_texts and texts:
        train_texts = texts[:1]

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


# =============================================================================
# Conversation Parsing Utilities
# =============================================================================

def parse_conversation_line(line: str) -> Optional[Dict[str, str]]:
    """
    Parse a single line containing a User/AI conversation pair.

    Supports formats:
    - "User: <question> AI: <response>"
    - "user: <question> ai: <response>" (case-insensitive)

    Args:
        line: Line to parse

    Returns:
        Dictionary with 'user' and 'assistant' keys, or None if parsing fails
    """
    import re
    # Case-insensitive matching for User:/AI: markers
    pattern = r"(?i)User:\s*(.+?)\s*(?:AI|Assistant):\s*(.+)$"
    match = re.match(pattern, line.strip())

    if match:
        user_text = match.group(1).strip()
        assistant_text = match.group(2).strip()
        if user_text and assistant_text:
            return {"user": user_text, "assistant": assistant_text}
    return None


def format_conversation(user_text: str, assistant_text: str) -> str:
    """
    Format a conversation turn with proper special tokens.

    Format: <user>\n{user_text}\n<assistant>\n{assistant_text}\n<|end|>

    Args:
        user_text: User's message
        assistant_text: Assistant's response

    Returns:
        Formatted conversation string
    """
    user_text = user_text.strip()
    assistant_text = assistant_text.strip()

    # Use module-level constants defined at top of file
    return f"{USER_TOKEN}\n{user_text}\n{ASSISTANT_TOKEN}\n{assistant_text}\n{END_TOKEN}"


def parse_conversation_file(
    filepath: str | Path,
    skip_malformed: bool = True,
    remove_duplicates: bool = False,
    format: str = "auto",
) -> List[Dict[str, str]]:
    """
    Parse a file containing User/AI conversation pairs.

    Supports multiple formats:
    - "inline": User: <text> AI: <text> on same line
    - "multiline": User/AI on separate lines (your format)
    - "auto": Auto-detect format

    Args:
        filepath: Path to the conversation file
        skip_malformed: Whether to skip lines that can't be parsed
        remove_duplicates: Whether to remove duplicate conversations
        format: File format ("inline", "multiline", "auto")

    Returns:
        List of dictionaries with 'user' and 'assistant' keys
    """
    filepath = Path(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Auto-detect format if needed
    if format == "auto":
        # Check if any line has both User: and AI: on same line
        has_inline = any("User:" in line and "AI:" in line for line in lines[:100])
        format = "inline" if has_inline else "multiline"

    conversations = []
    seen = set()

    if format == "multiline":
        # Parse multiline format (User on one line, AI on next)
        conversations = _parse_multiline_conversations(lines, skip_malformed)
    else:
        # Parse inline format
        for line in lines:
            line = line.strip()
            if not line:
                continue

            parsed = parse_conversation_line(line)
            if parsed:
                conversations.append(parsed)
            elif not skip_malformed:
                raise ValueError(f"Could not parse line: {line}")

    # Remove duplicates if requested
    if remove_duplicates:
        unique_conversations = []
        for conv in conversations:
            key = (conv["user"], conv["assistant"])
            if key not in seen:
                seen.add(key)
                unique_conversations.append(conv)
        conversations = unique_conversations

    return conversations


def _parse_multiline_conversations(
    lines: List[str],
    skip_malformed: bool = True,
) -> List[Dict[str, str]]:
    """
    Parse conversations where User and AI are on separate lines.

    Format:
        User: What is Python?
        AI: Python is a programming language.
        User: What is Java?
        AI: Java is another programming language.

    Args:
        lines: List of lines from file
        skip_malformed: Whether to skip malformed pairs

    Returns:
        List of conversation dictionaries
    """
    import re
    conversations = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Check for User: line
        if re.match(r"(?i)^(User|Human):\s*(.+)$", line):
            user_match = re.match(r"(?i)^(User|Human):\s*(.+)$", line)
            user_text = user_match.group(2).strip()

            # Look for AI: response on next line(s)
            i += 1
            assistant_lines = []

            while i < len(lines):
                next_line = lines[i].strip()

                # Check if it's a new User: line
                if re.match(r"(?i)^(User|Human):", next_line):
                    break

                # Check if it's an AI: line
                ai_match = re.match(r"(?i)^(AI|Assistant):\s*(.+)$", next_line)
                if ai_match:
                    assistant_lines.append(ai_match.group(2).strip())
                    i += 1
                    # Continue collecting multi-line responses
                    while i < len(lines):
                        cont_line = lines[i].strip()
                        # Stop at next User: or AI: marker
                        if re.match(r"(?i)^(User|Human|AI|Assistant):", cont_line):
                            break
                        if cont_line:  # Non-empty line
                            assistant_lines.append(cont_line)
                        i += 1

            if user_text and assistant_lines:
                assistant_text = " ".join(assistant_lines)
                conversations.append({
                    "user": user_text,
                    "assistant": assistant_text,
                })
            elif not skip_malformed:
                raise ValueError(f"Malformed conversation at line with: {user_text[:50]}")
        else:
            i += 1

    return conversations


def conversations_to_training_data(
    conversations: List[Dict[str, str]],
    shuffle: bool = True,
    seed: int = 42,
) -> List[str]:
    """
    Convert parsed conversations to formatted training data.

    Args:
        conversations: List of dicts with 'user' and 'assistant' keys
        shuffle: Whether to shuffle the data
        seed: Random seed for shuffling

    Returns:
        List of formatted conversation strings
    """
    import random

    formatted = []
    for conv in conversations:
        text = format_conversation(conv["user"], conv["assistant"])
        formatted.append(text)

    if shuffle:
        random.seed(seed)
        random.shuffle(formatted)

    return formatted
