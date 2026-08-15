"""
Tokenizer training utilities for KhatriVoice.

This module provides utilities for training the tokenizer vocabulary
from a Khatri text corpus.
"""

from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from khatrivoice.tokenizer.tokenizer import KhatriTokenizer


class TokenizerTrainer:
    """
    Trainer for KhatriVoice tokenizer.

    Trains the tokenizer vocabulary from a text corpus and saves
    the trained tokenizer for use in training the language model.
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        min_freq: int = 1,
        lowercase: bool = False,
        max_token_length: int = 50,
    ) -> None:
        """
        Initialize the tokenizer trainer.

        Args:
            vocab_size: Target vocabulary size
            min_freq: Minimum frequency for token inclusion
            lowercase: Whether to lowercase text before tokenization
            max_token_length: Maximum length for a single token
        """
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.lowercase = lowercase
        self.max_token_length = max_token_length

    def train_from_corpus(
        self,
        texts: List[str],
        mode: str = "word",
        show_progress: bool = True,
    ) -> KhatriTokenizer:
        """
        Train tokenizer from a list of texts.

        Args:
            texts: List of text samples
            mode: Tokenization mode ('word', 'char', 'hybrid')
            show_progress: Whether to show progress bar

        Returns:
            Trained KhatriTokenizer instance
        """
        # Create tokenizer
        tokenizer = KhatriTokenizer(
            lowercase=self.lowercase,
            max_token_length=self.max_token_length,
            min_freq=self.min_freq,
        )

        # Train vocabulary
        if show_progress:
            print(f"Processing {len(texts):,} texts...")

        tokenizer.train(texts, mode=mode, vocab_size=self.vocab_size)

        print(f"Vocabulary trained: {tokenizer.vocab_size:,} tokens")

        return tokenizer

    def train_from_file(
        self,
        filepath: str | Path,
        encoding: str = "utf-8",
        mode: str = "word",
        show_progress: bool = True,
    ) -> KhatriTokenizer:
        """
        Train tokenizer from a text file.

        Args:
            filepath: Path to text file
            encoding: File encoding
            mode: Tokenization mode
            show_progress: Whether to show progress bar

        Returns:
            Trained KhatriTokenizer instance
        """
        filepath = Path(filepath)

        print(f"Loading corpus from {filepath}...")
        with open(filepath, "r", encoding=encoding) as f:
            lines = f.readlines()

        # Remove empty lines and strip whitespace
        texts = [line.strip() for line in lines if line.strip()]

        return self.train_from_corpus(texts, mode=mode, show_progress=show_progress)

    def train_from_directory(
        self,
        directory: str | Path,
        pattern: str = "*.txt",
        encoding: str = "utf-8",
        mode: str = "word",
        show_progress: bool = True,
    ) -> KhatriTokenizer:
        """
        Train tokenizer from multiple text files in a directory.

        Args:
            directory: Directory containing text files
            pattern: Glob pattern for text files
            encoding: File encoding
            mode: Tokenization mode
            show_progress: Whether to show progress bar

        Returns:
            Trained KhatriTokenizer instance
        """
        directory = Path(directory)
        files = list(directory.glob(pattern))

        if not files:
            raise FileNotFoundError(f"No files found matching {pattern} in {directory}")

        print(f"Found {len(files)} files in {directory}")

        all_texts: List[str] = []

        # Use tqdm for progress if available
        if show_progress:
            files_iter = tqdm(files, desc="Reading files")
        else:
            files_iter = files

        for filepath in files_iter:
            with open(filepath, "r", encoding=encoding) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_texts.append(line)

        return self.train_from_corpus(all_texts, mode=mode, show_progress=False)


def create_tiny_test_tokenizer() -> KhatriTokenizer:
    """
    Create a tiny tokenizer for testing.

    This creates a minimal tokenizer with a small vocabulary
    suitable for CPU testing and debugging.

    Returns:
        KhatriTokenizer with a tiny vocabulary
    """
    # Create a simple synthetic vocabulary for testing
    test_corpus = [
        "hello world",
        "this is a test",
        "hello again",
        "testing one two three",
        "hello hello hello",
        "the quick brown fox",
        "jumped over the lazy dog",
        "this is another test",
        "hello world test",
        "final test hello",
    ]

    trainer = TokenizerTrainer(
        vocab_size=100,
        min_freq=1,
        lowercase=True,
    )

    tokenizer = trainer.train_from_corpus(test_corpus, show_progress=False)
    return tokenizer


def train_and_save_tokenizer(
    corpus_path: str | Path,
    output_path: str | Path,
    vocab_size: int = 10000,
    mode: str = "word",
) -> KhatriTokenizer:
    """
    Train and save a tokenizer from a corpus file.

    This is a convenience function for training tokenizers.

    Args:
        corpus_path: Path to corpus file
        output_path: Directory to save tokenizer
        vocab_size: Target vocabulary size
        mode: Tokenization mode

    Returns:
        Trained KhatriTokenizer
    """
    trainer = TokenizerTrainer(vocab_size=vocab_size)
    tokenizer = trainer.train_from_file(corpus_path, mode=mode)

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_path)

    print(f"Tokenizer saved to {output_path}")
    return tokenizer
