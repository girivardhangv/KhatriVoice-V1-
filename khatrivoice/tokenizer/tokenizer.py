"""
KhatriVoice Tokenizer.

A character-level and word-level tokenizer designed for the Khatri language.
Supports Unicode, special tokens, and vocabulary training from corpus.

This tokenizer is designed to be trained from scratch on Khatri text
and does not assume any specific linguistic properties.
"""

import re
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from collections import Counter

from khatrivoice.tokenizer.vocabulary import Vocabulary


class KhatriTokenizer:
    """
    Tokenizer for KhatriVoice language model.

    This is a simple but effective tokenizer that supports:
    - Character-level tokenization
    - Word-level tokenization with subword fallback
    - Special tokens (BOS, EOS, PAD, UNK)
    - Unicode support
    - Vocabulary training from corpus

    The tokenizer does NOT assume Hindi-specific processing.
    Khatri text is processed as-is without normalization to other scripts.

    Attributes:
        vocab: Vocabulary instance
        lowercase: Whether to lowercase text before tokenization
        max_token_length: Maximum length for a single token
        min_freq: Minimum frequency for a token to be included in vocabulary
    """

    def __init__(
        self,
        vocab: Optional[Vocabulary] = None,
        lowercase: bool = False,
        max_token_length: int = 50,
        min_freq: int = 1,
    ) -> None:
        """
        Initialize the tokenizer.

        Args:
            vocab: Vocabulary instance (created if None)
            lowercase: Whether to lowercase before tokenization
            max_token_length: Maximum length for a single token
            min_freq: Minimum frequency for vocabulary inclusion during training
        """
        self.vocab = vocab if vocab is not None else Vocabulary()
        self.lowercase = lowercase
        self.max_token_length = max_token_length
        self.min_freq = min_freq

        # Regular expressions for tokenization
        # Matches words (including Unicode letters), numbers, and punctuation
        self._word_pattern = re.compile(
            r"""(
                \d+ |                          # Numbers
                [^\W\d_]+ |                    # Words (Unicode letters)
                [^\s\w]                        # Punctuation and symbols
            )""",
            re.UNICODE | re.VERBOSE,
        )

        # Pattern for whitespace
        self._whitespace_pattern = re.compile(r"\s+")

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab.vocab_size

    @property
    def bos_id(self) -> int:
        """Get BOS token ID."""
        return self.vocab.bos_id

    @property
    def eos_id(self) -> int:
        """Get EOS token ID."""
        return self.vocab.eos_id

    @property
    def pad_id(self) -> int:
        """Get PAD token ID."""
        return self.vocab.pad_id

    @property
    def unk_id(self) -> int:
        """Get UNK token ID."""
        return self.vocab.unk_id

    def _preprocess(self, text: str) -> str:
        """
        Preprocess text before tokenization.

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        # Apply lowercasing if enabled
        if self.lowercase:
            text = text.lower()

        # Normalize whitespace
        text = self._whitespace_pattern.sub(" ", text)

        return text.strip()

    def _tokenize_word_level(self, text: str) -> List[str]:
        """
        Tokenize text at word level.

        Args:
            text: Input text

        Returns:
            List of word tokens
        """
        matches = self._word_pattern.findall(text)
        return matches

    def _tokenize_char_level(self, text: str) -> List[str]:
        """
        Tokenize text at character level.

        This is useful for languages where word boundaries are unclear
        or for handling unknown words.

        Args:
            text: Input text

        Returns:
            List of character tokens
        """
        return list(text)

    def _tokenize_hybrid(self, text: str) -> List[str]:
        """
        Hybrid tokenization: word-level for known words, character-level for unknown.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        words = self._tokenize_word_level(text)
        tokens: List[str] = []

        for word in words:
            if word in self.vocab:
                tokens.append(word)
            else:
                # Fall back to character-level for unknown words
                tokens.extend(self._tokenize_char_level(word))

        return tokens

    def tokenize(self, text: str, mode: str = "hybrid") -> List[str]:
        """
        Tokenize text into a list of tokens.

        Args:
            text: Input text to tokenize
            mode: Tokenization mode ('word', 'char', 'hybrid')

        Returns:
            List of tokens
        """
        text = self._preprocess(text)

        if mode == "word":
            return self._tokenize_word_level(text)
        elif mode == "char":
            return self._tokenize_char_level(text)
        elif mode == "hybrid":
            return self._tokenize_hybrid(text)
        else:
            raise ValueError(f"Unknown tokenization mode: {mode}")

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        mode: str = "hybrid",
    ) -> List[int]:
        """
        Encode text into a list of token IDs.

        Args:
            text: Input text to encode
            add_bos: Whether to add BOS token at the beginning
            add_eos: Whether to add EOS token at the end
            mode: Tokenization mode ('word', 'char', 'hybrid')

        Returns:
            List of token IDs
        """
        tokens = self.tokenize(text, mode=mode)

        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)

        for token in tokens:
            ids.append(self.vocab.get_id(token))

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization: bool = True,
    ) -> str:
        """
        Decode token IDs back into text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            clean_up_tokenization: Whether to clean up tokenization artifacts

        Returns:
            Decoded text
        """
        tokens: List[str] = []
        for token_id in token_ids:
            token = self.vocab.get_token(token_id)

            # Skip special tokens if requested
            if skip_special_tokens and token in self.vocab.special_tokens:
                continue

            tokens.append(token)

        # Join tokens
        if clean_up_tokenization:
            text = self._detokenize(tokens)
        else:
            text = "".join(tokens)

        return text

    def _detokenize(self, tokens: List[str]) -> str:
        """
        Detokenize a list of tokens into text.

        Attempts to reconstruct readable text from tokens.

        Args:
            tokens: List of tokens

        Returns:
            Detokenized text
        """
        if not tokens:
            return ""

        result: List[str] = []
        prev_token: Optional[str] = None

        for token in tokens:
            # Add space before word tokens (not punctuation/symbols)
            if prev_token is not None:
                # Check if we should add a space
                if not self._is_punctuation(token) and not self._is_punctuation(prev_token):
                    # Add space between word-like tokens
                    if not self._is_char_token(prev_token) or not self._is_char_token(token):
                        result.append(" ")

            result.append(token)
            prev_token = token

        return "".join(result)

    def _is_punctuation(self, token: str) -> bool:
        """Check if a token is punctuation."""
        return bool(re.match(r"^[^\s\w]$", token))

    def _is_char_token(self, token: str) -> bool:
        """Check if a token is a single character."""
        return len(token) == 1 and not token.isspace()

    def encode_batch(
        self,
        texts: List[str],
        add_bos: bool = False,
        add_eos: bool = False,
        mode: str = "hybrid",
    ) -> List[List[int]]:
        """
        Encode multiple texts into token IDs.

        Args:
            texts: List of texts to encode
            add_bos: Whether to add BOS token
            add_eos: Whether to add EOS token
            mode: Tokenization mode

        Returns:
            List of token ID lists
        """
        return [
            self.encode(text, add_bos=add_bos, add_eos=add_eos, mode=mode)
            for text in texts
        ]

    def decode_batch(
        self,
        batch_token_ids: List[List[int]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """
        Decode multiple token ID lists into texts.

        Args:
            batch_token_ids: List of token ID lists
            skip_special_tokens: Whether to skip special tokens

        Returns:
            List of decoded texts
        """
        return [
            self.decode(ids, skip_special_tokens=skip_special_tokens)
            for ids in batch_token_ids
        ]

    def train(
        self,
        corpus: List[str],
        mode: str = "word",
        vocab_size: Optional[int] = None,
    ) -> None:
        """
        Train the tokenizer vocabulary from a corpus.

        Args:
            corpus: List of text samples
            mode: Tokenization mode for vocabulary building
            vocab_size: Target vocabulary size (None for no limit)
        """
        # Count token frequencies
        token_counts: Counter = Counter()

        for text in corpus:
            tokens = self.tokenize(text, mode=mode)
            token_counts.update(tokens)

        # Filter by minimum frequency
        filtered_tokens = [
            (token, count)
            for token, count in token_counts.items()
            if count >= self.min_freq and len(token) <= self.max_token_length
        ]

        # Sort by frequency
        filtered_tokens.sort(key=lambda x: (-x[1], x[0]))

        # Limit vocabulary size if specified
        # Reserve space for special tokens (4 special tokens)
        if vocab_size is not None:
            max_new_tokens = vocab_size - 4
            filtered_tokens = filtered_tokens[:max_new_tokens]

        # Add tokens to vocabulary
        for token, _ in filtered_tokens:
            self.vocab._add_token(token)

    def save(self, directory: str | Path) -> None:
        """
        Save the tokenizer to a directory.

        Args:
            directory: Directory path to save to
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save vocabulary
        vocab_path = directory / "vocabulary.json"
        self.vocab.save(vocab_path)

        # Save tokenizer config
        import json

        config = {
            "lowercase": self.lowercase,
            "max_token_length": self.max_token_length,
            "min_freq": self.min_freq,
        }

        config_path = directory / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, directory: str | Path) -> "KhatriTokenizer":
        """
        Load a tokenizer from a directory.

        Args:
            directory: Directory path to load from

        Returns:
            KhatriTokenizer instance
        """
        directory = Path(directory)

        # Load vocabulary
        vocab_path = directory / "vocabulary.json"
        vocab = Vocabulary.load(vocab_path)

        # Load config
        import json

        config_path = directory / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        return cls(
            vocab=vocab,
            lowercase=config.get("lowercase", False),
            max_token_length=config.get("max_token_length", 50),
            min_freq=config.get("min_freq", 1),
        )

    def build_inputs_with_special_tokens(
        self,
        token_ids: List[int],
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        """
        Add special tokens to a token ID sequence.

        Args:
            token_ids: List of token IDs
            add_bos: Whether to add BOS token
            add_eos: Whether to add EOS token

        Returns:
            Token IDs with special tokens
        """
        result: List[int] = []
        if add_bos:
            result.append(self.bos_id)
        result.extend(token_ids)
        if add_eos:
            result.append(self.eos_id)
        return result

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"KhatriTokenizer(vocab_size={self.vocab_size}, "
            f"lowercase={self.lowercase})"
        )
