"""
KhatriVoice Tokenizer.

A simple word-level tokenizer with configurable vocabulary size.
"""

import json
import re
from pathlib import Path
from typing import List, Optional, Dict, Set
from collections import Counter


class KhatriTokenizer:
    """
    Simple word-level tokenizer for KhatriVoice.

    Supports word-level and character-level tokenization modes.

    Args:
        lowercase: Whether to lowercase text before tokenization
        unk_token: Token for unknown words
        pad_token: Padding token
        bos_token: Beginning of sequence token
        eos_token: End of sequence token
    """

    # Special tokens
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"

    def __init__(
        self,
        lowercase: bool = True,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
    ):
        self.lowercase = lowercase
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        # Vocabulary
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}

        # Special token IDs
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

        # Token counts (for vocabulary building)
        self._token_counts: Counter = Counter()

        # Trained flag
        self._trained = False

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return len(self.vocab)

    def train(
        self,
        texts: List[str],
        vocab_size: int = 8000,
        min_freq: int = 1,
        mode: str = "word",
    ) -> None:
        """
        Train the tokenizer on a corpus.

        Args:
            texts: List of text strings to train on
            vocab_size: Maximum vocabulary size
            min_freq: Minimum frequency for a token to be included
            mode: Tokenization mode ('word' or 'char')
        """
        # Count tokens
        self._token_counts = Counter()
        for text in texts:
            tokens = self._tokenize(text, mode=mode)
            self._token_counts.update(tokens)

        # Build vocabulary
        self._build_vocab(vocab_size, min_freq)
        self._trained = True

    def _tokenize(self, text: str, mode: str = "word") -> List[str]:
        """
        Tokenize text into tokens.

        Args:
            text: Input text string
            mode: Tokenization mode ('word' or 'char')

        Returns:
            List of tokens
        """
        if self.lowercase:
            text = text.lower()

        if mode == "char":
            return list(text)
        else:
            # Word-level tokenization with punctuation
            pattern = re.compile(
                r"""(
                    \d+ |
                    [^\W\d_]+ |
                    [^\s\w]
                )""",
                re.UNICODE | re.VERBOSE,
            )
            return pattern.findall(text)

    def _build_vocab(self, vocab_size: int, min_freq: int) -> None:
        """Build vocabulary from token counts."""
        # Start with special tokens
        self.vocab = {
            self.pad_token: self.pad_id,
            self.unk_token: self.unk_id,
            self.bos_token: self.bos_id,
            self.eos_token: self.eos_id,
        }

        # Add most frequent tokens
        next_id = 4
        for token, count in self._token_counts.most_common():
            if count < min_freq:
                break
            if token not in self.vocab:
                self.vocab[token] = next_id
                next_id += 1
                if next_id >= vocab_size:
                    break

        # Build inverse vocabulary
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text: str, mode: str = "word") -> List[str]:
        """
        Tokenize text.

        Args:
            text: Input text string
            mode: Tokenization mode

        Returns:
            List of tokens
        """
        return self._tokenize(text, mode)

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text string
            add_bos: Whether to add BOS token
            add_eos: Whether to add EOS token

        Returns:
            List of token IDs
        """
        tokens = self._tokenize(text)

        ids = []
        if add_bos:
            ids.append(self.bos_id)

        for token in tokens:
            ids.append(self.vocab.get(token, self.unk_id))

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.

        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text string
        """
        tokens = []
        for id_ in ids:
            if id_ in self.inverse_vocab:
                token = self.inverse_vocab[id_]
                if skip_special_tokens and token in [
                    self.pad_token,
                    self.unk_token,
                    self.bos_token,
                    self.eos_token,
                ]:
                    continue
                tokens.append(token)
            else:
                if not skip_special_tokens:
                    tokens.append(self.unk_token)

        return " ".join(tokens)

    def save(self, path: str) -> None:
        """
        Save tokenizer vocabulary to directory.

        Args:
            path: Directory path to save tokenizer
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save vocabulary
        vocab_file = path / "vocabulary.json"
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        # Save config
        config = {
            "lowercase": self.lowercase,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "vocab_size": self.vocab_size,
        }
        config_file = path / "config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "KhatriTokenizer":
        """
        Load tokenizer from directory.

        Args:
            path: Directory path containing tokenizer files

        Returns:
            KhatriTokenizer instance
        """
        path = Path(path)

        # Load config
        config_file = path / "config.json"
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        tokenizer = cls(
            lowercase=config["lowercase"],
            unk_token=config["unk_token"],
            pad_token=config["pad_token"],
            bos_token=config["bos_token"],
            eos_token=config["eos_token"],
        )

        # Load vocabulary
        vocab_file = path / "vocabulary.json"
        with open(vocab_file, "r", encoding="utf-8") as f:
            tokenizer.vocab = json.load(f)

        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer._trained = True

        return tokenizer

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size

    def __repr__(self) -> str:
        """Return string representation."""
        return f"KhatriTokenizer(vocab_size={self.vocab_size}, lowercase={self.lowercase})"
