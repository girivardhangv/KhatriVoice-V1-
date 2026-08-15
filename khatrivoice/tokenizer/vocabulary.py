"""
Vocabulary management for KhatriVoice tokenizer.

This module handles the vocabulary for tokenization, including
special tokens and token-to-ID mappings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import json
from pathlib import Path


@dataclass
class Vocabulary:
    """
    Vocabulary for KhatriVoice tokenizer.

    Manages token-to-ID and ID-to-token mappings with support for
    special tokens (BOS, EOS, PAD, UNK).

    Attributes:
        token_to_id: Mapping from tokens to their integer IDs
        special_tokens: Set of special token strings
        bos_token: Beginning of sequence token
        eos_token: End of sequence token
        pad_token: Padding token
        unk_token: Unknown token
    """

    # Token mappings
    token_to_id: Dict[str, int] = field(default_factory=dict)
    id_to_token: Dict[int, str] = field(default_factory=dict, init=False)

    # Special tokens
    special_tokens: Set[str] = field(default_factory=set)
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"

    def __post_init__(self) -> None:
        """Initialize the vocabulary and special tokens."""
        self._build_reverse_mapping()
        self._add_special_tokens()

    def _build_reverse_mapping(self) -> None:
        """Build the ID-to-token mapping from token-to-ID mapping."""
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def _add_special_tokens(self) -> None:
        """Add special tokens to the vocabulary if not present."""
        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token,
        ]
        for token in special_tokens:
            self.special_tokens.add(token)
            if token not in self.token_to_id:
                self._add_token(token)

    def _add_token(self, token: str) -> int:
        """
        Add a token to the vocabulary.

        Args:
            token: Token string to add

        Returns:
            Token ID
        """
        if token not in self.token_to_id:
            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
        return self.token_to_id[token]

    def add_tokens(self, tokens: List[str]) -> None:
        """
        Add multiple tokens to the vocabulary.

        Args:
            tokens: List of token strings to add
        """
        for token in tokens:
            self._add_token(token)

    def get_id(self, token: str) -> int:
        """
        Get the ID for a token.

        Args:
            token: Token string

        Returns:
            Token ID (returns UNK ID if token not found)
        """
        if token in self.token_to_id:
            return self.token_to_id[token]
        return self.token_to_id[self.unk_token]

    def get_token(self, token_id: int) -> str:
        """
        Get the token for an ID.

        Args:
            token_id: Token ID

        Returns:
            Token string (returns UNK token if ID not found)
        """
        if token_id in self.id_to_token:
            return self.id_to_token[token_id]
        return self.unk_token

    @property
    def bos_id(self) -> int:
        """Get the ID for the BOS token."""
        return self.token_to_id[self.bos_token]

    @property
    def eos_id(self) -> int:
        """Get the ID for the EOS token."""
        return self.token_to_id[self.eos_token]

    @property
    def pad_id(self) -> int:
        """Get the ID for the PAD token."""
        return self.token_to_id[self.pad_token]

    @property
    def unk_id(self) -> int:
        """Get the ID for the UNK token."""
        return self.token_to_id[self.unk_token]

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        return len(self.token_to_id)

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size

    def __contains__(self, token: str) -> bool:
        """Check if a token is in the vocabulary."""
        return token in self.token_to_id

    def save(self, path: str | Path) -> None:
        """
        Save vocabulary to a JSON file.

        Args:
            path: Path to save the vocabulary
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "token_to_id": self.token_to_id,
            "special_tokens": list(self.special_tokens),
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        """
        Load vocabulary from a JSON file.

        Args:
            path: Path to load the vocabulary from

        Returns:
            Vocabulary instance
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vocab = cls(
            token_to_id=data["token_to_id"],
            special_tokens=set(data["special_tokens"]),
            bos_token=data["bos_token"],
            eos_token=data["eos_token"],
            pad_token=data["pad_token"],
            unk_token=data["unk_token"],
        )
        return vocab

    def get_special_tokens_mask(self, token_ids: List[int]) -> List[bool]:
        """
        Get a mask indicating which tokens are special tokens.

        Args:
            token_ids: List of token IDs

        Returns:
            List of booleans (True for special tokens)
        """
        return [self.id_to_token.get(tid, "") in self.special_tokens for tid in token_ids]

    def __repr__(self) -> str:
        """Return string representation."""
        return f"Vocabulary(size={self.vocab_size}, special_tokens={len(self.special_tokens)})"
