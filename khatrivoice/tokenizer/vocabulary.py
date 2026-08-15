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
    # Conversational special tokens - MUST match training data format
    user_token: str = "<user>"
    assistant_token: str = "<|assistant>"
    end_token: str = "<|end|>"

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
            self.user_token,
            self.assistant_token,
            self.end_token,
        ]
        for token in special_tokens:
            self.special_tokens.add(token)
            if token not in self.token_to_id:
                self._add_token(token)

    def _add_token(self, token: str) -> int:
        """Add a token to the vocabulary and return its ID."""
        if token not in self.token_to_id:
            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
        return self.token_to_id[token]

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.token_to_id)

    def __contains__(self, token: str) -> bool:
        """Check if token is in vocabulary."""
        return token in self.token_to_id

    def get_id(self, token: str) -> int:
        """Get ID for a token, returning UNK ID if not found."""
        return self.token_to_id.get(token, self.token_to_id.get(self.unk_token, 0))

    def get_token(self, token_id: int) -> str:
        """Get token for an ID, returning UNK token if not found."""
        return self.id_to_token.get(token_id, self.unk_token)

    @property
    def pad_id(self) -> int:
        """Get padding token ID."""
        return self.get_id(self.pad_token)

    @property
    def unk_id(self) -> int:
        """Get unknown token ID."""
        return self.get_id(self.unk_token)

    @property
    def bos_id(self) -> int:
        """Get beginning-of-sequence token ID."""
        return self.get_id(self.bos_token)

    @property
    def eos_id(self) -> int:
        """Get end-of-sequence token ID."""
        return self.get_id(self.eos_token)

    @property
    def user_id(self) -> int:
        """Get user token ID."""
        return self.get_id(self.user_token)

    @property
    def assistant_id(self) -> int:
        """Get assistant token ID."""
        return self.get_id(self.assistant_token)

    @property
    def end_id(self) -> int:
        """Get end token ID."""
        return self.get_id(self.end_token)

    def save(self, path: Path) -> None:
        """Save vocabulary to JSON file."""
        data = {
            "token_to_id": self.token_to_id,
            "special_tokens": list(self.special_tokens),
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "user_token": self.user_token,
            "assistant_token": self.assistant_token,
            "end_token": self.end_token,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        """Load vocabulary from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vocab = cls(token_to_id=data["token_to_id"])
        vocab.special_tokens = set(data.get("special_tokens", []))
        vocab.bos_token = data.get("bos_token", "<s>")
        vocab.eos_token = data.get("eos_token", "</s>")
        vocab.pad_token = data.get("pad_token", "<pad>")
        vocab.unk_token = data.get("unk_token", "<unk>")
        vocab.user_token = data.get("user_token", "<user>")
        vocab.assistant_token = data.get("assistant_token", "<|assistant>")
        vocab.end_token = data.get("end_token", "<|end|>")

        return vocab

    def __repr__(self) -> str:
        """Return string representation."""
        return f"Vocabulary(size={len(self)}, special_tokens={len(self.special_tokens)})"
