"""
Conversation-aware dataset for KhatriVoice.

This module provides a dataset that properly handles User/AI conversations
with label masking so the model only learns to predict assistant responses.
"""

import re
import random
from typing import Dict, List, Optional, Tuple
import torch
from torch import Tensor

from khatrivoice.tokenizer.tokenizer import KhatriTokenizer


class ConversationDataset(torch.utils.data.Dataset):
    """
    Dataset for conversational training with proper label masking.

    This dataset:
    1. Parses User/AI conversation pairs
    2. Formats them with special tokens
    3. Masks user tokens in labels (-100) so loss is only computed on assistant responses

    The format is:
        <user>\n{user_text}\n<assistant>\n{assistant_text}\n<|end|>

    Labels for user tokens are set to -100 (ignored by loss).
    """

    # Token markers (must match vocabulary.py)
    USER_MARKER = ""
    ASSISTANT_MARKER = ""
    END_MARKER = "<|end|>"

    def __init__(
        self,
        tokenizer: KhatriTokenizer,
        texts: List[str],
        max_length: int = 512,
        mask_user_tokens: bool = True,
        stride: Optional[int] = None,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> None:
        """
        Initialize conversation dataset.

        Args:
            tokenizer: Tokenizer instance
            texts: List of conversation strings (User: ... AI: ... format)
            max_length: Maximum sequence length
            mask_user_tokens: Whether to mask user tokens in labels
            stride: Stride for sliding window (default: max_length // 2)
            add_bos: Add BOS token
            add_eos: Add EOS token
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_user_tokens = mask_user_tokens
        self.stride = stride if stride is not None else max_length // 2
        self.add_bos = add_bos
        self.add_eos = add_eos

        # Parse and format conversations
        self.samples = self._process_conversations(texts)

    def _parse_conversation_line(self, line: str) -> Optional[Dict[str, str]]:
        """Parse a User: ... AI: ... line."""
        # Support both inline and multiline formats
        pattern = r"(?i)User:\s*(.+?)\s*(?:AI|Assistant):\s*(.+)$"
        match = re.match(pattern, line.strip())
        if match:
            user_text = match.group(1).strip()
            assistant_text = match.group(2).strip()
            if user_text and assistant_text:
                return {"user": user_text, "assistant": assistant_text}
        return None

    def _format_conversation(self, user_text: str, assistant_text: str) -> str:
        """Format conversation with special tokens."""
        return (
            f"{self.USER_MARKER}\n{user_text}\n"
            f"{self.ASSISTANT_MARKER}\n{assistant_text}\n"
            f"{self.END_MARKER}"
        )

    def _process_conversations(self, texts: List[str]) -> List[Dict]:
        """Process all conversations into tokenized samples."""
        samples = []
        seen = set()  # Track duplicates

        for text in texts:
            # Parse conversation
            parsed = self._parse_conversation_line(text)
            if not parsed:
                continue

            user_text = parsed["user"]
            assistant_text = parsed["assistant"]

            # Skip duplicates
            key = (user_text, assistant_text)
            if key in seen:
                continue
            seen.add(key)

            # Format and tokenize
            formatted = self._format_conversation(user_text, assistant_text)

            # Find user/assistant boundaries for masking
            user_section = f"{self.USER_MARKER}\n{user_text}\n"
            assistant_section = f"{self.ASSISTANT_MARKER}\n{assistant_text}\n{self.END_MARKER}"

            # Tokenize with markers to find boundaries
            user_tokens = self.tokenizer.encode(user_section, add_bos=self.add_bos, add_eos=False)
            full_tokens = self.tokenizer.encode(
                formatted,
                add_bos=self.add_bos,
                add_eos=self.add_eos,
            )

            # Create labels with masking
            labels = full_tokens.copy()

            if self.mask_user_tokens:
                # Mask user tokens (keep BOS if present)
                start_mask = 0 if not self.add_bos else 1
                end_mask = len(user_tokens)

                for i in range(start_mask, min(end_mask, len(labels))):
                    labels[i] = -100  # Ignore index for loss

            # Create sample
            sample = {
                "input_ids": full_tokens,
                "labels": labels,
                "user_text": user_text,
                "assistant_text": assistant_text,
            }
            samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        sample = self.samples[idx]

        input_ids = sample["input_ids"]
        labels = sample["labels"]

        # Truncate if too long
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        # Pad if too short
        seq_len = len(input_ids)
        if seq_len < self.max_length:
            pad_length = self.max_length - seq_len
            input_ids = input_ids + [self.tokenizer.pad_id] * pad_length
            labels = labels + [-100] * pad_length
            attention_mask = [1] * seq_len + [0] * pad_length
        else:
            attention_mask = [1] * seq_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def get_sample_texts(self, n: int = 5) -> List[Dict[str, str]]:
        """Get sample texts for inspection."""
        samples = random.sample(self.samples, min(n, len(self.samples)))
        return [
            {
                "user": s["user_text"],
                "assistant": s["assistant_text"],
            }
            for s in samples
        ]
