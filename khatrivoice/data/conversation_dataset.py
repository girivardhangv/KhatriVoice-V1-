"""
Conversation-aware dataset for KhatriVoice.

This module provides a dataset that properly handles User/AI conversations
with label masking so the model only learns to predict assistant responses.
"""

import re
from typing import Dict, List, Optional, Tuple
import torch
from torch import Tensor

from khatrivoice.tokenizer.tokenizer import KhatriTokenizer


class ConversationDataset(torch.utils.data.Dataset):
    """
    Dataset for conversational training with proper label masking.

    This dataset:
    1. Parses conversations in <user>...\n<|assistant>...\n<|end|> format
    2. Masks user tokens in labels (-100) so loss is only computed on assistant responses

    Labels for user tokens are set to -100 (ignored by loss).
    """

    # Token markers (must match vocabulary.py)
    USER_MARKER = "<user>"
    ASSISTANT_MARKER = "<|assistant>"
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
            texts: List of text lines (will be joined and parsed)
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

        # Join all lines and parse conversations
        full_text = "\n".join(texts)
        self.samples = self._parse_conversations(full_text)

    def _parse_conversations(self, text: str) -> List[Dict]:
        """Parse conversations from text with special token format."""
        samples = []

        # Split by conversation blocks (separated by blank lines)
        blocks = re.split(r'\n\s*\n', text)

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            # Must have all special tokens
            if self.USER_MARKER not in block:
                continue
            if self.ASSISTANT_MARKER not in block:
                continue
            if self.END_MARKER not in block:
                continue

            # Extract user message
            user_pattern = re.escape(self.USER_MARKER) + r'\s*\n(.+?)\n\s*' + re.escape(self.ASSISTANT_MARKER)
            user_match = re.search(user_pattern, block, re.DOTALL)
            if not user_match:
                continue
            user_text = user_match.group(1).strip()

            # Extract assistant message
            asst_pattern = re.escape(self.ASSISTANT_MARKER) + r'\s*\n(.+?)\n\s*' + re.escape(self.END_MARKER)
            asst_match = re.search(asst_pattern, block, re.DOTALL)
            if not asst_match:
                continue
            asst_text = asst_match.group(1).strip()

            # Validate content
            if len(user_text) < 1 or len(asst_text) < 1:
                continue

            samples.append({
                'user': user_text,
                'assistant': asst_text,
            })

        return samples

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """Get a single sample."""
        sample = self.samples[idx]
        user_text = sample['user']
        assistant_text = sample['assistant']

        # Format conversation
        formatted = f"{self.USER_MARKER}\n{user_text}\n{self.ASSISTANT_MARKER}\n{assistant_text}\n{self.END_MARKER}"

        # Tokenize
        input_ids = self.tokenizer.encode(formatted, add_bos=self.add_bos, add_eos=self.add_eos)

        # Create labels - mask user tokens if requested
        labels = input_ids.copy()

        if self.mask_user_tokens:
            # Find user/assistant boundaries
            user_section = f"{self.USER_MARKER}\n{user_text}\n"
            assistant_section = f"{self.ASSISTANT_MARKER}\n{assistant_text}\n{self.END_MARKER}"

            user_tokens = self.tokenizer.encode(user_section, add_bos=self.add_bos, add_eos=False)

            # Mask user portion (set to -100)
            for i in range(min(len(user_tokens), len(labels))):
                labels[i] = -100

        # Truncate to max_length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        }
