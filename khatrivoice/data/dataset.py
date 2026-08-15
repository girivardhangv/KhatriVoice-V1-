"""
PyTorch Dataset classes for KhatriVoice.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Optional

from khatrivoice.tokenizer.tokenizer import KhatriTokenizer


class KhatriDataset(Dataset):
    """
    Dataset for KhatriVoice language model training.

    Args:
        tokenizer: KhatriTokenizer instance
        texts: List of text samples
        max_length: Maximum sequence length
    """

    def __init__(
        self,
        tokenizer: KhatriTokenizer,
        texts: List[str],
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]

        # Encode text
        input_ids = self.tokenizer.encode(text, add_eos=True)

        # Truncate if needed
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]

        # Create labels (same as input_ids for causal LM)
        labels = input_ids.copy()

        # Pad to max_length
        attention_mask = [1] * len(input_ids)
        padding_length = self.max_length - len(input_ids)

        input_ids = input_ids + [self.tokenizer.pad_id] * padding_length
        labels = labels + [-100] * padding_length  # -100 is ignored in loss
        attention_mask = attention_mask + [0] * padding_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class ConversationDataset(Dataset):
    """
    Dataset for conversation-style training with label masking.

    Masks user tokens so the model only learns to predict assistant responses.

    Args:
        tokenizer: KhatriTokenizer instance
        texts: List of conversation text samples
        max_length: Maximum sequence length
        mask_user_tokens: Whether to mask user tokens in labels
    """

    def __init__(
        self,
        tokenizer: KhatriTokenizer,
        texts: List[str],
        max_length: int = 512,
        mask_user_tokens: bool = True,
    ):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
        self.mask_user_tokens = mask_user_tokens

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]

        # Encode text
        input_ids = self.tokenizer.encode(text, add_eos=True)

        # Create labels
        labels = input_ids.copy()

        # Mask user tokens if requested
        if self.mask_user_tokens:
            # Find "User:" and "Assistant:" tokens
            user_token = self.tokenizer.vocab.get("user", self.tokenizer.unk_id)
            assistant_token = self.tokenizer.vocab.get("assistant", self.tokenizer.unk_id)

            # Simple heuristic: mask until we find "assistant"
            in_user_section = False
            for i, token_id in enumerate(input_ids):
                if token_id == user_token:
                    in_user_section = True
                    labels[i] = -100
                elif token_id == assistant_token:
                    in_user_section = False
                    labels[i] = -100
                elif in_user_section:
                    labels[i] = -100

        # Truncate if needed
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        # Create attention mask
        attention_mask = [1] * len(input_ids)

        # Pad to max_length
        padding_length = self.max_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_id] * padding_length
        labels = labels + [-100] * padding_length
        attention_mask = attention_mask + [0] * padding_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
