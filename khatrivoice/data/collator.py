"""
Data collation utilities for KhatriVoice.

This module provides collators for batching dataset samples
with proper padding and attention mask handling.
"""

from typing import Dict, List, Any, Optional
import torch
from dataclasses import dataclass


@dataclass
class DataCollator:
    """
    Data collator for language model training.

    Handles batching of samples with:
    - Padding to the longest sequence in batch
    - Attention mask creation
    - Labels with ignore index for padding

    Attributes:
        tokenizer: KhatriTokenizer instance
        pad_to_multiple_of: Pad sequence length to multiple of this value
        max_length: Maximum sequence length (None for no limit)
        return_tensors: Return tensor type
    """

    tokenizer: Any  # KhatriTokenizer type hint causes circular import
    pad_to_multiple_of: Optional[int] = None
    max_length: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.

        Args:
            features: List of sample dictionaries

        Returns:
            Batch dictionary with padded tensors
        """
        # Find max length in batch
        max_len = max(f["input_ids"].shape[0] for f in features)

        # Apply max_length constraint
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)

        # Pad to multiple
        if self.pad_to_multiple_of is not None:
            max_len = ((max_len + self.pad_to_multiple_of - 1) //
                       self.pad_to_multiple_of * self.pad_to_multiple_of)

        batch_input_ids: List[torch.Tensor] = []
        batch_labels: List[torch.Tensor] = []
        batch_attention_mask: List[torch.Tensor] = []

        for feature in features:
            input_ids = feature["input_ids"]
            labels = feature["labels"]
            attention_mask = feature.get(
                "attention_mask",
                torch.ones_like(input_ids, dtype=torch.long)
            )

            # Truncate if needed
            if self.max_length is not None and input_ids.shape[0] > self.max_length:
                input_ids = input_ids[:self.max_length]
                labels = labels[:self.max_length]
                attention_mask = attention_mask[:self.max_length]

            # Pad if needed
            padding_length = max_len - input_ids.shape[0]
            if padding_length > 0:
                # Pad with pad_token_id
                pad_token_id = self.tokenizer.pad_id
                input_ids = torch.cat([
                    input_ids,
                    torch.full((padding_length,), pad_token_id, dtype=torch.long)
                ])

                # Pad labels with -100 (ignore index)
                labels = torch.cat([
                    labels,
                    torch.full((padding_length,), -100, dtype=torch.long)
                ])

                # Pad attention mask with 0
                attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros(padding_length, dtype=torch.long)
                ])

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_attention_mask.append(attention_mask)

        # Stack into batch
        return {
            "input_ids": torch.stack(batch_input_ids),
            "labels": torch.stack(batch_labels),
            "attention_mask": torch.stack(batch_attention_mask),
        }


@dataclass
class DataCollatorForCausalLM:
    """
    Data collator specifically for causal language modeling.

    This is an alternative to DataCollator with explicit support
    for causal language modeling specific features.

    Attributes:
        tokenizer: KhatriTokenizer instance
        max_length: Maximum sequence length
        pad_to_multiple_of: Pad to multiple of this value
        ignore_index: Index to ignore in loss calculation
        shift_labels: Whether to shift labels (default True for causal LM)
    """

    tokenizer: Any
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    ignore_index: int = -100
    shift_labels: bool = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.

        Args:
            features: List of sample dictionaries

        Returns:
            Batch dictionary with padded tensors
        """
        # Extract input_ids
        input_ids_list: List[torch.Tensor] = []
        for f in features:
            ids = f["input_ids"]
            if isinstance(ids, list):
                ids = torch.tensor(ids, dtype=torch.long)
            input_ids_list.append(ids)

        # Find max length
        max_len = max(ids.shape[0] for ids in input_ids_list)

        # Apply constraints
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)

        if self.pad_to_multiple_of is not None:
            max_len = ((max_len + self.pad_to_multiple_of - 1) //
                       self.pad_to_multiple_of * self.pad_to_multiple_of)

        # Pad all sequences
        batch_input_ids: List[torch.Tensor] = []
        batch_attention_mask: List[torch.Tensor] = []

        for input_ids in input_ids_list:
            # Truncate if needed
            if self.max_length is not None and input_ids.shape[0] > self.max_length:
                input_ids = input_ids[:self.max_length]

            # Pad if needed
            padding_length = max_len - input_ids.shape[0]
            if padding_length > 0:
                pad_token_id = self.tokenizer.pad_id
                input_ids = torch.cat([
                    input_ids,
                    torch.full((padding_length,), pad_token_id, dtype=torch.long)
                ])

                # Create attention mask with 0 for padding
                attention_mask = torch.cat([
                    torch.ones(input_ids.shape[0] - padding_length, dtype=torch.long),
                    torch.zeros(padding_length, dtype=torch.long)
                ])
            else:
                attention_mask = torch.ones(input_ids.shape[0], dtype=torch.long)

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)

        # Stack
        input_ids = torch.stack(batch_input_ids)
        attention_mask = torch.stack(batch_attention_mask)

        # Create labels (shifted input_ids for causal LM)
        if self.shift_labels:
            # Labels are input_ids shifted by 1
            labels = input_ids.clone()
            # Replace pad tokens with ignore_index
            labels[labels == self.tokenizer.pad_id] = self.ignore_index
        else:
            labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def create_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    collator: Optional[DataCollator] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for the dataset.

    Args:
        dataset: KhatriDataset or similar
        batch_size: Batch size
        shuffle: Whether to shuffle
        collator: DataCollator instance (required for variable-length sequences)
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        DataLoader instance
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
