"""
Data collator for batching samples.
"""

import torch
from typing import List, Dict, Any


class DataCollator:
    """
    Collates multiple samples into a batch.

    Args:
        pad_token_id: Token ID for padding
        ignore_index: Index to ignore in loss calculation
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        ignore_index: int = -100,
    ):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.

        Args:
            batch: List of sample dictionaries

        Returns:
            Batched tensor dictionary
        """
        input_ids = torch.stack([item["input_ids"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
