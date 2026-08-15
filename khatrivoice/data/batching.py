"""
Batching utilities for KhatriVoice.

This module provides utilities for creating and managing batches
during training, including dynamic batching strategies.
"""

from typing import List, Dict, Any, Iterator, Optional
import torch
from torch.utils.data import Sampler, DataLoader

from khatrivoice.tokenizer.tokenizer import KhatriTokenizer


class DynamicBatchSampler(Sampler):
    """
    Dynamic batch sampler that groups samples by length.

    This sampler groups samples of similar length together to
    minimize padding and improve training efficiency.
    """

    def __init__(
        self,
        data_source: torch.utils.data.Dataset,
        max_tokens: int = 8192,
        max_batch_size: int = 32,
        drop_last: bool = False,
        sort_key: Optional[callable] = None,
    ) -> None:
        """
        Initialize the dynamic batch sampler.

        Args:
            data_source: Dataset to sample from
            max_tokens: Maximum tokens per batch
            max_batch_size: Maximum samples per batch
            drop_last: Whether to drop the last incomplete batch
            sort_key: Function to extract length from sample
        """
        self.data_source = data_source
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.drop_last = drop_last
        self.sort_key = sort_key or (lambda x: x["input_ids"].shape[0])

    def __iter__(self) -> Iterator[List[int]]:
        """Iterate over batches."""
        # Get all indices with their lengths
        indices_with_lengths = []
        for idx in range(len(self.data_source)):
            sample = self.data_source[idx]
            length = self.sort_key(sample)
            indices_with_lengths.append((idx, length))

        # Sort by length (shortest first for less padding, or longest first)
        indices_with_lengths.sort(key=lambda x: x[1])

        # Create batches
        current_batch: List[int] = []
        current_length = 0

        for idx, length in indices_with_lengths:
            # Check if adding this sample would exceed limits
            new_length = max(current_length, length) * (len(current_batch) + 1)

            if len(current_batch) >= self.max_batch_size or new_length > self.max_tokens:
                if current_batch:
                    yield current_batch
                current_batch = [idx]
                current_length = length
            else:
                current_batch.append(idx)
                current_length = max(current_length, length)

        # Yield last batch
        if current_batch and not self.drop_last:
            yield current_batch

    def __len__(self) -> int:
        """Return number of batches."""
        # Approximate
        return (len(self.data_source) + self.max_batch_size - 1) // self.max_batch_size


def chunk_sequences(
    sequences: List[List[int]],
    chunk_size: int,
    drop_last: bool = False,
) -> List[torch.Tensor]:
    """
    Chunk sequences into fixed-size pieces.

    Args:
        sequences: List of token ID sequences
        chunk_size: Size of each chunk
        drop_last: Whether to drop incomplete chunks

    Returns:
        List of chunk tensors
    """
    chunks: List[torch.Tensor] = []

    for seq in sequences:
        start = 0
        while start < len(seq):
            end = min(start + chunk_size, len(seq))
            if end - start < chunk_size and drop_last:
                break
            chunk = torch.tensor(seq[start:end], dtype=torch.long)
            chunks.append(chunk)
            start += chunk_size

    return chunks


def pack_sequences(
    sequences: List[List[int]],
    max_length: int,
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    """
    Pack multiple sequences into a single tensor with padding.

    Args:
        sequences: List of token ID sequences
        max_length: Maximum length (truncate longer sequences)
        pad_token_id: Token ID for padding

    Returns:
        Dictionary with input_ids and attention_mask
    """
    packed_ids: List[int] = []
    attention_masks: List[int] = []

    for seq in sequences:
        # Truncate
        seq = seq[:max_length]

        # Add tokens
        packed_ids.extend(seq)
        attention_masks.extend([1] * len(seq))

        # Add padding to max_length
        padding_length = max_length - len(seq)
        packed_ids.extend([pad_token_id] * padding_length)
        attention_masks.extend([0] * padding_length)

    return {
        "input_ids": torch.tensor(packed_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
    }
