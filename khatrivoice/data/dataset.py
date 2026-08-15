"""
Dataset implementation for KhatriVoice.

This module provides PyTorch Dataset implementations for
language model training with proper handling of sequence packing
and causal language modeling targets.
"""

from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import torch
from torch.utils.data import Dataset

from khatrivoice.tokenizer.tokenizer import KhatriTokenizer


class KhatriDataset(Dataset):
    """
    Dataset for KhatriVoice language model training.

    This dataset handles:
    - Tokenization of text samples
    - Sequence packing into fixed-length windows
    - Creating input/target pairs for causal language modeling

    For causal LM:
        input: tokens[0:N-1]
        target: tokens[1:N]

    Attributes:
        tokenizer: KhatriTokenizer instance
        sequences: List of token ID sequences
        max_length: Maximum sequence length
        stride: Stride for sliding window (default: max_length)
    """

    def __init__(
        self,
        tokenizer: KhatriTokenizer,
        texts: Optional[List[str]] = None,
        sequences: Optional[List[List[int]]] = None,
        max_length: int = 512,
        stride: Optional[int] = None,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            tokenizer: KhatriTokenizer instance
            texts: List of text samples (will be tokenized)
            sequences: Pre-tokenized sequences (alternative to texts)
            max_length: Maximum sequence length
            stride: Stride for sliding window (default: max_length, no overlap)
            add_bos: Whether to add BOS token
            add_eos: Whether to add EOS token
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride if stride is not None else max_length
        self.add_bos = add_bos
        self.add_eos = add_eos

        # Process texts or use pre-tokenized sequences
        if sequences is not None:
            self.sequences = sequences
        elif texts is not None:
            self.sequences = self._tokenize_texts(texts)
        else:
            raise ValueError("Must provide either 'texts' or 'sequences'")

        # Build the flat token array and index
        self._build_index()

    def _tokenize_texts(self, texts: List[str]) -> List[List[int]]:
        """
        Tokenize a list of texts.

        Args:
            texts: List of text samples

        Returns:
            List of token ID sequences
        """
        sequences: List[List[int]] = []
        for text in texts:
            ids = self.tokenizer.encode(
                text,
                add_bos=self.add_bos,
                add_eos=self.add_eos,
            )
            if len(ids) > 1:  # At least 2 tokens for input/target pair
                sequences.append(ids)
        return sequences

    def _build_index(self) -> None:
        """
        Build the index for accessing sequences.

        This creates a flat list of all tokens and computes
        indices for creating fixed-length windows.
        """
        # Flatten all sequences with separator handling
        self.all_tokens: List[int] = []
        self.sample_indices: List[Tuple[int, int]] = []

        current_pos = 0
        for seq in self.sequences:
            # For each sequence, create windows
            seq_len = len(seq)

            # Create windows with stride
            start = 0
            while start < seq_len:
                end = min(start + self.max_length + 1, seq_len)  # +1 for target

                # Only include if we have at least 2 tokens (input + target)
                if end - start >= 2:
                    # Extract window tokens
                    window = seq[start:end]

                    # Record position in all_tokens
                    token_start = len(self.all_tokens)
                    self.all_tokens.extend(window)
                    token_end = len(self.all_tokens)

                    self.sample_indices.append((token_start, token_end))

                # Move to next window
                start += self.stride

        # Handle case where no valid windows were created
        if not self.sample_indices:
            # Add a dummy sample to avoid empty dataset
            self.all_tokens = [self.tokenizer.pad_id] * 2
            self.sample_indices = [(0, 2)]

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample by index.

        Args:
            idx: Sample index

        Returns:
            Dictionary with 'input_ids' and 'labels' tensors
        """
        token_start, token_end = self.sample_indices[idx]
        tokens = self.all_tokens[token_start:token_end]

        # Create input/target pair for causal LM
        # Input: tokens[0:N-1], Target: tokens[1:N]
        input_ids = tokens[:-1]
        labels = tokens[1:]

        # Pad to max_length if needed
        seq_len = len(input_ids)
        if seq_len < self.max_length:
            pad_length = self.max_length - seq_len
            input_ids = input_ids + [self.tokenizer.pad_id] * pad_length
            labels = labels + [-100] * pad_length  # -100 is ignored by PyTorch loss

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(
                [1] * seq_len + [0] * (self.max_length - seq_len),
                dtype=torch.long,
            ),
        }

    def get_sequence_lengths(self) -> List[int]:
        """Get the length of each original sequence."""
        return [len(seq) for seq in self.sequences]

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        lengths = self.get_sequence_lengths()
        total_tokens = sum(lengths)

        return {
            "num_sequences": len(self.sequences),
            "num_samples": len(self),
            "total_tokens": total_tokens,
            "avg_length": total_tokens / len(self.sequences) if self.sequences else 0,
            "min_length": min(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "max_window_size": self.max_length,
        }


class KhatriIterableDataset(torch.utils.data.IterableDataset):
    """
    Iterable dataset for streaming large corpora.

    This is useful for very large datasets that don't fit in memory.
    Streams data from files on-the-fly.
    """

    def __init__(
        self,
        tokenizer: KhatriTokenizer,
        filepaths: List[str | Path],
        max_length: int = 512,
        buffer_size: int = 10000,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> None:
        """
        Initialize the iterable dataset.

        Args:
            tokenizer: KhatriTokenizer instance
            filepaths: List of file paths to read from
            max_length: Maximum sequence length
            buffer_size: Buffer size for shuffling
            add_bos: Whether to add BOS token
            add_eos: Whether to add EOS token
        """
        self.tokenizer = tokenizer
        self.filepaths = [Path(fp) for fp in filepaths]
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.add_bos = add_bos
        self.add_eos = add_eos

    def _stream_lines(self):
        """Stream lines from files."""
        import random

        for filepath in self.filepaths:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line

    def __iter__(self):
        """Iterate over the dataset."""
        import random
        from collections import deque

        buffer = deque()

        for line in self._stream_lines():
            # Tokenize
            ids = self.tokenizer.encode(
                line,
                add_bos=self.add_bos,
                add_eos=self.add_eos,
            )

            # Split into windows
            start = 0
            while start < len(ids):
                end = min(start + self.max_length + 1, len(ids))
                window = ids[start:end]

                if len(window) >= 2:
                    # Create input/target pair
                    input_ids = window[:-1]
                    labels = window[1:]

                    # Pad if needed
                    seq_len = len(input_ids)
                    if seq_len < self.max_length:
                        input_ids = input_ids + [self.tokenizer.pad_id] * (self.max_length - seq_len)
                        labels = labels + [-100] * (self.max_length - seq_len)

                    sample = {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "labels": torch.tensor(labels, dtype=torch.long),
                        "attention_mask": torch.tensor(
                            [1] * seq_len + [0] * (self.max_length - seq_len),
                            dtype=torch.long,
                        ),
                    }

                    buffer.append(sample)

                    # Yield from buffer when full
                    if len(buffer) >= self.buffer_size:
                        idx = random.randint(0, len(buffer) - 1)
                        yield buffer[idx]
                        buffer.remove(buffer[idx])

                start += self.max_length

        # Yield remaining items
        while buffer:
            yield buffer.popleft()


def create_dataset_from_files(
    tokenizer: KhatriTokenizer,
    directory: str | Path,
    pattern: str = "*.txt",
    max_length: int = 512,
    encoding: str = "utf-8",
) -> KhatriDataset:
    """
    Create a dataset from text files in a directory.

    Args:
        tokenizer: KhatriTokenizer instance
        directory: Directory containing text files
        pattern: Glob pattern for files
        max_length: Maximum sequence length
        encoding: File encoding

    Returns:
        KhatriDataset instance
    """
    from khatrivoice.data.preprocessing import load_text_files

    texts = load_text_files(directory, pattern=pattern, encoding=encoding)
    return KhatriDataset(
        tokenizer=tokenizer,
        texts=texts,
        max_length=max_length,
    )


def create_tiny_dataset_for_testing(
    tokenizer: KhatriTokenizer,
    max_length: int = 32,
) -> KhatriDataset:
    """
    Create a tiny dataset for testing and debugging.

    Args:
        tokenizer: KhatriTokenizer instance
        max_length: Maximum sequence length

    Returns:
        KhatriDataset with tiny samples
    """
    from khatrivoice.data.preprocessing import create_tiny_dataset

    # Train tokenizer on tiny data first
    tiny_texts = create_tiny_dataset()
    tokenizer.train(tiny_texts, vocab_size=50)

    # Create dataset
    return KhatriDataset(
        tokenizer=tokenizer,
        texts=tiny_texts,
        max_length=max_length,
    )


class ConversationDataset(Dataset):
    """
    Dataset for conversation-style training data.
    
    This dataset handles conversation data formatted with special tokens:
        <user>\n{user_text}\n<assistant>\n{assistant_text}\n<|end|>
    
    Key features:
    - Masks user prompt tokens (labels=-100) so model only learns assistant responses
    - Properly handles conversation boundaries
    - Supports multi-turn conversations
    """

    def __init__(
        self,
        tokenizer: KhatriTokenizer,
        texts: List[str],
        max_length: int = 512,
        stride: Optional[int] = None,
        mask_user_tokens: bool = True,
    ) -> None:
        """
        Initialize the conversation dataset.

        Args:
            tokenizer: KhatriTokenizer instance
            texts: List of formatted conversation strings
            max_length: Maximum sequence length
            stride: Stride for sliding window (default: max_length)
            mask_user_tokens: Whether to mask user tokens in labels
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride if stride is not None else max_length
        self.mask_user_tokens = mask_user_tokens

        # Get special token IDs
        self.user_id = tokenizer.vocab.user_id
        self.assistant_id = tokenizer.vocab.assistant_id
        self.end_id = tokenizer.vocab.end_id
        self.pad_id = tokenizer.pad_id

        # Tokenize all texts
        self.sequences: List[List[int]] = []
        self._tokenize_texts(texts)

        # Build index for sliding windows
        self.sample_indices: List[Tuple[int, int]] = []
        self._build_index()

    def _tokenize_texts(self, texts: List[str]) -> None:
        """Tokenize all texts."""
        for text in texts:
            # Encode without BOS/EOS since we use special conversation tokens
            ids = self.tokenizer.encode(text, add_bos=False, add_eos=False)
            if len(ids) >= 2:
                self.sequences.append(ids)

    def _build_index(self) -> None:
        """Build sample index with sliding windows."""
        for seq_idx, seq in enumerate(self.sequences):
            seq_len = len(seq)
            start = 0

            while start < seq_len:
                end = min(start + self.max_length + 1, seq_len)
                
                # Need at least 2 tokens for input/target pair
                if end - start >= 2:
                    self.sample_indices.append((seq_idx, start, end))
                
                # Move to next window
                start += self.stride

        # Handle empty dataset
        if not self.sample_indices:
            self.sequences = [[self.pad_id] * 2]
            self.sample_indices = [(0, 0, 2)]

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq_idx, start, end = self.sample_indices[idx]
        tokens = self.sequences[seq_idx][start:end]

        # Create input/target pair
        input_ids = tokens[:-1]
        labels = tokens[1:].copy()

        # Mask user tokens in labels if enabled
        if self.mask_user_tokens:
            labels = self._mask_user_prompts(tokens[:-1], labels)

        # Pad to max_length
        seq_len = len(input_ids)
        if seq_len < self.max_length:
            pad_length = self.max_length - seq_len
            input_ids = input_ids + [self.pad_id] * pad_length
            labels = labels + [-100] * pad_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(
                [1] * seq_len + [0] * (self.max_length - seq_len),
                dtype=torch.long,
            ),
        }

    def _mask_user_prompts(self, input_ids: List[int], labels: List[int]) -> List[int]:
        """
        Mask user prompt tokens in labels.
        
        Only assistant response tokens should have loss computed.
        """
        labels = labels.copy()
        in_user_section = False
        in_assistant_section = False

        for i, token_id in enumerate(input_ids):
            if token_id == self.user_id:
                # Start of user section
                in_user_section = True
                in_assistant_section = False
                labels[i] = -100  # Mask the user token itself
            elif token_id == self.assistant_id:
                # Start of assistant section
                in_user_section = False
                in_assistant_section = True
                labels[i] = -100  # Mask the assistant token itself
            elif token_id == self.end_id:
                # End of turn
                in_user_section = False
                in_assistant_section = False
                labels[i] = -100  # Mask the end token
            elif in_user_section:
                # User content - mask it
                labels[i] = -100
            # Assistant content - keep labels (don't mask)
            # This is the part we want the model to learn

        return labels

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        total_tokens = sum(len(seq) for seq in self.sequences)
        avg_length = total_tokens / len(self.sequences) if self.sequences else 0
        
        return {
            "num_sequences": len(self.sequences),
            "num_samples": len(self.sample_indices),
            "total_tokens": total_tokens,
            "avg_sequence_length": avg_length,
            "max_length": self.max_length,
        }
