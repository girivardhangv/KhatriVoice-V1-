"""
Text generation for KhatriVoice.

This module provides text generation capabilities using the KhatriVoice model.
Supports greedy decoding, temperature sampling, top-k, and top-p sampling.
"""

from typing import Optional, List, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor

from khatrivoice.model.khatrivoice import KhatriVoice
from khatrivoice.tokenizer.tokenizer import KhatriTokenizer


class KhatriVoiceGenerator:
    """
    Text generator for KhatriVoice model.

    Supports multiple decoding strategies:
    - Greedy decoding
    - Temperature sampling
    - Top-k sampling
    - Top-p (nucleus) sampling
    - Combined strategies

    Attributes:
        model: KhatriVoice model instance
        tokenizer: KhatriTokenizer instance
        device: Device to run generation on
    """

    def __init__(
        self,
        model: KhatriVoice,
        tokenizer: KhatriTokenizer,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize the generator.

        Args:
            model: Trained KhatriVoice model
            tokenizer: Tokenizer for encoding/decoding
            device: Device for generation (auto-detected if None)
        """
        self.model = model
        self.tokenizer = tokenizer

        # Auto-detect device if not provided
        if device is None:
            from khatrivoice.utils.device import get_device
            device = get_device("auto")

        self.device = device
        self.model = model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        repetition_penalty: float = 1.0,
        stop_tokens: Optional[List[int]] = None,
    ) -> List[str]:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (1.0 = normal, <1.0 = more deterministic)
            top_k: If set, only sample from top-k tokens
            top_p: If set, sample from nucleus with cumulative probability top_p
            do_sample: If False, use greedy decoding
            num_return_sequences: Number of sequences to generate
            eos_token_id: End-of-sequence token ID (default: tokenizer.eos_id)
            pad_token_id: Padding token ID (default: tokenizer.pad_id)
            repetition_penalty: Penalty for repeated tokens (1.0 = no penalty, >1.0 = less repetition)
            stop_tokens: Additional token IDs to stop generation

        Returns:
            List of generated text strings
        """
        # Set defaults from tokenizer
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_id

        # Set up stop tokens (include EOS and conversation end token)
        if stop_tokens is None:
            stop_tokens = []
        if hasattr(self.tokenizer.vocab, 'end_id'):
            end_id = self.tokenizer.vocab.end_id
            if end_id not in stop_tokens:
                stop_tokens = stop_tokens + [eos_token_id, end_id]
            else:
                stop_tokens = stop_tokens + [eos_token_id]
        else:
            stop_tokens = stop_tokens + [eos_token_id]

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Expand for multiple sequences
        if num_return_sequences > 1:
            input_ids = input_ids.expand(num_return_sequences, -1)

        # Generate
        generated_ids = self._generate_loop(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            repetition_penalty=repetition_penalty,
            stop_tokens=stop_tokens,
        )

        # Decode
        results = []
        for ids in generated_ids:
            text = self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
            results.append(text)

        return results

    @torch.no_grad()
    def generate_from_ids(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        repetition_penalty: float = 1.0,
        stop_tokens: Optional[List[int]] = None,
    ) -> Tensor:
        """
        Generate from token IDs directly.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: If False, use greedy decoding
            eos_token_id: End-of-sequence token ID
            pad_token_id: Padding token ID
            repetition_penalty: Penalty for repeated tokens (1.0 = no penalty)
            stop_tokens: Additional token IDs to stop generation

        Returns:
            Generated token IDs [batch_size, seq_len + max_new_tokens]
        """
        # Set defaults from tokenizer
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_id

        return self._generate_loop(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            repetition_penalty=repetition_penalty,
            stop_tokens=stop_tokens,
        )

    def _generate_loop(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        do_sample: bool,
        eos_token_id: int,
        pad_token_id: int,
        repetition_penalty: float = 1.0,
        stop_tokens: Optional[List[int]] = None,
    ) -> Tensor:
        """
        Main generation loop.

        Uses KV cache for efficient generation.
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Track which sequences are finished
        unfinished_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)

        # Initialize KV cache
        past_key_values = None

        # Stop tokens for early stopping
        if stop_tokens is None:
            stop_tokens = [eos_token_id]
        elif eos_token_id not in stop_tokens:
            stop_tokens = stop_tokens + [eos_token_id]

        # Generation loop
        for step in range(max_new_tokens):
            # Prepare model inputs
            if past_key_values is not None:
                # Only need last token when using cache
                model_input_ids = input_ids[:, -1:]
            else:
                model_input_ids = input_ids

            # Forward pass
            outputs = self.model(
                input_ids=model_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = outputs["logits"]
            past_key_values = outputs.get("past_key_values")

            # Get logits for last position
            next_token_logits = logits[:, -1, :]

            # Apply repetition penalty
            if repetition_penalty > 1.0:
                for token_id in input_ids[0].tolist():
                    next_token_logits[0, token_id] /= repetition_penalty

            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature

            # Greedy or sampling
            if not do_sample or temperature == 0:
                # Greedy decoding
                next_tokens = next_token_logits.argmax(dim=-1)
            else:
                # Apply top-k filtering
                if top_k is not None and top_k > 0:
                    indices_to_remove = self._get_top_k_to_remove(next_token_logits, top_k)
                    next_token_logits = next_token_logits.masked_fill(
                        indices_to_remove, float("-inf")
                    )

                # Apply top-p (nucleus) filtering
                if top_p is not None and top_p < 1.0:
                    indices_to_remove = self._get_top_p_to_remove(next_token_logits, top_p)
                    next_token_logits = next_token_logits.masked_fill(
                        indices_to_remove, float("-inf")
                    )

                # Sample from the distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # Handle stop tokens (EOS and end of turn)
            if stop_tokens:
                # If finished, replace with pad token
                next_tokens = torch.where(
                    unfinished_sequences,
                    next_tokens,
                    pad_token_id,
                )

                # Check if any stop token was generated
                for stop_id in stop_tokens:
                    unfinished_sequences = unfinished_sequences & (next_tokens != stop_id)

            # Append next token
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)

            # Stop if all sequences are finished
            if not unfinished_sequences.any():
                break

        return input_ids

    def _get_top_k_to_remove(self, logits: Tensor, top_k: int) -> Tensor:
        """
        Get mask for tokens to remove for top-k sampling.

        Args:
            logits: Token logits [batch_size, vocab_size]
            top_k: Number of top tokens to keep

        Returns:
            Boolean mask where True indicates tokens to remove
        """
        vocab_size = logits.shape[-1]
        top_k = min(top_k, vocab_size)

        # Get the k-th largest value
        top_k_values, _ = torch.topk(logits, top_k, dim=-1)
        kth_value = top_k_values[:, -1].unsqueeze(-1)

        # Mask tokens below k-th value
        return logits < kth_value

    def _get_top_p_to_remove(self, logits: Tensor, top_p: float) -> Tensor:
        """
        Get mask for tokens to remove for top-p (nucleus) sampling.

        Args:
            logits: Token logits [batch_size, vocab_size]
            top_p: Cumulative probability threshold

        Returns:
            Boolean mask where True indicates tokens to remove
        """
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

        # Compute cumulative probabilities
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Create mask for tokens to remove
        # Keep tokens with cumulative probability <= top_p
        # Also keep at least one token (the highest probability one)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[:, 0] = False  # Always keep the best token

        # Unsort the mask
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)

        return mask

    @torch.no_grad()
    def compute_perplexity(
        self,
        text: str,
        max_length: int = 512,
    ) -> float:
        """
        Compute perplexity of a text under the model.

        Args:
            text: Input text
            max_length: Maximum sequence length

        Returns:
            Perplexity value
        """
        # Encode
        input_ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Truncate if needed
        if input_ids.shape[1] > max_length:
            input_ids = input_ids[:, :max_length]

        # Forward pass with labels
        outputs = self.model(input_ids=input_ids, labels=input_ids)
        loss = outputs["loss"]

        # Compute perplexity
        perplexity = torch.exp(loss).item()

        return perplexity

    @torch.no_grad()
    def compute_batch_perplexity(
        self,
        texts: List[str],
        max_length: int = 512,
        batch_size: int = 8,
    ) -> float:
        """
        Compute average perplexity over multiple texts.

        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            batch_size: Batch size for processing

        Returns:
            Average perplexity
        """
        total_loss = 0.0
        total_tokens = 0

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Encode all texts in batch
            batch_ids = []
            for text in batch_texts:
                ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
                if len(ids) > max_length:
                    ids = ids[:max_length]
                batch_ids.append(ids)

            # Pad to same length
            max_len = max(len(ids) for ids in batch_ids)
            padded_ids = []
            attention_mask = []
            for ids in batch_ids:
                padding_length = max_len - len(ids)
                padded_ids.append(ids + [self.tokenizer.pad_id] * padding_length)
                attention_mask.append([1] * len(ids) + [0] * padding_length)

            input_ids = torch.tensor(padded_ids, dtype=torch.long, device=self.device)
            labels = input_ids.clone()
            labels[torch.tensor(attention_mask, device=self.device) == 0] = -100

            # Forward pass
            outputs = self.model(input_ids=input_ids, labels=labels, attention_mask=torch.tensor(attention_mask, device=self.device))
            loss = outputs["loss"]

            # Accumulate
            total_loss += loss.item() * len(batch_texts)
            total_tokens += sum(len(ids) for ids in batch_ids)

        return torch.exp(torch.tensor(total_loss / total_tokens)).item()


def create_generator(
    model: KhatriVoice,
    tokenizer: KhatriTokenizer,
    device: Optional[str] = None,
) -> KhatriVoiceGenerator:
    """
    Create a generator instance.

    Args:
        model: Trained KhatriVoice model
        tokenizer: Tokenizer for encoding/decoding
        device: Device (auto, cpu, cuda)

    Returns:
        KhatriVoiceGenerator instance
    """
    if device is not None:
        from khatrivoice.utils.device import get_device
        device = get_device(device)

    return KhatriVoiceGenerator(model=model, tokenizer=tokenizer, device=device)


def load_generator_from_checkpoint(
    checkpoint_path: str,
    tokenizer_dir: str,
    device: str = "auto",
) -> KhatriVoiceGenerator:
    """
    Load a generator from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        tokenizer_dir: Path to tokenizer directory
        device: Device to use

    Returns:
        KhatriVoiceGenerator instance
    """
    from khatrivoice.model.khatrivoice import load_model

    # Load tokenizer
    tokenizer = KhatriTokenizer.load(tokenizer_dir)

    # Load model
    model = load_model(checkpoint_path, device=device)

    # Create generator
    return create_generator(model=model, tokenizer=tokenizer, device=device)


def generate_chat_response(
    generator: KhatriVoiceGenerator,
    user_message: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> str:
    """
    Generate a chat response to a user message.

    This formats the input with proper conversation tokens and generates a response.

    Args:
        generator: KhatriVoiceGenerator instance
        user_message: User's message
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (lower = more focused)
        top_p: Nucleus sampling threshold
        repetition_penalty: Penalty for repeated tokens

    Returns:
        Assistant's response text (stripped of special tokens)
    """
    # Format with conversation tokens
    USER_TOKEN = "<user>"
    ASSISTANT_TOKEN = "<|assistant>"

    prompt = f"{USER_TOKEN}\n{user_message}\n{ASSISTANT_TOKEN}\n"

    # Generate response
    responses = generator.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
    )

    if responses:
        # Clean up the response
        response = responses[0]
        # Remove any remaining special tokens
        response = response.replace(ASSISTANT_TOKEN, "")
        response = response.replace("<|end|>", "")
        response = response.strip()
        return response

    return ""
