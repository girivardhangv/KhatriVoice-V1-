"""
Text generation for KhatriVoice.
"""

import torch
from typing import Optional, List
from khatrivoice.model.khatrivoice import KhatriVoice
from khatrivoice.tokenizer.tokenizer import KhatriTokenizer


class Generator:
    """
    Text generator for trained KhatriVoice models.

    Args:
        model: Trained KhatriVoice model
        tokenizer: Trained KhatriTokenizer
        device: Device to run inference on
    """

    def __init__(
        self,
        model: KhatriVoice,
        tokenizer: KhatriTokenizer,
        device: str = "auto",
    ):
        self.model = model
        self.tokenizer = tokenizer

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample or take argmax

        Returns:
            Generated text
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_bos=False)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )

        # Decode
        output_text = self.tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

        return output_text

    def chat(
        self,
        user_input: str,
        history: Optional[List[str]] = None,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> str:
        """
        Generate a response in conversation format.

        Args:
            user_input: User's input text
            history: Optional conversation history
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            Generated assistant response
        """
        # Build prompt
        if history:
            prompt = " ".join(history) + f" User: {user_input} Assistant:"
        else:
            prompt = f"User: {user_input} Assistant:"

        # Generate
        response = self.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)

        # Extract just the assistant's response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()

        return response
