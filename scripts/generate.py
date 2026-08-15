#!/usr/bin/env python3
"""
Generate text with a trained KhatriVoice model.

Usage:
    python scripts/generate.py --model output/model_final.pt --prompt "hello"
"""

import argparse
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from khatrivoice.config.model_config import KhatriVoiceConfig
from khatrivoice.model.khatrivoice import KhatriVoice
from khatrivoice.tokenizer.tokenizer import KhatriTokenizer
from khatrivoice.utils.device import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with KhatriVoice")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer (default: same dir as model)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Input prompt",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (1.0 = normal, <1.0 = more conservative)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling (0 = disabled)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto, cpu, cuda)",
    )
    return parser.parse_args()


@torch.no_grad()
def generate(
    model: KhatriVoice,
    tokenizer: KhatriTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    device: torch.device = None,
) -> str:
    """
    Generate text from a prompt.

    Args:
        model: KhatriVoice model
        tokenizer: KhatriTokenizer
        prompt: Input prompt
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        device: Device to use

    Returns:
        Generated text
    """
    model.eval()
    device = device or next(model.parameters()).device

    # Encode prompt
    input_ids = tokenizer.encode(prompt, add_bos=True)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Generate tokens
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        # Forward pass
        outputs = model(generated, use_cache=True)
        logits = outputs["logits"]

        # Get logits for last token
        next_logits = logits[:, -1, :] / temperature

        # Apply top-k filtering
        if top_k > 0:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[:, [-1]]] = float("-inf")

        # Sample
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Stop if EOS
        if next_token.item() == tokenizer.eos_id:
            break

        # Append
        generated = torch.cat([generated, next_token], dim=1)

    # Decode
    text = tokenizer.decode(generated[0].tolist())
    return text


def main():
    args = parse_args()

    # Get device
    device = get_device(args.device)
    print(f"Device: {device}")

    # Load model
    model_path = Path(args.model)
    print(f"Loading model from: {model_path}")

    checkpoint = torch.load(model_path, map_location=str(device))

    if "config" in checkpoint:
        config_dict = checkpoint["config"]
        config = KhatriVoiceConfig.from_dict(config_dict)
    else:
        # Use default config
        config = KhatriVoiceConfig()

    model = KhatriVoice(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")

    # Load tokenizer
    if args.tokenizer:
        tokenizer_path = Path(args.tokenizer)
    else:
        tokenizer_path = model_path.parent / "tokenizer.json"

    print(f"Loading tokenizer from: {tokenizer_path}")

    tokenizer = KhatriTokenizer()
    tokenizer.load(tokenizer_path)

    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Generate
    print("\n" + "=" * 60)
    print("Generation")
    print("=" * 60)
    print(f"Prompt: {args.prompt or '(empty)'}")

    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )

    print(f"\nGenerated:\n{generated_text}")
    print("=" * 60)


if __name__ == "__main__":
    main()
