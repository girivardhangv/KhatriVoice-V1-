#!/usr/bin/env python3
"""
Generate text with KhatriVoice language model.

Usage:
    python scripts/generate.py --checkpoint checkpoints/best.pt --prompt "hello"
    python scripts/generate.py --checkpoint checkpoints/best.pt --prompt "hello" --temperature 0.8 --top-k 50

This script loads a trained KhatriVoice model and generates text.
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from khatrivoice.config.model_config import KhatriVoiceConfig
from khatrivoice.model.khatrivoice import KhatriVoice
from khatrivoice.tokenizer.tokenizer import KhatriTokenizer
from khatrivoice.inference.generator import KhatriVoiceGenerator, create_generator
from khatrivoice.utils.device import get_device


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate text with KhatriVoice",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Greedy decoding (deterministic)
  python scripts/generate.py --checkpoint checkpoints/best.pt --prompt "hello world"

  # Temperature sampling
  python scripts/generate.py --checkpoint checkpoints/best.pt --prompt "hello" --temperature 0.8

  # Top-k sampling
  python scripts/generate.py --checkpoint checkpoints/best.pt --prompt "hello" --top-k 50

  # Top-p (nucleus) sampling
  python scripts/generate.py --checkpoint checkpoints/best.pt --prompt "hello" --top-p 0.95

  # Combined strategies
  python scripts/generate.py --checkpoint checkpoints/best.pt --prompt "hello" --temperature 0.7 --top-k 50 --top-p 0.9

  # Multiple sequences
  python scripts/generate.py --checkpoint checkpoints/best.pt --prompt "hello" --num-sequences 5
        """,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer directory (default: loaded from same dir as checkpoint)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: loaded from checkpoint)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="File containing the prompt (alternative to --prompt)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (1.0 = normal, <1.0 = more deterministic)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling (keep only top-k tokens)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding (ignores temperature/top-k/top-p)",
    )
    parser.add_argument(
        "--num-sequences",
        "--num-return-sequences",
        dest="num_sequences",
        type=int,
        default=1,
        help="Number of sequences to generate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run on",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable KV cache during generation (slower)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str, device: str = "auto"):
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Tuple of (model, config, checkpoint)
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Get config
    if "config" in checkpoint:
        config = KhatriVoiceConfig.from_dict(checkpoint["config"])
    else:
        raise ValueError("Checkpoint does not contain config")

    # Create model
    model = KhatriVoice(config)

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])

    # Move to device
    device = get_device(device)
    model = model.to(device)
    model.eval()

    return model, config, checkpoint


def load_tokenizer(tokenizer_path: str, checkpoint_dir: Path):
    """
    Load tokenizer.

    Args:
        tokenizer_path: Explicit path to tokenizer
        checkpoint_dir: Directory where checkpoint is located

    Returns:
        KhatriTokenizer
    """
    # Try explicit path first
    if tokenizer_path:
        tokenizer_path = Path(tokenizer_path)
        if tokenizer_path.exists():
            return KhatriTokenizer.load(tokenizer_path)

    # Try checkpoint directory
    checkpoint_tokenizer = checkpoint_dir / "tokenizer"
    if checkpoint_tokenizer.exists():
        return KhatriTokenizer.load(checkpoint_tokenizer)

    # Try data directory
    data_tokenizer = Path("data/processed/tokenizer")
    if data_tokenizer.exists():
        return KhatriTokenizer.load(data_tokenizer)

    raise FileNotFoundError(
        f"Tokenizer not found. Please specify --tokenizer path or "
        f"ensure tokenizer is in {checkpoint_tokenizer}"
    )


def main():
    """Main generation function."""
    args = parse_args()

    # Set seed if provided
    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Get prompt
    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
    if not prompt:
        print("Error: Please provide a prompt via --prompt or --prompt-file")
        sys.exit(1)

    print("=" * 60)
    print("KhatriVoice Generation")
    print("=" * 60)
    print()

    # Load model
    print("Loading model...")
    checkpoint_path = Path(args.checkpoint)
    model, config, checkpoint = load_model_from_checkpoint(
        args.checkpoint,
        device=args.device,
    )

    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Config: hidden_size={config.hidden_size}, layers={config.num_layers}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    try:
        tokenizer = load_tokenizer(args.tokenizer, checkpoint_path.parent)
        print(f"  Vocabulary size: {tokenizer.vocab_size}")
    except FileNotFoundError:
        print("\nWarning: Tokenizer not found. Creating a new one from the prompt.")
        # Create a minimal tokenizer for testing
        tokenizer = KhatriTokenizer(lowercase=True)
        tokenizer.train([prompt], vocab_size=config.vocab_size)
        print(f"  Created tokenizer with vocab size: {tokenizer.vocab_size}")

    # Create generator
    print("\nCreating generator...")
    generator = create_generator(model=model, tokenizer=tokenizer, device=args.device)
    print(f"  Device: {generator.device}")

    # Generate
    print("\n" + "=" * 60)
    print("Generating...")
    print("=" * 60)
    print()
    print(f"Prompt: \"{prompt}\"")
    print()
    print(f"Settings:")
    print(f"  max_new_tokens: {args.max_new_tokens}")
    print(f"  temperature: {args.temperature if not args.greedy else 'N/A (greedy)'}")
    print(f"  top_k: {args.top_k}")
    print(f"  top_p: {args.top_p}")
    print(f"  greedy: {args.greedy}")
    print(f"  num_sequences: {args.num_sequences}")
    print()

    # Run generation
    results = generator.generate(
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=not args.greedy,
        num_return_sequences=args.num_sequences,
    )

    # Display results
    print("=" * 60)
    print("Results")
    print("=" * 60)

    for i, text in enumerate(results):
        if args.num_sequences > 1:
            print(f"\n[{i + 1}] {text}")
        else:
            print(text)

    # Compute perplexity if available
    if len(results) > 0:
        print()
        print("-" * 60)
        try:
            ppl = generator.compute_perplexity(prompt + results[0])
            print(f"Perplexity of prompt+generation: {ppl:.2f}")
        except Exception:
            pass  # Skip if computation fails

    print()


if __name__ == "__main__":
    main()
