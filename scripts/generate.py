#!/usr/bin/env python3
"""
Generate text with KhatriVoice language model.

Supports both:
1. Free-form text generation
2. Conversational chat mode (formats input with special tokens)

Usage:
    # Free-form generation
    python scripts/generate.py --checkpoint checkpoints/best.pt --prompt "hello"

    # Conversational mode
    python scripts/generate.py --checkpoint checkpoints/best.pt --chat --prompt "What is Python?"

    # Interactive chat
    python scripts/generate.py --checkpoint checkpoints/best.pt --interactive
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
from khatrivoice.inference.generator import KhatriVoiceGenerator
from khatrivoice.utils.device import get_device


# Token markers (must match vocabulary.py)
USER_MARKER = ""
ASSISTANT_MARKER = ""
END_MARKER = "<|end|>"


def format_chat_prompt(user_message: str) -> str:
    """Format a user message for the model."""
    return f"{USER_MARKER}\n{user_message}\n{ASSISTANT_MARKER}\n"


def extract_assistant_response(text: str) -> str:
    """Extract just the assistant's response from generated text."""
    # Remove user section
    if ASSISTANT_MARKER in text:
        idx = text.find(ASSISTANT_MARKER)
        text = text[idx + len(ASSISTANT_MARKER):].strip()

    # Stop at END marker
    if END_MARKER in text:
        text = text[:text.find(END_MARKER)].strip()

    # Remove any remaining special tokens
    text = text.replace(USER_MARKER, "").strip()
    text = text.replace(ASSISTANT_MARKER, "").strip()
    text = text.replace(END_MARKER, "").strip()

    return text


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate text with KhatriVoice",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to tokenizer directory")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt for generation")
    parser.add_argument("--chat", action="store_true",
                        help="Use conversational format")
    parser.add_argument("--interactive", action="store_true",
                        help="Start interactive chat session")
    parser.add_argument("--max-new-tokens", type=int, default=100,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Nucleus sampling threshold")
    parser.add_argument("--repetition-penalty", type=float, default=1.1,
                        help="Repetition penalty")
    parser.add_argument("--num-sequences", type=int, default=1,
                        help="Number of sequences to generate")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")

    return parser.parse_args()


def load_generator(checkpoint_path: str, tokenizer_path: str, device: str):
    """Load model and create generator."""
    device = get_device(device)

    # Load tokenizer
    tokenizer = KhatriTokenizer.load(tokenizer_path)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "config" in checkpoint:
        config_dict = checkpoint["config"]
        config = KhatriVoiceConfig(**config_dict)
    else:
        raise ValueError("No config found in checkpoint")

    # Load model
    model = KhatriVoice(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Get stop tokens
    stop_tokens = [tokenizer.eos_id]
    if hasattr(tokenizer.vocab, 'end_id'):
        stop_tokens.append(tokenizer.vocab.end_id)

    generator = KhatriVoiceGenerator(model=model, tokenizer=tokenizer, device=device)

    return generator, stop_tokens


def generate_response(generator, prompt: str, max_new_tokens: int, temperature: float,
                      top_k: int, top_p: float, repetition_penalty: float,
                      stop_tokens: list) -> str:
    """Generate a response from the model."""
    outputs = generator.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=(temperature > 0),
        stop_tokens=stop_tokens,
    )
    return outputs[0] if outputs else ""


def interactive_chat(generator, args, stop_tokens: list):
    """Run interactive chat session."""
    print("\n" + "=" * 60)
    print("KhatriVoice Interactive Chat")
    print("=" * 60)
    print("Type 'quit' or 'exit' to end.")
    print("Type 'clear' to start new conversation.\n")

    conversation_history = []

    while True:
        try:
            user_input = input("User: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit']:
                print("\nGoodbye!")
                break

            if user_input.lower() == 'clear':
                conversation_history = []
                print("\nConversation cleared.\n")
                continue

            # Build prompt with history
            full_prompt = "\n".join(conversation_history) + "\n" if conversation_history else ""
            full_prompt += format_chat_prompt(user_input)

            print("Assistant: ", end="", flush=True)

            response = generate_response(
                generator=generator,
                prompt=full_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop_tokens=stop_tokens,
            )

            clean_response = extract_assistant_response(response)
            print(clean_response)

            # Update history
            conversation_history.append(f"User: {user_input}")
            conversation_history.append(f"Assistant: {clean_response}")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\n\nGoodbye!")
            break


def main():
    args = parse_args()

    # Random seed
    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Determine paths
    checkpoint_path = Path(args.checkpoint)
    if args.tokenizer:
        tokenizer_path = Path(args.tokenizer)
    else:
        tokenizer_path = checkpoint_path.parent / "tokenizer"
        if not tokenizer_path.exists():
            tokenizer_path = checkpoint_path.parent

    print(f"Loading model from: {checkpoint_path}")
    print(f"Loading tokenizer from: {tokenizer_path}")

    # Load generator
    generator, stop_tokens = load_generator(
        checkpoint_path=str(checkpoint_path),
        tokenizer_path=str(tokenizer_path),
        device=args.device,
    )

    print(f"Device: {generator.device}")
    print(f"Vocabulary size: {generator.tokenizer.vocab_size}")

    # Interactive mode
    if args.interactive:
        interactive_chat(generator, args, stop_tokens)
        return

    # Single prompt mode
    if args.prompt:
        prompt = format_chat_prompt(args.prompt) if args.chat else args.prompt

        print(f"\nPrompt: {args.prompt}")
        print("Generating response...")

        for i in range(args.num_sequences):
            response = generate_response(
                generator=generator,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop_tokens=stop_tokens,
            )

            clean_response = extract_assistant_response(response) if args.chat else response

            if args.num_sequences > 1:
                print(f"\n--- Response {i+1} ---")
            else:
                print()

            print(f"Assistant: {clean_response}")

    else:
        print("\nNo prompt. Use --prompt or --interactive")
        print("\nExample usage:")
        print("  python scripts/generate.py --checkpoint checkpoints/best.pt --chat --prompt 'Hello'")
        print("  python scripts/generate.py --checkpoint checkpoints/best.pt --interactive")


if __name__ == "__main__":
    main()
