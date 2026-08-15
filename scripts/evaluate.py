#!/usr/bin/env python3
"""
Evaluate KhatriVoice language model.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --data data/test.txt
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --data data/test.txt --output results.json

This script evaluates a trained KhatriVoice model on test data.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from khatrivoice.config.model_config import KhatriVoiceConfig
from khatrivoice.model.khatrivoice import KhatriVoice
from khatrivoice.tokenizer.tokenizer import KhatriTokenizer
from khatrivoice.data.dataset import KhatriDataset
from khatrivoice.data.collator import DataCollator
from khatrivoice.inference.generator import KhatriVoiceGenerator
from khatrivoice.utils.device import get_device


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate KhatriVoice model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on test file
  python scripts/evaluate.py --checkpoint checkpoints/best.pt --data data/test.txt

  # Save results to JSON
  python scripts/evaluate.py --checkpoint checkpoints/best.pt --data data/test.txt --output results.json

  # Compute generation samples
  python scripts/evaluate.py --checkpoint checkpoints/best.pt --data data/test.txt --generate-samples 10
        """,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to test data (file or directory)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: loaded from checkpoint)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run on",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--generate-samples",
        type=int,
        default=0,
        help="Number of generation samples to produce",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--generation-length",
        type=int,
        default=50,
        help="Number of tokens to generate for samples",
    )

    return parser.parse_args()


def load_test_data(data_path: str) -> List[str]:
    """
    Load test data from file or directory.

    Args:
        data_path: Path to test data

    Returns:
        List of text samples
    """
    data_path = Path(data_path)

    if data_path.is_file():
        # Load single file
        with open(data_path, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
        return [line.strip() for line in lines if line.strip()]

    elif data_path.is_dir():
        # Load all .txt files from directory
        texts = []
        for file_path in sorted(data_path.glob("*.txt")):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                texts.append(content)
        return texts

    else:
        raise FileNotFoundError(f"Data path not found: {data_path}")


def evaluate_perplexity(
    model: KhatriVoice,
    tokenizer: KhatriTokenizer,
    texts: List[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate model perplexity on texts.

    Args:
        model: KhatriVoice model
        tokenizer: Tokenizer
        texts: List of text samples
        batch_size: Batch size
        max_length: Maximum sequence length
        device: Device

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\nEvaluating perplexity on {len(texts)} samples...")

    # Create dataset
    dataset = KhatriDataset(
        tokenizer=tokenizer,
        texts=texts,
        max_length=max_length,
    )

    # Create data loader
    collator = DataCollator(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    with torch.no_grad():
        from tqdm import tqdm
        for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask")

            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs["loss"]

            # Count valid tokens
            valid_mask = labels != -100
            num_tokens = valid_mask.sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            num_batches += 1

    # Compute metrics
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "num_samples": len(texts),
        "num_batches": num_batches,
        "total_tokens": total_tokens,
    }


def generate_samples(
    model: KhatriVoice,
    tokenizer: KhatriTokenizer,
    texts: List[str],
    num_samples: int,
    generation_length: int,
    device: torch.device,
) -> List[Dict[str, str]]:
    """
    Generate sample texts.

    Args:
        model: KhatriVoice model
        tokenizer: Tokenizer
        texts: List of prompt texts
        num_samples: Number of samples to generate
        generation_length: Length of generation
        device: Device

    Returns:
        List of dictionaries with prompt and generated text
    """
    print(f"\nGenerating {num_samples} samples...")

    generator = KhatriVoiceGenerator(model=model, tokenizer=tokenizer, device=device)

    samples = []
    import random
    prompts = random.sample(texts, min(num_samples, len(texts)))

    for i, prompt in enumerate(prompts):
        # Truncate prompt if too long
        if len(prompt) > 100:
            prompt = prompt[:100]

        # Generate
        try:
            generated = generator.generate(
                prompt=prompt,
                max_new_tokens=generation_length,
                temperature=0.8,
                top_p=0.95,
                do_sample=True,
            )[0]

            samples.append({
                "prompt": prompt,
                "generated": generated,
            })

            print(f"\n[{i + 1}/{num_samples}]")
            print(f"  Prompt: {prompt[:50]}...")
            print(f"  Generated: {generated[:100]}...")

        except Exception as e:
            print(f"\n[{i + 1}] Generation failed: {e}")

    return samples


def main():
    """Main evaluation function."""
    args = parse_args()

    print("=" * 60)
    print("KhatriVoice Evaluation")
    print("=" * 60)
    print()

    # Load model
    print("Loading model...")
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "config" in checkpoint:
        config = KhatriVoiceConfig.from_dict(checkpoint["config"])
    else:
        raise ValueError("Checkpoint does not contain config")

    model = KhatriVoice(config)
    model.load_state_dict(checkpoint["model_state_dict"])

    device = get_device(args.device)
    model = model.to(device)
    model.eval()

    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device: {device}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    if args.tokenizer:
        tokenizer = KhatriTokenizer.load(args.tokenizer)
    else:
        # Try checkpoint directory
        checkpoint_tokenizer = checkpoint_path.parent / "tokenizer"
        if checkpoint_tokenizer.exists():
            tokenizer = KhatriTokenizer.load(checkpoint_tokenizer)
        else:
            # Try default path
            default_tokenizer = Path("data/processed/tokenizer")
            if default_tokenizer.exists():
                tokenizer = KhatriTokenizer.load(default_tokenizer)
            else:
                raise FileNotFoundError(
                    f"Tokenizer not found. Please specify --tokenizer path"
                )

    print(f"  Vocabulary size: {tokenizer.vocab_size}")

    # Load test data
    print("\nLoading test data...")
    texts = load_test_data(args.data)
    print(f"  Loaded {len(texts)} samples")

    if len(texts) == 0:
        print("Error: No test data found")
        sys.exit(1)

    # Evaluate perplexity
    metrics = evaluate_perplexity(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Perplexity: {metrics['perplexity']:.2f}")
    print(f"Total tokens: {metrics['total_tokens']:,}")
    print(f"Num samples: {metrics['num_samples']}")
    print()

    # Generate samples if requested
    generation_samples = []
    if args.generate_samples > 0:
        generation_samples = generate_samples(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            num_samples=args.generate_samples,
            generation_length=args.generation_length,
            device=device,
        )

    # Prepare results
    results = {
        "checkpoint": str(args.checkpoint),
        "data_path": str(args.data),
        "device": str(device),
        "perplexity_metrics": metrics,
    }

    if generation_samples:
        results["generation_samples"] = generation_samples

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")

    print()
    print("=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
