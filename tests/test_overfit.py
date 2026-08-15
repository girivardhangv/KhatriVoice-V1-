#!/usr/bin/env python3
"""
Test that KhatriVoice can overfit a tiny dataset.

This is a critical test for any machine learning model:
If the model cannot overfit a tiny dataset (memorize it),
there's likely a bug in the model or training code.

This script trains the model on a very small, repetitive dataset
and verifies that the loss decreases to near-zero.

Run with: python tests/test_overfit.py
"""

import sys
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from khatrivoice.config.model_config import get_tiny_config
from khatrivoice.model.khatrivoice import KhatriVoice
from khatrivoice.tokenizer.tokenizer import KhatriTokenizer
from khatrivoice.data.dataset import KhatriDataset
from khatrivoice.data.collator import DataCollator
from khatrivoice.data.preprocessing import create_tiny_dataset
from khatrivoice.training.optimizer import create_optimizer, create_cosine_scheduler
from khatrivoice.utils.device import get_device
from khatrivoice.utils.seed import set_seed


def test_overfit_tiny_dataset(
    num_steps: int = 500,
    target_loss: float = 0.1,
    learning_rate: float = 1e-3,
):
    """
    Test that the model can overfit a tiny dataset.

    Args:
        num_steps: Number of training steps
        target_loss: Target loss to achieve (should go below this)
        learning_rate: Learning rate for training
    """
    print("=" * 60)
    print("KhatriVoice Overfit Test")
    print("=" * 60)
    print()
    print("This test verifies the model can memorize a tiny dataset.")
    print("If this fails, there may be a bug in the model or training.")
    print()

    # Set seed for reproducibility
    set_seed(42)

    # Get device
    device = get_device("auto")
    print(f"Device: {device}")

    # Create config (use tiny config for faster testing)
    config = get_tiny_config()
    config.learning_rate = learning_rate

    print(f"Model config:")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_layers: {config.num_layers}")
    print(f"  num_heads: {config.num_attention_heads}")
    print(f"  Parameters: {config.total_parameters:,}")

    # Create tokenizer
    print("\nCreating tokenizer...")
    tokenizer = KhatriTokenizer(lowercase=True)

    # Use very small, repetitive dataset
    tiny_texts = [
        "hello world hello world hello world",
        "hello world hello world hello world",
        "hello world hello world hello world",
        "hello world hello world hello world",
        "hello world hello world hello world",
    ] * 10  # Repeat for more training samples

    tokenizer.train(tiny_texts, vocab_size=50)
    print(f"  Vocabulary size: {tokenizer.vocab_size}")

    # IMPORTANT: Update config vocab_size to match tokenizer
    config.vocab_size = tokenizer.vocab_size

    # Create dataset
    print("\nCreating dataset...")
    dataset = KhatriDataset(
        tokenizer=tokenizer,
        texts=tiny_texts,
        max_length=16,  # Short sequences
    )

    print(f"  Dataset size: {len(dataset)} samples")

    # Create data loader
    collator = DataCollator(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collator,
    )

    # Create model
    print("\nCreating model...")
    model = KhatriVoice(config).to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Create optimizer
    optimizer = create_optimizer(model, learning_rate=learning_rate)

    # Training loop
    print(f"\nTraining for {num_steps} steps...")
    print("-" * 60)

    losses = []
    step = 0

    while step < num_steps:
        for batch in dataloader:
            if step >= num_steps:
                break

            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # Track loss
            loss_value = loss.item()
            losses.append(loss_value)

            # Print progress
            if (step + 1) % 50 == 0 or step == 0:
                avg_loss = sum(losses[-10:]) / min(len(losses), 10)
                print(f"Step {step + 1:4d} | Loss: {loss_value:.4f} | Avg Loss: {avg_loss:.4f}")

            step += 1

    print("-" * 60)

    # Analyze results
    print("\nAnalyzing results...")

    initial_loss = losses[0]
    final_loss = losses[-1]
    min_loss = min(losses)

    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Minimum loss: {min_loss:.4f}")

    # Compute perplexity
    initial_ppl = torch.exp(torch.tensor(initial_loss)).item()
    final_ppl = torch.exp(torch.tensor(final_loss)).item()
    min_ppl = torch.exp(torch.tensor(min_loss)).item()

    print(f"  Initial perplexity: {initial_ppl:.2f}")
    print(f"  Final perplexity: {final_ppl:.2f}")
    print(f"  Minimum perplexity: {min_ppl:.2f}")

    # Check if model can generate the training data
    print("\nTesting generation...")
    model.eval()

    with torch.no_grad():
        # Start with "hello"
        start_text = "hello"
        input_ids = torch.tensor([tokenizer.encode(start_text)], device=device)

        # Generate a few tokens
        generated = input_ids.clone()
        for _ in range(10):
            outputs = model(generated, use_cache=True)
            logits = outputs["logits"]
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

        # Decode
        generated_text = tokenizer.decode(generated[0].tolist())
        print(f"  Generated: '{generated_text}'")

    # Check if loss decreased enough
    print("\n" + "=" * 60)
    if min_loss < target_loss:
        print(f"SUCCESS: Model successfully overfit the tiny dataset!")
        print(f"  Minimum loss {min_loss:.4f} < target {target_loss:.4f}")
        print("=" * 60)
        return True
    else:
        print(f"FAILED: Model did not overfit the dataset.")
        print(f"  Minimum loss {min_loss:.4f} >= target {target_loss:.4f}")
        print("  This suggests there may be a bug in the model or training code.")
        print("=" * 60)
        return False


def main():
    """Run the overfit test."""
    success = test_overfit_tiny_dataset(
        num_steps=500,
        target_loss=0.5,  # Target loss (should go below this)
        learning_rate=1e-3,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
