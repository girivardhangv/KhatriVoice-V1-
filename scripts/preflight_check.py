#!/usr/bin/env python3
"""
Preflight diagnostic test for KhatriVoice training.

This script performs all checks BEFORE training to catch issues early:
1. Load dataset and show statistics
2. Train/load tokenizer
3. Print actual vocabulary size and special token IDs
4. Encode and decode a conversation
5. Create one training batch
6. Print input IDs and labels
7. Run one forward pass
8. Run one backward pass
9. Run one generation test
10. Check device consistency

Usage:
    python scripts/preflight_check.py --config configs/small.yaml
"""

import argparse
import sys
from pathlib import Path
from collections import Counter
import random

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from khatrivoice.config.model_config import KhatriVoiceConfig
from khatrivoice.model.khatrivoice import KhatriVoice
from khatrivoice.tokenizer.tokenizer import KhatriTokenizer
from khatrivoice.tokenizer.vocabulary import Vocabulary
from khatrivoice.data.dataset import KhatriDataset, ConversationDataset
from khatrivoice.data.collator import DataCollator
from khatrivoice.utils.device import get_device


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_dataset_quality(config: KhatriVoiceConfig) -> dict:
    """Test 1: Load dataset and analyze quality."""
    print_section("TEST 1: Dataset Quality Analysis")

    from khatrivoice.data.preprocessing import load_text_file, split_train_val_test

    data_path = Path(config.data_path)

    if not data_path.exists():
        print(f"x Data path not found: {data_path}")
        return {"passed": False, "reason": "data_path_not_found"}

    # Load data
    if data_path.is_file():
        text = load_text_file(data_path, clean=False)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
    else:
        print(f"x Data path is not a file: {data_path}")
        return {"passed": False, "reason": "not_a_file"}

    print(f"Total lines loaded: {len(lines)}")

    # Parse User/AI pairs
    user_prompts = []
    ai_responses = []
    for line in lines:
        if line.startswith('User:'):
            user_prompts.append(line[6:].strip())
        elif line.startswith('AI:'):
            ai_responses.append(line[4:].strip())

    unique_users = len(set(user_prompts))
    unique_ais = len(set(ai_responses))

    print(f"  User prompts: {len(user_prompts)} total, {unique_users} unique")
    print(f"  AI responses: {len(ai_responses)} total, {unique_ais} unique")

    # Calculate repetition
    if len(user_prompts) > 0:
        user_dup_pct = (1 - unique_users / len(user_prompts)) * 100
        ai_dup_pct = (1 - unique_ais / len(ai_responses)) * 100 if ai_responses else 0
        print(f"  User repetition: {user_dup_pct:.1f}%")
        print(f"  AI repetition: {ai_dup_pct:.1f}%")

        if user_dup_pct > 90:
            print(f"[WARN]  WARNING: User prompts are {user_dup_pct:.1f}% duplicates!")
            print("   Consider using a more diverse dataset.")

    # Sample conversations
    print(f"\n  Sample conversations (first 3):")
    for i, line in enumerate(lines[:6]):
        print(f"    {line[:80]}...")

    return {
        "passed": True,
        "total_lines": len(lines),
        "unique_users": unique_users,
        "unique_ais": unique_ais,
        "user_dup_pct": user_dup_pct if user_prompts else 0,
    }


def test_tokenizer(config: KhatriVoiceConfig, sample_texts: list) -> dict:
    """Test 2-4: Tokenizer training and vocabulary."""
    print_section("TEST 2-4: Tokenizer Analysis")

    # Create and train tokenizer
    print("Training tokenizer on sample data...")
    tokenizer = KhatriTokenizer()

    # Use a sample for quick test
    train_sample = sample_texts[:min(1000, len(sample_texts))]
    tokenizer.train(train_sample, vocab_size=config.vocab_size)

    actual_vocab_size = tokenizer.vocab_size
    print(f"\n  Requested vocab_size: {config.vocab_size}")
    print(f"  Actual vocab_size: {actual_vocab_size}")

    if actual_vocab_size < config.vocab_size * 0.1:
        print(f"[WARN]  WARNING: Actual vocab is less than 10% of requested!")
        print("   This usually means the corpus lacks diversity.")

    # Print special token IDs
    print(f"\n  Special token IDs:")
    print(f"    PAD: {tokenizer.pad_id} ({repr(tokenizer.vocab.pad_token)})")
    print(f"    UNK: {tokenizer.unk_id} ({repr(tokenizer.vocab.unk_token)})")
    print(f"    BOS: {tokenizer.bos_id} ({repr(tokenizer.vocab.bos_token)})")
    print(f"    EOS: {tokenizer.eos_id} ({repr(tokenizer.vocab.eos_token)})")

    # Check conversation tokens
    if hasattr(tokenizer.vocab, 'user_id'):
        print(f"    USER: {tokenizer.vocab.user_id} ({repr(tokenizer.vocab.user_token)})")
        print(f"    ASSISTANT: {tokenizer.vocab.assistant_id} ({repr(tokenizer.vocab.assistant_token)})")
        print(f"    END: {tokenizer.vocab.end_id} ({repr(tokenizer.vocab.end_token)})")

    # Test encode/decode
    print(f"\n  Testing encode/decode:")
    test_text = "Hello, how are you?"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"    Original: {test_text}")
    print(f"    Encoded: {encoded[:10]}...")
    print(f"    Decoded: {decoded}")

    return {
        "passed": True,
        "actual_vocab_size": actual_vocab_size,
        "requested_vocab_size": config.vocab_size,
        "tokenizer": tokenizer,
    }


def test_training_batch(config: KhatriVoiceConfig, tokenizer: KhatriTokenizer, texts: list) -> dict:
    """Test 5-6: Create training batch and inspect."""
    print_section("TEST 5-6: Training Batch Creation")

    # Use ConversationDataset if conversation mode
    print("Creating dataset...")
    dataset = ConversationDataset(
        tokenizer=tokenizer,
        texts=texts[:100],  # Small sample
        max_length=config.max_sequence_length,
        mask_user_tokens=True,
    )

    print(f"  Dataset size: {len(dataset)} samples")

    # Create dataloader
    collator = DataCollator(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collator)

    # Get one batch
    batch = next(iter(dataloader))

    print(f"\n  Batch shapes:")
    print(f"    input_ids: {batch['input_ids'].shape}")
    print(f"    labels: {batch['labels'].shape}")
    print(f"    attention_mask: {batch['attention_mask'].shape}")

    # Decode first sample
    print(f"\n  First sample decoded:")
    input_ids = batch['input_ids'][0]
    labels = batch['labels'][0]

    decoded_input = tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)
    print(f"    Input: {decoded_input[:100]}...")

    # Show labels (masked positions are -100)
    valid_labels = [l for l in labels.tolist() if l != -100]
    print(f"    Valid labels count: {len(valid_labels)}/{len(labels)}")
    print(f"    Sample labels: {labels[:20].tolist()}")

    return {
        "passed": True,
        "batch": batch,
        "dataset": dataset,
    }


def test_forward_pass(config: KhatriVoiceConfig, batch: dict, device: torch.device) -> dict:
    """Test 7: Forward pass."""
    print_section("TEST 7: Forward Pass")

    # Update config vocab_size to match tokenizer
    print(f"Creating model with vocab_size={config.vocab_size}...")

    model = KhatriVoice(config)
    model = model.to(device)
    model.eval()

    print(f"  Device: {device}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Move batch to device
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    # Forward pass
    print(f"  Running forward pass...")
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)

    loss = outputs['loss']
    logits = outputs['logits']

    print(f"    Loss: {loss.item():.4f}")
    print(f"    Logits shape: {logits.shape}")

    if torch.isnan(loss) or torch.isinf(loss):
        print(f"x NaN/Inf detected in loss!")
        return {"passed": False, "reason": "nan_loss"}

    print(f"v Forward pass successful")
    return {"passed": True, "loss": loss.item(), "model": model}


def test_backward_pass(model: nn.Module, batch: dict, device: torch.device) -> dict:
    """Test 8: Backward pass."""
    print_section("TEST 8: Backward Pass")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    # Forward
    outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
    loss = outputs['loss'] / 4  # Simulate gradient accumulation

    # Backward
    loss.backward()

    # Check gradients
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    print(f"  Gradient norm: {total_norm:.4f}")

    if total_norm > 1000:
        print(f"[WARN]  WARNING: Large gradient norm ({total_norm:.1f}). Consider gradient clipping.")

    print(f"v Backward pass successful")
    return {"passed": True, "grad_norm": total_norm}


def test_generation(model: KhatriVoice, tokenizer: KhatriTokenizer, device: torch.device) -> dict:
    """Test 9: Generation test."""
    print_section("TEST 9: Generation Test")

    model.eval()
    from khatrivoice.inference.generator import KhatriVoiceGenerator, generate_chat_response

    generator = KhatriVoiceGenerator(model=model, tokenizer=tokenizer, device=device)

    test_prompts = ["Hi", "What is Python?", "Hello"]

    print(f"\n  Testing generation:")
    for prompt in test_prompts:
        try:
            # Format with conversation tokens if available
            if hasattr(tokenizer.vocab, 'user_token'):
                formatted = f"{tokenizer.vocab.user_token}\n{prompt}\n{tokenizer.vocab.assistant_token}\n"
            else:
                formatted = prompt

            outputs = generator.generate(
                prompt=formatted,
                max_new_tokens=30,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

            output = outputs[0] if outputs else "(no output)"
            print(f"\n    Prompt: {prompt}")
            print(f"    Output: {output[:100]}...")

            # Check for issues
            words = output.split()
            if len(words) > 5:
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < 0.3:
                    print(f"    [WARN] Low diversity: {unique_ratio:.2%} unique words")

        except Exception as e:
            print(f"    x Generation failed: {e}")
            return {"passed": False, "reason": str(e)}

    print(f"\nv Generation test complete")
    return {"passed": True}


def test_device_consistency(model: KhatriVoice, device: torch.device) -> dict:
    """Test 10: Check all tensors are on correct device."""
    print_section("TEST 10: Device Consistency")

    issues = []

    # Check model parameters
    for name, param in model.named_parameters():
        if param.device != device:
            issues.append(f"Parameter {name} on {param.device}, expected {device}")

    # Check buffers (like causal mask)
    for name, buffer in model.named_buffers():
        if buffer.device != device:
            issues.append(f"Buffer {name} on {buffer.device}, expected {device}")

    if issues:
        print(f"x Device issues found:")
        for issue in issues[:5]:
            print(f"  - {issue}")
        return {"passed": False, "issues": issues}
    else:
        print(f"v All tensors on correct device ({device})")
        return {"passed": True}


def main():
    parser = argparse.ArgumentParser(description="Preflight check for KhatriVoice")
    parser.add_argument("--config", type=str, default="configs/tiny.yaml")
    parser.add_argument("--data", type=str, default=None, help="Override data path")
    args = parser.parse_args()

    print("=" * 60)
    print("  KHATRIVOICE PREFLIGHT DIAGNOSTIC")
    print("=" * 60)

    # Load config
    config = KhatriVoiceConfig.load(args.config)
    if args.data:
        config.data_path = args.data

    print(f"\nConfig: {args.config}")
    print(f"Data path: {config.data_path}")
    print(f"Target vocab_size: {config.vocab_size}")
    print(f"Device: {config.device}")

    # Get device
    device = get_device(config.device)
    print(f"Using device: {device}")

    # Load sample texts for testing
    from khatrivoice.data.preprocessing import load_text_file, split_train_val_test

    data_path = Path(config.data_path)
    if data_path.exists():
        text = load_text_file(data_path, clean=False)
        texts = [line.strip() for line in text.split('\n') if line.strip()]
    else:
        print(f"Creating sample data...")
        texts = [f"User: Hello\nAI: Hi there! How can I help?" for _ in range(100)]

    # Run all tests
    results = {}

    # Test 1: Dataset quality
    results['dataset'] = test_dataset_quality(config)
    if not results['dataset']['passed']:
        print("\nx STOPPING: Dataset test failed")
        return 1

    # Test 2-4: Tokenizer
    results['tokenizer'] = test_tokenizer(config, texts)
    tokenizer = results['tokenizer']['tokenizer']

    # Update config to match actual vocab size
    if config.vocab_size != tokenizer.vocab_size:
        print(f"\n Updating config.vocab_size: {config.vocab_size} -> {tokenizer.vocab_size}")
        config.vocab_size = tokenizer.vocab_size

    # Test 5-6: Training batch
    results['batch'] = test_training_batch(config, tokenizer, texts)
    batch = results['batch']['batch']

    # Test 7: Forward pass
    results['forward'] = test_forward_pass(config, batch, device)
    if not results['forward']['passed']:
        print("\nx STOPPING: Forward pass failed")
        return 1
    model = results['forward']['model']

    # Test 8: Backward pass
    results['backward'] = test_backward_pass(model, batch, device)

    # Test 9: Generation
    results['generation'] = test_generation(model, tokenizer, device)

    # Test 10: Device consistency
    results['device'] = test_device_consistency(model, device)

    # Summary
    print_section("SUMMARY")

    passed = sum(1 for r in results.values() if r.get('passed', False))
    total = len(results)

    print(f"\n  Tests passed: {passed}/{total}")

    for name, result in results.items():
        status = "v" if result.get('passed', False) else "x"
        print(f"    {status} {name}")

    if passed == total:
        print("\n[PASS] ALL TESTS PASSED - Ready for training!")
        print("\nRecommended training command:")
        print(f"  python scripts/train.py --config {args.config} --conversation-mode")
        return 0
    else:
        print("\nx SOME TESTS FAILED - Fix issues before training")
        return 1


if __name__ == "__main__":
    sys.exit(main())
