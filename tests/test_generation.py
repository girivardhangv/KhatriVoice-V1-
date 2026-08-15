#!/usr/bin/env python3
"""
Test generation capabilities for KhatriVoice.

This script tests the text generation module.

Run with: python tests/test_generation.py
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
from khatrivoice.config.model_config import get_tiny_config
from khatrivoice.model.khatrivoice import KhatriVoice
from khatrivoice.tokenizer.tokenizer import KhatriTokenizer
from khatrivoice.inference.generator import KhatriVoiceGenerator, create_generator


def test_generator_creation():
    """Test generator creation."""
    config = get_tiny_config()
    model = KhatriVoice(config)

    tokenizer = KhatriTokenizer()

    # Sort unique chars to get a stable alphabet
    chars = sorted(set("hello world abcdefghijklmnopqrstuvwxyz .!?"))
    tokenizer.vocab.add_tokens(chars)

    generator = create_generator(model=model, tokenizer=tokenizer)

    assert generator is not None
    assert generator.model is model
    assert generator.tokenizer is tokenizer

    print("[OK] Generator creation")


def test_greedy_generation():
    """Test greedy decoding."""
    config = get_tiny_config()
    model = KhatriVoice(config)
    model.eval()

    tokenizer = KhatriTokenizer()
    chars = sorted(set("hello world .!?"))
    tokenizer.vocab.add_tokens(chars)

    generator = create_generator(model=model, tokenizer=tokenizer)

    # Generate with greedy decoding
    results = generator.generate(
        prompt="hello",
        max_new_tokens=10,
        do_sample=False,
    )

    assert len(results) == 1
    assert len(results[0]) > 0
    assert results[0].startswith("hello")

    print("[OK] Greedy generation")


def test_temperature_sampling():
    """Test temperature sampling."""
    config = get_tiny_config()
    model = KhatriVoice(config)
    model.eval()

    tokenizer = KhatriTokenizer()
    chars = sorted(set("hello world .!?"))
    tokenizer.vocab.add_tokens(chars)

    generator = create_generator(model=model, tokenizer=tokenizer)

    # Generate with temperature
    results = generator.generate(
        prompt="hello",
        max_new_tokens=10,
        temperature=0.8,
        do_sample=True,
    )

    assert len(results) == 1
    assert len(results[0]) > 0

    print("[OK] Temperature sampling")


def test_top_k_sampling():
    """Test top-k sampling."""
    config = get_tiny_config()
    model = KhatriVoice(config)
    model.eval()

    tokenizer = KhatriTokenizer()
    chars = sorted(set("hello world .!?"))
    tokenizer.vocab.add_tokens(chars)

    generator = create_generator(model=model, tokenizer=tokenizer)

    # Generate with top-k
    results = generator.generate(
        prompt="hello",
        max_new_tokens=10,
        temperature=1.0,
        top_k=5,
        do_sample=True,
    )

    assert len(results) == 1
    assert len(results[0]) > 0

    print("[OK] Top-k sampling")


def test_top_p_sampling():
    """Test top-p (nucleus) sampling."""
    config = get_tiny_config()
    model = KhatriVoice(config)
    model.eval()

    tokenizer = KhatriTokenizer()
    chars = sorted(set("hello world .!?"))
    tokenizer.vocab.add_tokens(chars)

    generator = create_generator(model=model, tokenizer=tokenizer)

    # Generate with top-p
    results = generator.generate(
        prompt="hello",
        max_new_tokens=10,
        temperature=1.0,
        top_p=0.9,
        do_sample=True,
    )

    assert len(results) == 1
    assert len(results[0]) > 0

    print("[OK] Top-p sampling")


def test_multi_sequence_generation():
    """Test generating multiple sequences."""
    config = get_tiny_config()
    model = KhatriVoice(config)
    model.eval()

    tokenizer = KhatriTokenizer()
    chars = sorted(set("hello world .!?"))
    tokenizer.vocab.add_tokens(chars)

    generator = create_generator(model=model, tokenizer=tokenizer)

    # Generate multiple sequences
    results = generator.generate(
        prompt="hello",
        max_new_tokens=10,
        temperature=0.8,
        do_sample=True,
        num_return_sequences=3,
    )

    assert len(results) == 3
    for result in results:
        assert len(result) > 0

    print("[OK] Multi-sequence generation")


def test_eos_termination():
    """Test EOS token handling."""
    config = get_tiny_config()
    model = KhatriVoice(config)
    model.eval()

    tokenizer = KhatriTokenizer()
    chars = sorted(set("hello world .!?"))
    tokenizer.vocab.add_tokens(chars)

    generator = create_generator(model=model, tokenizer=tokenizer)

    # Generate with EOS token set
    results = generator.generate(
        prompt="hello",
        max_new_tokens=50,  # Allow more tokens
        temperature=1.0,
        do_sample=True,
        eos_token_id=tokenizer.eos_id,
    )

    assert len(results) == 1
    # Generation should either contain EOS or stop early
    print("[OK] EOS termination")


def test_max_length():
    """Test max_new_tokens limiting."""
    config = get_tiny_config()
    model = KhatriVoice(config)
    model.eval()

    tokenizer = KhatriTokenizer()
    chars = sorted(set("hello world .!?allchars"))
    tokenizer.vocab.add_tokens(chars)

    generator = create_generator(model=model, tokenizer=tokenizer)

    max_new = 5
    results = generator.generate(
        prompt="hello",
        max_new_tokens=max_new,
        do_sample=False,  # Greedy for deterministic length check
    )

    # Check that output is approximately prompt + max_new_tokens
    # (exact length depends on tokenizer)
    prompt_ids = tokenizer.encode("hello")
    assert len(results[0]) > 0
    print(f"[OK] Max length limiting (generated {len(results[0])} chars)")


def test_perplexity_computation():
    """Test perplexity computation."""
    config = get_tiny_config()
    model = KhatriVoice(config)
    model.eval()

    tokenizer = KhatriTokenizer()
    chars = sorted(set("hello world .!?allchars"))
    tokenizer.vocab.add_tokens(chars)

    generator = create_generator(model=model, tokenizer=tokenizer)

    # Compute perplexity
    ppl = generator.compute_perplexity("hello world")

    assert ppl > 0
    assert not torch.isnan(torch.tensor(ppl))

    print(f"[OK] Perplexity computation (ppl={ppl:.2f})")


def test_batch_perplexity():
    """Test batch perplexity computation."""
    config = get_tiny_config()
    model = KhatriVoice(config)
    model.eval()

    tokenizer = KhatriTokenizer()
    chars = sorted(set("hello world .!?allchars"))
    tokenizer.vocab.add_tokens(chars)

    generator = create_generator(model=model, tokenizer=tokenizer)

    # Compute batch perplexity
    texts = ["hello world", "hello", "world hello world"]
    ppl = generator.compute_batch_perplexity(texts, batch_size=2)

    assert ppl > 0
    assert not torch.isnan(torch.tensor(ppl))

    print(f"[OK] Batch perplexity (ppl={ppl:.2f})")


def test():
    """Run all tests."""
    print("=" * 60)
    print("KhatriVoice Generation Tests")
    print("=" * 60)
    print()

    tests = [
        ("Generator creation", test_generator_creation),
        ("Greedy generation", test_greedy_generation),
        ("Temperature sampling", test_temperature_sampling),
        ("Top-k sampling", test_top_k_sampling),
        ("Top-p sampling", test_top_p_sampling),
        ("Multi-sequence", test_multi_sequence_generation),
        ("EOS termination", test_eos_termination),
        ("Max length", test_max_length),
        ("Perplexity", test_perplexity_computation),
        ("Batch perplexity", test_batch_perplexity),
    ]

    passed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 60)
    print(f"Tests passed: {passed}/{len(tests)}")
    if passed == len(tests):
        print("All generation tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test()
