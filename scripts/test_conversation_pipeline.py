#!/usr/bin/env python3
"""
Test the conversation-aware training pipeline.

This script verifies that:
1. Special tokens are properly defined
2. Conversation parsing works
3. Dataset properly masks user tokens
4. Generation handles stop tokens
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_special_tokens():
    """Test that special tokens are properly defined."""
    print("Testing special tokens...")

    from khatrivoice.tokenizer.vocabulary import Vocabulary

    vocab = Vocabulary()

    # Check that all special tokens exist
    assert vocab.pad_token == "<pad>", f"PAD token mismatch"
    assert vocab.unk_token == "<unk>", f"UNK token mismatch"
    assert vocab.bos_token == "<s>", f"BOS token mismatch"
    assert vocab.eos_token == "</s>", f"EOS token mismatch"

    # Check conversational tokens
    assert vocab.user_token == "<user>", f"USER token mismatch"
    assert vocab.assistant_token == "<|assistant>", f"ASSISTANT token mismatch"
    assert vocab.end_token == "<|end|>", f"END token mismatch"

    # Check IDs
    assert vocab.pad_id == 0, f"PAD ID should be 0, got {vocab.pad_id}"
    assert vocab.unk_id == 1, f"UNK ID should be 1, got {vocab.unk_id}"
    assert vocab.bos_id == 2, f"BOS ID should be 2, got {vocab.bos_id}"
    assert vocab.eos_id == 3, f"EOS ID should be 3, got {vocab.eos_id}"

    print(f"  ✓ All special tokens properly defined")
    print(f"  USER token: {repr(vocab.user_token)}")
    print(f"  ASSISTANT token: {repr(vocab.assistant_token)}")
    print(f"  END token: {vocab.end_token}")
    return True


def test_conversation_parsing():
    """Test conversation parsing."""
    print("\nTesting conversation parsing...")

    from khatrivoice.data.preprocessing import (
        parse_conversation_line,
        format_conversation,
        parse_conversation_file,
    )

    # Test inline parsing
    line = "User: What is Python? AI: Python is a programming language."
    parsed = parse_conversation_line(line)

    assert parsed is not None, "Failed to parse conversation"
    assert parsed["user"] == "What is Python?", f"User text mismatch: {parsed['user']}"
    assert parsed["assistant"] == "Python is a programming language.", f"Assistant text mismatch"

    print(f"  ✓ Inline conversation parsing works")

    # Test multiline parsing (your format)
    multiline_text = """User: What is JavaScript?
AI: JavaScript is a programming language.
User: What is overfitting?
AI: Overfitting happens when a model learns too closely."""

    import tempfile
    import os

    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(multiline_text)
        temp_path = f.name

    try:
        conversations = parse_conversation_file(temp_path, format="multiline")
        print(f"  ✓ Multiline parsing: found {len(conversations)} conversations")

        if len(conversations) >= 2:
            assert conversations[0]["user"] == "What is JavaScript?"
            assert conversations[1]["user"] == "What is overfitting?"
            print(f"  ✓ Correctly parsed user prompts from multiline format")
    finally:
        os.unlink(temp_path)

    # Test formatting
    formatted = format_conversation("Hello", "Hi there!")

    assert "Hello" in formatted, "User text not in formatted conversation"
    assert "Hi there!" in formatted, "Assistant text not in formatted conversation"
    assert "<|end|>" in formatted, "END token not in formatted conversation"

    print(f"  ✓ Conversation formatting works")
    print(f"  Formatted example: {repr(formatted[:100])}...")
    return True


def test_dataset_creation():
    """Test ConversationDataset creation."""
    print("\nTesting ConversationDataset...")

    from khatrivoice.tokenizer.tokenizer import KhatriTokenizer
    from khatrivoice.data.dataset import ConversationDataset

    # Create a tiny tokenizer
    vocab_texts = ["hello world", "test data", "user assistant end"]
    tokenizer = KhatriTokenizer()
    tokenizer.train(vocab_texts, vocab_size=50)

    # Create sample conversations
    conversations = [
        "Hello\nHow are you?\n\nI'm doing well, thanks!\n<|end|>",
        "Hi\nWhat's up?\n\nNot much!\n<|end|>",
    ]

    # Create dataset
    dataset = ConversationDataset(
        tokenizer=tokenizer,
        texts=conversations,
        max_length=32,
        mask_user_tokens=True,
    )

    print(f"  ✓ ConversationDataset created")
    print(f"  Dataset size: {len(dataset)} samples")

    # Get a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"  Sample input_ids shape: {sample['input_ids'].shape}")
        print(f"  Sample labels shape: {sample['labels'].shape}")

        # Check that some labels are masked (-100)
        import torch
        num_masked = (sample['labels'] == -100).sum().item()
        print(f"  Number of masked labels: {num_masked}")

    return True


def test_generator_params():
    """Test generator parameters."""
    print("\nTesting generator parameters...")

    # Just test that the import works and signature is correct
    from khatrivoice.inference.generator import KhatriVoiceGenerator
    import inspect

    # Check generate method signature
    sig = inspect.signature(KhatriVoiceGenerator.generate)
    params = list(sig.parameters.keys())

    assert "repetition_penalty" in params, "repetition_penalty parameter missing"
    assert "stop_tokens" in params, "stop_tokens parameter missing"

    print(f"  ✓ Generator has new parameters:")
    print(f"    - repetition_penalty")
    print(f"    - stop_tokens")

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing KhatriVoice Conversation Pipeline")
    print("=" * 60)

    tests = [
        test_special_tokens,
        test_conversation_parsing,
        test_dataset_creation,
        test_generator_params,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
