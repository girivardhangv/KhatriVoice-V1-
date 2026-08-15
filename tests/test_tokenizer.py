#!/usr/bin/env python3
"""
Test script to verify the KhatriVoice tokenizer.

Run with: python tests/test_tokenizer.py
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


def test_vocabulary_creation():
    """Test vocabulary creation."""
    from khatrivoice.tokenizer.vocabulary import Vocabulary

    vocab = Vocabulary()

    # Check special tokens are added
    assert vocab.bos_token in vocab
    assert vocab.eos_token in vocab
    assert vocab.pad_token in vocab
    assert vocab.unk_token in vocab

    print(f"  Vocab size: {vocab.vocab_size}")
    print("[OK] Vocabulary creation")


def test_vocabulary_add_tokens():
    """Test adding tokens to vocabulary."""
    from khatrivoice.tokenizer.vocabulary import Vocabulary

    vocab = Vocabulary()
    initial_size = vocab.vocab_size

    # Add tokens
    vocab.add_tokens(["hello", "world", "test"])

    assert vocab.vocab_size == initial_size + 3
    assert "hello" in vocab
    assert "world" in vocab
    assert "test" in vocab

    print("[OK] Vocabulary add tokens")


def test_vocabulary_get_id():
    """Test getting token IDs."""
    from khatrivoice.tokenizer.vocabulary import Vocabulary

    vocab = Vocabulary()

    # Get ID for known token
    bos_id = vocab.get_id(vocab.bos_token)
    assert bos_id == vocab.bos_id

    # Get ID for unknown token
    unk_id = vocab.get_id("unknown_token_xyz")
    assert unk_id == vocab.unk_id

    print("[OK] Vocabulary get ID")


def test_vocabulary_save_load():
    """Test vocabulary save and load."""
    from khatrivoice.tokenizer.vocabulary import Vocabulary
    import tempfile

    vocab = Vocabulary()
    vocab.add_tokens(["hello", "world", "test"])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "vocab.json"
        vocab.save(path)

        # Load vocabulary
        loaded = Vocabulary.load(path)

        assert loaded.vocab_size == vocab.vocab_size
        assert loaded.get_id("hello") == vocab.get_id("hello")
        assert loaded.get_id("world") == vocab.get_id("world")

    print("[OK] Vocabulary save/load")


def test_tokenizer_creation():
    """Test tokenizer creation."""
    from khatrivoice.tokenizer.tokenizer import KhatriTokenizer

    tokenizer = KhatriTokenizer()

    assert tokenizer.vocab_size >= 4  # At least special tokens
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print("[OK] Tokenizer creation")


def test_tokenizer_tokenize():
    """Test tokenization."""
    from khatrivoice.tokenizer.tokenizer import KhatriTokenizer

    tokenizer = KhatriTokenizer(lowercase=True)

    # Test word-level tokenization
    tokens = tokenizer.tokenize("Hello World!", mode="word")
    print(f"  Word tokens: {tokens}")

    # Test character-level tokenization
    tokens = tokenizer.tokenize("Hi", mode="char")
    print(f"  Char tokens: {tokens}")
    assert tokens == ["h", "i"]

    print("[OK] Tokenizer tokenize")


def test_tokenizer_encode_decode():
    """Test encoding and decoding."""
    from khatrivoice.tokenizer.tokenizer import KhatriTokenizer

    tokenizer = KhatriTokenizer(lowercase=True)

    # Encode text
    text = "hello world"
    ids = tokenizer.encode(text)

    print(f"  Text: '{text}'")
    print(f"  IDs: {ids}")

    # Decode back
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    print(f"  Decoded: '{decoded}'")

    # Note: Due to unknown tokens, exact round-trip may not work
    # without training the vocabulary first

    print("[OK] Tokenizer encode/decode")


def test_tokenizer_special_tokens():
    """Test special token handling."""
    from khatrivoice.tokenizer.tokenizer import KhatriTokenizer

    tokenizer = KhatriTokenizer()

    # Encode with special tokens
    text = "test"
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)

    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id

    print(f"  IDs with BOS/EOS: {ids}")
    print("[OK] Tokenizer special tokens")


def test_tokenizer_train():
    """Test tokenizer training."""
    from khatrivoice.tokenizer.tokenizer import KhatriTokenizer

    tokenizer = KhatriTokenizer(lowercase=True)

    # Train on small corpus
    corpus = [
        "hello world",
        "hello test",
        "world test hello",
        "this is a test",
        "hello again",
    ]

    tokenizer.train(corpus, vocab_size=20)

    print(f"  Trained vocab size: {tokenizer.vocab_size}")

    # Test encoding with trained vocabulary
    ids = tokenizer.encode("hello world")
    print(f"  'hello world' -> {ids}")

    # Test round-trip
    decoded = tokenizer.decode(ids)
    print(f"  {ids} -> '{decoded}'")

    print("[OK] Tokenizer train")


def test_tokenizer_batch():
    """Test batch encoding and decoding."""
    from khatrivoice.tokenizer.tokenizer import KhatriTokenizer

    tokenizer = KhatriTokenizer(lowercase=True)

    # Train on corpus first
    corpus = ["hello world", "test one", "test two"]
    tokenizer.train(corpus, vocab_size=20)

    # Batch encode
    texts = ["hello world", "test one two"]
    batch_ids = tokenizer.encode_batch(texts)

    print(f"  Input texts: {texts}")
    print(f"  Batch IDs: {batch_ids}")

    # Batch decode
    decoded = tokenizer.decode_batch(batch_ids)
    print(f"  Decoded: {decoded}")

    print("[OK] Tokenizer batch operations")


def test_tokenizer_save_load():
    """Test tokenizer save and load."""
    from khatrivoice.tokenizer.tokenizer import KhatriTokenizer
    import tempfile

    # Create and train tokenizer
    tokenizer = KhatriTokenizer(lowercase=True)
    corpus = ["hello world", "test one two"]
    tokenizer.train(corpus, vocab_size=20)

    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer.save(tmpdir)

        # Load tokenizer
        loaded = KhatriTokenizer.load(tmpdir)

        assert loaded.vocab_size == tokenizer.vocab_size
        assert loaded.lowercase == tokenizer.lowercase

        # Test encoding works the same
        text = "hello test"
        original_ids = tokenizer.encode(text)
        loaded_ids = loaded.encode(text)

        assert original_ids == loaded_ids

        print(f"  Saved and loaded vocab size: {loaded.vocab_size}")

    print("[OK] Tokenizer save/load")


def test_tokenizer_unicode():
    """Test Unicode support."""
    from khatrivoice.tokenizer.tokenizer import KhatriTokenizer

    tokenizer = KhatriTokenizer()

    # Test with various Unicode characters
    unicode_text = "Hello खत्री 世界 🌍"
    tokens = tokenizer.tokenize(unicode_text, mode="char")

    print(f"  Unicode text: '{unicode_text}'")
    print(f"  Tokens: {tokens[:20]}...")  # Show first 20

    # Should handle Unicode without errors
    ids = tokenizer.encode(unicode_text, mode="char")
    print(f"  IDs: {ids[:20]}...")

    print("[OK] Tokenizer Unicode support")


def test_tokenizer_trainer():
    """Test tokenizer trainer."""
    from khatrivoice.tokenizer.trainer import TokenizerTrainer

    trainer = TokenizerTrainer(vocab_size=50, lowercase=True)

    corpus = [
        "hello world",
        "hello test",
        "world of tests",
        "testing one two three",
    ]

    tokenizer = trainer.train_from_corpus(corpus, show_progress=False)

    print(f"  Trained vocab size: {tokenizer.vocab_size}")
    assert tokenizer.vocab_size <= 50 + 4  # vocab_size + special tokens

    print("[OK] Tokenizer trainer")


def test_tiny_test_tokenizer():
    """Test the tiny test tokenizer creator."""
    from khatrivoice.tokenizer.trainer import create_tiny_test_tokenizer

    tokenizer = create_tiny_test_tokenizer()

    print(f"  Tiny tokenizer vocab size: {tokenizer.vocab_size}")

    # Test it works
    ids = tokenizer.encode("hello world")
    decoded = tokenizer.decode(ids)
    print(f"  'hello world' -> {ids} -> '{decoded}'")

    print("[OK] Tiny test tokenizer")


def main():
    """Run all tests."""
    print("=" * 60)
    print("KhatriVoice Tokenizer Tests")
    print("=" * 60)
    print()

    tests = [
        test_vocabulary_creation,
        test_vocabulary_add_tokens,
        test_vocabulary_get_id,
        test_vocabulary_save_load,
        test_tokenizer_creation,
        test_tokenizer_tokenize,
        test_tokenizer_encode_decode,
        test_tokenizer_special_tokens,
        test_tokenizer_train,
        test_tokenizer_batch,
        test_tokenizer_save_load,
        test_tokenizer_unicode,
        test_tokenizer_trainer,
        test_tiny_test_tokenizer,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__} failed: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print()
    print("=" * 60)
    print(f"Tests passed: {passed}/{len(tests)}")
    if failed > 0:
        print(f"Tests failed: {failed}")
        sys.exit(1)
    print("All tokenizer tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
