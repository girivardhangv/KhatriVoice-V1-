#!/usr/bin/env python3
"""
Test script to verify the KhatriVoice model components.

Run with: python tests/test_model.py
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


def test_embedding_creation():
    """Test embedding layer creation."""
    from khatrivoice.model.embeddings import Embedding, TokenEmbedding

    # Token embedding
    token_emb = TokenEmbedding(num_embeddings=1000, embedding_dim=64)
    assert token_emb.num_embeddings == 1000
    assert token_emb.embedding_dim == 64

    print("[OK] TokenEmbedding creation")

    # Full embedding
    embedding = Embedding(
        vocab_size=1000,
        hidden_size=64,
        max_position_embeddings=128,
    )

    assert embedding.vocab_size == 1000

    print("[OK] Embedding creation")


def test_embedding_forward():
    """Test embedding forward pass."""
    import torch
    from khatrivoice.model.embeddings import Embedding

    embedding = Embedding(
        vocab_size=100,
        hidden_size=32,
        max_position_embeddings=16,
    )

    # Create input
    input_ids = torch.randint(0, 100, (2, 8))

    # Forward pass
    output = embedding(input_ids)

    assert output.shape == (2, 8, 32)

    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {output.shape}")
    print("[OK] Embedding forward")


def test_rope_creation():
    """Test RoPE creation."""
    from khatrivoice.model.rope import RotaryPositionEmbedding

    rope = RotaryPositionEmbedding(dim=16, max_seq_len=128)

    assert rope.dim == 16
    assert rope.max_seq_len == 128

    print("[OK] RotaryPositionEmbedding creation")


def test_rope_forward():
    """Test RoPE forward pass."""
    import torch
    from khatrivoice.model.rope import RotaryPositionEmbedding

    rope = RotaryPositionEmbedding(dim=16, max_seq_len=128)

    # Create Q and K tensors
    batch_size, seq_len, num_heads, head_dim = 2, 8, 4, 16
    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)

    # Apply RoPE
    q_rotated, k_rotated = rope(q, k)

    assert q_rotated.shape == q.shape
    assert k_rotated.shape == k.shape

    print(f"  Q shape: {q.shape}")
    print(f"  Q_rotated shape: {q_rotated.shape}")
    print("[OK] RoPE forward")


def test_rmsnorm():
    """Test RMSNorm."""
    import torch
    from khatrivoice.model.normalization import RMSNorm

    norm = RMSNorm(hidden_size=32)

    x = torch.randn(2, 8, 32)
    y = norm(x)

    assert y.shape == x.shape
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print("[OK] RMSNorm")


def test_swiglu():
    """Test SwiGLU MLP."""
    import torch
    from khatrivoice.model.mlp import SwiGLU

    mlp = SwiGLU(hidden_size=32, intermediate_size=64)

    x = torch.randn(2, 8, 32)
    y = mlp(x)

    assert y.shape == x.shape
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print("[OK] SwiGLU")


def test_attention_creation():
    """Test attention layer creation."""
    from khatrivoice.model.attention import CausalSelfAttention

    attn = CausalSelfAttention(
        hidden_size=64,
        num_heads=4,
        num_kv_heads=4,
        max_seq_len=128,
    )

    assert attn.hidden_size == 64
    assert attn.num_heads == 4

    print("[OK] CausalSelfAttention creation")


def test_attention_forward():
    """Test attention forward pass."""
    import torch
    from khatrivoice.model.attention import CausalSelfAttention

    attn = CausalSelfAttention(
        hidden_size=64,
        num_heads=4,
        num_kv_heads=4,
        max_seq_len=128,
    )

    x = torch.randn(2, 8, 64)
    output, cache = attn(x)

    assert output.shape == x.shape
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print("[OK] Attention forward")


def test_attention_gqa():
    """Test Grouped Query Attention."""
    import torch
    from khatrivoice.model.attention import CausalSelfAttention

    # GQA: 4 query heads, 2 KV heads
    attn = CausalSelfAttention(
        hidden_size=64,
        num_heads=4,
        num_kv_heads=2,  # GQA
        max_seq_len=128,
    )

    x = torch.randn(2, 8, 64)
    output, _ = attn(x)

    assert output.shape == x.shape
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print("[OK] GQA attention")


def test_attention_kv_cache():
    """Test KV cache for generation."""
    import torch
    from khatrivoice.model.attention import CausalSelfAttention

    attn = CausalSelfAttention(
        hidden_size=64,
        num_heads=4,
        num_kv_heads=4,
        max_seq_len=128,
    )

    # First forward pass
    x = torch.randn(1, 8, 64)
    output1, cache1 = attn(x, use_cache=True)

    assert cache1 is not None
    assert cache1[0].shape == (1, 8, 4, 16)  # K shape
    assert cache1[1].shape == (1, 8, 4, 16)  # V shape

    # Second forward pass with cache
    x2 = torch.randn(1, 1, 64)
    output2, cache2 = attn(x2, past_key_value=cache1, use_cache=True)

    assert cache2 is not None
    assert cache2[0].shape == (1, 9, 4, 16)  # K shape (8 + 1)

    print(f"  Cache K shape: {cache1[0].shape}")
    print(f"  After new token: {cache2[0].shape}")
    print("[OK] KV cache")


def test_transformer_block():
    """Test transformer block."""
    import torch
    from khatrivoice.model.block import TransformerBlock

    block = TransformerBlock(
        hidden_size=64,
        num_heads=4,
        num_kv_heads=4,
        intermediate_size=128,
        max_seq_len=128,
    )

    x = torch.randn(2, 8, 64)
    output, _ = block(x)

    assert output.shape == x.shape
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print("[OK] TransformerBlock")


def test_transformer():
    """Test full transformer."""
    import torch
    from khatrivoice.model.transformer import Transformer

    transformer = Transformer(
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        intermediate_size=128,
        max_seq_len=128,
    )

    x = torch.randn(2, 8, 64)
    output, _ = transformer(x)

    assert output.shape == x.shape
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print("[OK] Transformer")


def test_khatrivoice_creation():
    """Test KhatriVoice model creation."""
    from khatrivoice.config.model_config import get_tiny_config
    from khatrivoice.model.khatrivoice import KhatriVoice

    config = get_tiny_config()
    model = KhatriVoice(config)

    assert model.config.vocab_size == config.vocab_size

    # Print parameter summary
    model.print_parameter_summary()

    print("[OK] KhatriVoice creation")


def test_khatrivoice_forward():
    """Test KhatriVoice forward pass."""
    import torch
    from khatrivoice.config.model_config import get_tiny_config
    from khatrivoice.model.khatrivoice import KhatriVoice

    config = get_tiny_config()
    model = KhatriVoice(config)

    # Create input
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass
    output = model(input_ids)

    assert "logits" in output
    assert output["logits"].shape == (batch_size, seq_len, config.vocab_size)

    print(f"  Input shape: {input_ids.shape}")
    print(f"  Logits shape: {output['logits'].shape}")
    print("[OK] KhatriVoice forward")


def test_khatrivoice_with_labels():
    """Test KhatriVoice with labels (loss calculation)."""
    import torch
    from khatrivoice.config.model_config import get_tiny_config
    from khatrivoice.model.khatrivoice import KhatriVoice

    config = get_tiny_config()
    model = KhatriVoice(config)

    # Create input and labels
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    # Forward pass with labels
    output = model(input_ids, labels=labels)

    assert "loss" in output
    assert "logits" in output
    assert output["loss"].item() > 0

    print(f"  Loss: {output['loss'].item():.4f}")
    print("[OK] KhatriVoice with labels")


def test_khatrivoice_generation():
    """Test KhatriVoice generation (with KV cache)."""
    import torch
    from khatrivoice.config.model_config import get_tiny_config
    from khatrivoice.model.khatrivoice import KhatriVoice

    config = get_tiny_config()
    model = KhatriVoice(config)
    model.eval()

    # Create input
    batch_size, seq_len = 1, 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # First forward pass
    with torch.no_grad():
        output1 = model(input_ids, use_cache=True)

    assert "past_key_values" in output1
    past_key_values = output1["past_key_values"]

    # Second forward pass with cache
    next_token = torch.argmax(output1["logits"][:, -1, :], dim=-1, keepdim=True)

    with torch.no_grad():
        output2 = model(next_token, past_key_values=past_key_values, use_cache=True)

    print(f"  First output shape: {output1['logits'].shape}")
    print(f"  Second output shape: {output2['logits'].shape}")
    print("[OK] KhatriVoice generation")


def test_gradient_flow():
    """Test gradient flow through the model."""
    import torch
    from khatrivoice.config.model_config import get_tiny_config
    from khatrivoice.model.khatrivoice import KhatriVoice

    config = get_tiny_config()
    model = KhatriVoice(config)

    # Create input and labels
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    # Forward pass
    output = model(input_ids, labels=labels)

    # Backward pass
    output["loss"].backward()

    # Check gradients exist
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            # Check gradient is not all zeros
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    # Check a few specific gradients
    print("  Gradient norms:")
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"    {name[:40]:40s}: {grad_norm:.6f}")

    print("[OK] Gradient flow")


def test_parameter_count():
    """Test parameter count estimation."""
    from khatrivoice.config.model_config import get_tiny_config
    from khatrivoice.model.khatrivoice import KhatriVoice

    config = get_tiny_config()
    model = KhatriVoice(config)

    params = model.count_parameters()

    # Check consistency
    total = params["total"]
    trainable = params["trainable"]

    assert trainable == total, "All parameters should be trainable"

    # Verify counts
    manual_total = sum(p.numel() for p in model.parameters())
    assert total == manual_total, "Parameter count mismatch"

    print(f"  Total parameters: {total:,}")
    print(f"  Estimated: {config.total_parameters:,}")

    print("[OK] Parameter count")


def main():
    """Run all tests."""
    print("=" * 60)
    print("KhatriVoice Model Tests")
    print("=" * 60)
    print()

    tests = [
        test_embedding_creation,
        test_embedding_forward,
        test_rope_creation,
        test_rope_forward,
        test_rmsnorm,
        test_swiglu,
        test_attention_creation,
        test_attention_forward,
        test_attention_gqa,
        test_attention_kv_cache,
        test_transformer_block,
        test_transformer,
        test_khatrivoice_creation,
        test_khatrivoice_forward,
        test_khatrivoice_with_labels,
        test_khatrivoice_generation,
        test_gradient_flow,
        test_parameter_count,
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
    print("All model tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
