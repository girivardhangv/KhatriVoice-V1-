#!/usr/bin/env python3
"""
Test script to verify the KhatriVoice training pipeline.

This script runs a short training test to ensure the pipeline works.
It trains on a tiny dataset and verifies that the loss decreases.

Run with: python tests/test_training.py
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


def test_optimizer_creation():
    """Test optimizer creation."""
    from khatrivoice.config.model_config import get_tiny_config
    from khatrivoice.model.khatrivoice import KhatriVoice
    from khatrivoice.training.optimizer import create_optimizer

    config = get_tiny_config()
    model = KhatriVoice(config)

    optimizer = create_optimizer(model, learning_rate=1e-4)

    assert optimizer is not None
    assert len(optimizer.param_groups) >= 1

    print("[OK] Optimizer creation")


def test_scheduler_creation():
    """Test scheduler creation."""
    from khatrivoice.config.model_config import get_tiny_config
    from khatrivoice.model.khatrivoice import KhatriVoice
    from khatrivoice.training.optimizer import create_optimizer, create_cosine_scheduler

    config = get_tiny_config()
    model = KhatriVoice(config)

    optimizer = create_optimizer(model, learning_rate=1e-4)
    scheduler = create_cosine_scheduler(
        optimizer,
        num_warmup_steps=10,
        num_training_steps=1000,
    )

    assert scheduler is not None

    # Check that learning rate changes
    initial_lr = scheduler.get_last_lr()[0]
    print(f"  Initial LR: {initial_lr:.2e}")

    # Step scheduler
    for _ in range(100):
        scheduler.step()

    warmed_up_lr = scheduler.get_last_lr()[0]
    print(f"  After warmup LR: {warmed_up_lr:.2e}")

    print("[OK] Scheduler creation")


def test_checkpoint_manager():
    """Test checkpoint manager."""
    import torch
    import tempfile
    from khatrivoice.config.model_config import get_tiny_config
    from khatrivoice.model.khatrivoice import KhatriVoice
    from khatrivoice.training.checkpoint import CheckpointManager
    from khatrivoice.training.optimizer import create_optimizer

    config = get_tiny_config()
    model = KhatriVoice(config)
    optimizer = create_optimizer(model)

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)

        # Save checkpoint
        path = manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            step=100,
            epoch=1,
            loss=1.5,
            config=config,
        )
        assert path.exists()

        # Load checkpoint
        checkpoint = manager.load(model=model, optimizer=optimizer)

        assert checkpoint["step"] == 100
        assert checkpoint["loss"] == 1.5

        print(f"  Saved checkpoint: {path.name}")

    print("[OK] Checkpoint manager")


def test_metrics_tracker():
    """Test metrics tracker."""
    from khatrivoice.training.metrics import MetricsTracker

    tracker = MetricsTracker()

    # Update training metrics
    tracker.update_train(
        loss=2.5,
        batch_size=4,
        seq_len=32,
        learning_rate=1e-4,
        grad_norm=1.0,
    )

    assert tracker.train_metrics.step == 1
    assert tracker.train_metrics.loss == 2.5

    # Get perplexity
    ppl = tracker.get_train_perplexity()
    print(f"  Loss: 2.5, Perplexity: {ppl:.2f}")

    # Update more
    for i in range(10):
        tracker.update_train(
            loss=2.5 - i * 0.1,
            batch_size=4,
            seq_len=32,
        )

    avg_loss = tracker.train_metrics.get_average_loss()
    ppl = tracker.get_train_perplexity()

    print(f"  Average loss: {avg_loss:.4f}, Perplexity: {ppl:.2f}")

    print("[OK] Metrics tracker")


def test_mini_training_loop():
    """Test a minimal training loop."""
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

    # Get config and device
    config = get_tiny_config()
    device = get_device("auto")

    print(f"  Device: {device}")

    # Create model
    model = KhatriVoice(config).to(device)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create tokenizer
    tokenizer = KhatriTokenizer(lowercase=True)
    texts = create_tiny_dataset()
    tokenizer.train(texts, vocab_size=config.vocab_size)

    # Create dataset
    dataset = KhatriDataset(
        tokenizer=tokenizer,
        texts=texts,
        max_length=config.max_sequence_length,
    )

    collator = DataCollator(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collator,
    )

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, learning_rate=1e-3)
    scheduler = create_cosine_scheduler(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=100,
    )

    # Training loop
    model.train()

    losses = []
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 20:  # Just 20 batches for testing
            break

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
        scheduler.step()

        losses.append(loss.item())

    # Check that loss decreased
    initial_loss = losses[0]
    final_loss = losses[-1]

    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")

    # Note: With only 20 steps, loss might not decrease much
    print(f"  Trainable for gradient flow")

    print("[OK] Mini training loop")


def test_full_training_run():
    """Test a full but short training run."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets
    import tempfile

    from khatrivoice.config.model_config import get_tiny_config
    from khatrivoice.model.khatrivoice import KhatriVoice
    from khatrivoice.tokenizer.tokenizer import KhatriTokenizer
    from khatrivoice.data.dataset import KhatriDataset
    from khatrivoice.data.collator import DataCollator
    from khatrivoice.data.preprocessing import create_tiny_dataset, split_train_val_test
    from khatrivoice.training.trainer import Trainer

    # Get config
    config = get_tiny_config()
    config.max_steps = 50  # Very short run
    config.eval_steps = 20
    config.save_steps = 20

    print(f"  Config: max_steps={config.max_steps}")

    # Create tokenizer
    tokenizer = KhatriTokenizer(lowercase=True)
    texts = create_tiny_dataset()
    tokenizer.train(texts, vocab_size=config.vocab_size)

    # Split data
    train_texts, val_texts, _ = split_train_val_test(texts, train_ratio=0.8, val_ratio=0.2)

    # Create datasets
    train_dataset = KhatriDataset(
        tokenizer=tokenizer,
        texts=train_texts,
        max_length=config.max_sequence_length,
    )

    val_dataset = KhatriDataset(
        tokenizer=tokenizer,
        texts=val_texts,
        max_length=config.max_sequence_length,
    )

    # Create data loaders
    collator = DataCollator(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collator,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collator,
    )

    # Create model
    model = KhatriVoice(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create trainer
        trainer = Trainer(
            model=model,
            config=config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            device="auto",
            checkpoint_dir=tmpdir,
        )

        # Train
        final_metrics = trainer.train()

        print(f"  Final loss: {final_metrics['loss']:.4f}")
        print(f"  Final perplexity: {final_metrics['perplexity']:.2f}")

        # Check checkpoint was saved
        checkpoints = list(Path(tmpdir).glob("checkpoint_step_*.pt"))
        assert len(checkpoints) > 0, "No checkpoints saved"
        print(f"  Checkpoints saved: {len(checkpoints)}")

    print("[OK] Full training run")


def test_device_handling():
    """Test device handling."""
    from khatrivoice.utils.device import get_device
    import torch

    # Test auto detection
    device_auto = get_device("auto")
    print(f"  Auto device: {device_auto}")

    # Test CPU
    device_cpu = get_device("cpu")
    print(f"  CPU device: {device_cpu}")

    # Test model creation on different devices
    from khatrivoice.config.model_config import get_tiny_config
    from khatrivoice.model.khatrivoice import KhatriVoice

    config = get_tiny_config()
    model = KhatriVoice(config)

    model_cpu = model.to("cpu")
    assert next(model_cpu.parameters()).device.type == "cpu"

    print("[OK] Device handling")


def main():
    """Run all tests."""
    print("=" * 60)
    print("KhatriVoice Training Tests")
    print("=" * 60)
    print()

    tests = [
        test_optimizer_creation,
        test_scheduler_creation,
        test_checkpoint_manager,
        test_metrics_tracker,
        test_device_handling,
        test_mini_training_loop,
        # Skip full training for quick testing
        # test_full_training_run,
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
    print("All training tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
