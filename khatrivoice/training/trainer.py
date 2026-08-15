"""
Training orchestration for KhatriVoice.

This module provides the main Trainer class that handles the complete
training loop including gradient accumulation, validation, and checkpointing.
"""

from typing import Optional, Dict, Any, List, Callable
import math
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from khatrivoice.config.model_config import KhatriVoiceConfig
from khatrivoice.model.khatrivoice import KhatriVoice
from khatrivoice.training.optimizer import create_optimizer, create_cosine_scheduler
from khatrivoice.training.checkpoint import CheckpointManager
from khatrivoice.training.metrics import MetricsTracker
from khatrivoice.utils.device import get_device
from khatrivoice.utils.seed import set_seed


class Trainer:
    """
    Main trainer for KhatriVoice language model.

    Handles:
    - Training loop with gradient accumulation
    - Validation evaluation
    - Checkpointing
    - Logging
    - Learning rate scheduling
    - Gradient clipping

    Attributes:
        model: KhatriVoice model
        config: Model configuration
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        device: Training device
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        checkpoint_manager: Checkpoint manager
        metrics_tracker: Metrics tracker
    """

    def __init__(
        self,
        model: KhatriVoice,
        config: KhatriVoiceConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        device: str = "auto",
        checkpoint_dir: str = "checkpoints",
        resume_from: Optional[str] = None,
    ) -> None:
        """
        Initialize the trainer.

        Args:
            model: KhatriVoice model to train
            config: Model configuration
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            device: Training device
            checkpoint_dir: Directory for checkpoints
            resume_from: Path to checkpoint to resume from
        """
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Setup device
        self.device = get_device(device)
        self.model = self.model.to(self.device)

        # Setup optimizer and scheduler
        self.optimizer = create_optimizer(
            self.model,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = create_cosine_scheduler(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.max_steps,
        )

        # Gradient scaling for mixed precision (device-agnostic API)
        self.scaler = GradScaler(self.device.type, enabled=(self.device.type == "cuda"))

        # Checkpointing
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=3,
            save_best=True,
        )

        # Metrics tracking
        self.metrics_tracker = MetricsTracker()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

        # Resume from checkpoint if specified
        if resume_from:
            self._resume_from_checkpoint(resume_from)

    def _resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Resume training from a checkpoint."""
        checkpoint = self.checkpoint_manager.load(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=str(self.device),
        )

        self.global_step = checkpoint.get("step", 0)
        self.epoch = checkpoint.get("epoch", 0)
        self.best_val_loss = checkpoint.get("loss", float("inf"))

        print(f"Resumed from step {self.global_step}, epoch {self.epoch}")

    def train(self) -> Dict[str, float]:
        """
        Run the training loop.

        Returns:
            Dictionary of final training metrics
        """
        print("=" * 60)
        print("KhatriVoice Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training steps: {self.config.max_steps}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Sequence length: {self.config.max_sequence_length}")
        print("=" * 60)
        print()

        # Set model to training mode
        self.model.train()

        # Training loop
        accumulated_loss = 0.0
        num_accumulated_steps = 0

        progress_bar = tqdm(
            total=self.config.max_steps,
            initial=self.global_step,
            desc="Training",
            unit="step",
        )

        while self.global_step < self.config.max_steps:
            for batch in self.train_dataloader:
                # Check if we've reached max steps
                if self.global_step >= self.config.max_steps:
                    break

                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                # Forward pass with device-agnostic autocast
                with autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                    outputs = self.model(
                        input_ids=input_ids,
                        labels=labels,
                        attention_mask=attention_mask,
                    )
                    loss = outputs["loss"]

                # Scale loss for gradient accumulation
                scaled_loss = loss / self.config.gradient_accumulation_steps

                # Backward pass
                self.scaler.scale(scaled_loss).backward()

                accumulated_loss += loss.item()
                num_accumulated_steps += 1

                # Step optimizer if accumulated enough
                if num_accumulated_steps >= self.config.gradient_accumulation_steps:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )

                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    # Update scheduler
                    self.scheduler.step()

                    # Update global step
                    self.global_step += 1
                    num_accumulated_steps = 0

                    # Get learning rate
                    current_lr = self.scheduler.get_last_lr()[0]

                    # Update metrics
                    self.metrics_tracker.update_train(
                        loss=accumulated_loss,
                        batch_size=self.config.batch_size,
                        seq_len=self.config.max_sequence_length,
                        learning_rate=current_lr,
                        grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    )

                    # Update progress bar
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        "loss": f"{accumulated_loss:.4f}",
                        "ppl": f"{self.metrics_tracker.get_train_perplexity():.2f}",
                        "lr": f"{current_lr:.2e}",
                    })

                    accumulated_loss = 0.0

                    # Log progress
                    if self.global_step % 10 == 0:
                        self._log_step()

                    # Validation
                    if self.config.eval_steps > 0 and self.global_step % self.config.eval_steps == 0:
                        val_loss = self.validate()
                        self.model.train()

                        # Save best checkpoint
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self._save_checkpoint(is_best=True)

                    # Save checkpoint
                    if self.config.save_steps > 0 and self.global_step % self.config.save_steps == 0:
                        self._save_checkpoint()

                # Accumulation point
                self.optimizer.zero_grad()

            # End of epoch
            self.epoch += 1

        progress_bar.close()

        # Final save
        self._save_checkpoint()

        return self.metrics_tracker.train_metrics.to_dict()

    def validate(self) -> float:
        """
        Run validation evaluation.

        Returns:
            Average validation loss
        """
        if self.val_dataloader is None:
            return float("inf")

        print("\nRunning validation...")
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                )

                total_loss += outputs["loss"].item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")

        # Update metrics
        self.metrics_tracker.update_val(
            loss=avg_loss,
            batch_size=self.config.batch_size,
            seq_len=self.config.max_sequence_length,
        )

        print(f"Validation Loss: {avg_loss:.4f}")
        print(f"Validation Perplexity: {self.metrics_tracker.get_val_perplexity():.2f}")

        return avg_loss

    def _log_step(self) -> None:
        """Log current training step."""
        metrics = self.metrics_tracker.log_step()
        train = metrics["train"]

        # Print detailed log every 100 steps
        if self.global_step % 100 == 0:
            print(f"\nStep {self.global_step}")
            print(f"  Loss: {train['avg_loss']:.4f}")
            print(f"  Perplexity: {train['perplexity']:.2f}")
            print(f"  Learning Rate: {train['learning_rate']:.2e}")
            print(f"  Gradient Norm: {train['grad_norm']:.4f}")

    def _save_checkpoint(self, is_best: bool = False) -> None:
        """Save a training checkpoint."""
        # Determine tokenizer path (saved in checkpoint directory)
        tokenizer_path = str(Path(self.checkpoint_manager.checkpoint_dir) / "tokenizer")

        self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.global_step,
            epoch=self.epoch,
            loss=self.metrics_tracker.train_metrics.loss,
            config=self.config,
            tokenizer_path=tokenizer_path,
        )

        if is_best:
            print(f"  Saved best checkpoint (loss: {self.best_val_loss:.4f})")


def create_trainer(
    model: KhatriVoice,
    config: KhatriVoiceConfig,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    device: str = "auto",
    checkpoint_dir: str = "checkpoints",
    resume_from: Optional[str] = None,
) -> Trainer:
    """
    Create a trainer instance.

    Args:
        model: KhatriVoice model
        config: Model configuration
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        device: Training device
        checkpoint_dir: Checkpoint directory
        resume_from: Checkpoint to resume from

    Returns:
        Trainer instance
    """
    return Trainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_from,
    )
