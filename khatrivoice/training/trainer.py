"""
Trainer for KhatriVoice language model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Any
from pathlib import Path

from khatrivoice.config.model_config import KhatriVoiceConfig
from khatrivoice.training.checkpoint import save_checkpoint, load_checkpoint
from khatrivoice.training.optimizer import create_optimizer, clip_gradients
from khatrivoice.training.scheduler import create_scheduler
from khatrivoice.training.metrics import MetricsTracker


class Trainer:
    """
    Trainer for KhatriVoice language model.

    Args:
        model: KhatriVoice model to train
        train_dataloader: DataLoader for training data
        val_dataloader: Optional DataLoader for validation data
        config: Training configuration
        tokenizer: Tokenizer instance
        device: Device to train on
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        config: KhatriVoiceConfig = None,
        tokenizer: Any = None,
        device: str = "auto",
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.tokenizer = tokenizer

        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Training on device: {self.device}")
        self.model = self.model.to(self.device)

        # Create optimizer and scheduler
        self.optimizer = create_optimizer(
            model=self.model,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            beta1=config.beta1,
            beta2=config.beta2,
        )

        self.scheduler = create_scheduler(
            optimizer=self.optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
        )

        # Mixed precision
        self.scaler = GradScaler(enabled=self.device.type == "cuda")

        # Metrics
        self.metrics = MetricsTracker()

        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")

    def train(self) -> None:
        """Run training loop."""
        print(f"\nStarting training for {self.config.max_steps} steps...")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Warmup steps: {self.config.warmup_steps}")
        print(f"  Save every: {self.config.save_steps} steps")
        print(f"  Eval every: {self.config.eval_steps} steps")

        self.model.train()
        epoch = 0

        while self.global_step < self.config.max_steps:
            epoch += 1
            print(f"\nEpoch {epoch}")

            for batch in self.train_dataloader:
                if self.global_step >= self.config.max_steps:
                    break

                # Training step
                loss = self._train_step(batch)

                # Log
                if self.global_step % self.config.log_steps == 0:
                    self._log_step(loss)

                # Evaluate
                if self.val_dataloader and self.global_step % self.config.eval_steps == 0:
                    val_loss = self.validate()
                    is_best = val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_loss

                    # Save checkpoint
                    self._save_checkpoint(is_best=is_best)

                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint(is_best=False)

                if self.global_step >= self.config.max_steps:
                    break

        # Save final checkpoint
        self._save_checkpoint(is_best=False)
        print("\nTraining complete!")

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute a single training step."""
        self.model.train()

        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # Forward pass
        with autocast(enabled=self.device.type == "cuda"):
            _, loss, _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        self.scaler.scale(loss).backward()

        # Gradient accumulation
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Clip gradients
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            # Scheduler step
            self.scheduler.step()

            # Track metrics
            self.metrics.add_loss(loss.item() * self.config.gradient_accumulation_steps)
            self.metrics.add_grad_norm(grad_norm.item())
            self.metrics.add_learning_rate(self.scheduler.get_last_lr()[0])

        self.global_step += 1

        return loss.item() * self.config.gradient_accumulation_steps

    def validate(self) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        print("\nRunning validation...")

        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                _, loss, _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"  Validation loss: {avg_loss:.4f}")

        self.model.train()
        return avg_loss

    def _log_step(self, loss: float) -> None:
        """Log training progress."""
        lr = self.scheduler.get_last_lr()[0]
        avg_loss = self.metrics.get_avg_loss()

        print(
            f"  Step {self.global_step}/{self.config.max_steps} | "
            f"Loss: {loss:.4f} (avg: {avg_loss:.4f}) | "
            f"LR: {lr:.6f}"
        )

    def _save_checkpoint(self, is_best: bool = False) -> None:
        """Save a checkpoint."""
        save_checkpoint(
            checkpoint_dir=self.config.checkpoint_dir,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.global_step,
            loss=self.metrics.get_avg_loss(),
            config=self.config,
            is_best=is_best,
        )

        # Save tokenizer
        if self.tokenizer:
            tokenizer_path = Path(self.config.checkpoint_dir) / "tokenizer"
            self.tokenizer.save(str(tokenizer_path))

    def resume(self, checkpoint_path: str) -> None:
        """Resume training from checkpoint."""
        checkpoint = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=str(self.device),
        )

        self.global_step = checkpoint["step"]
        print(f"Resumed from step {self.global_step}")
