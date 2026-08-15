"""
Logging utilities for KhatriVoice.

Provides structured logging with support for different output handlers
and formatting for both training and inference.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    name: str = "KhatriVoice",
    level: int = logging.INFO,
    log_file: Optional[str | Path] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging for KhatriVoice.

    Args:
        name: Logger name
        level: Logging level (default: INFO)
        log_file: Optional path to log file
        format_string: Optional custom format string

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Default format
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)s | %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "KhatriVoice") -> logging.Logger:
    """
    Get an existing logger or create a default one.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logging(name)
    return logger


class TrainingLogger:
    """
    Context manager and utility for training progress logging.

    Provides formatted output for training steps, epochs, and metrics.
    """

    def __init__(self, name: str = "KhatriVoice"):
        self.logger = get_logger(name)
        self.step = 0
        self.epoch = 0

    def log_startup(self, config: dict) -> None:
        """Log training startup information."""
        self.logger.info("=" * 60)
        self.logger.info("KhatriVoice v1 - Training")
        self.logger.info("=" * 60)
        self.logger.info("")
        self.logger.info("Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("")

    def log_device(self, device: str) -> None:
        """Log device information."""
        self.logger.info(f"Device: {device}")
        self.logger.info("")

    def log_parameters(self, total: int, trainable: int) -> None:
        """Log parameter counts."""
        self.logger.info(f"Total parameters: {total:,}")
        self.logger.info(f"Trainable parameters: {trainable:,}")
        self.logger.info("")

    def log_train_start(self) -> None:
        """Log training start."""
        self.logger.info("-" * 60)
        self.logger.info("Starting training...")
        self.logger.info("-" * 60)

    def log_step(
        self,
        step: int,
        loss: float,
        lr: float,
        grad_norm: Optional[float] = None,
    ) -> None:
        """Log training step information."""
        msg = f"Step {step:>6} | Loss: {loss:.4f} | LR: {lr:.2e}"
        if grad_norm is not None:
            msg += f" | Grad Norm: {grad_norm:.4f}"
        self.logger.info(msg)

    def log_eval(self, step: int, eval_loss: float, perplexity: float) -> None:
        """Log evaluation results."""
        self.logger.info(
            f"Eval at step {step:>6} | Loss: {eval_loss:.4f} | Perplexity: {perplexity:.2f}"
        )

    def log_save(self, path: str) -> None:
        """Log checkpoint save."""
        self.logger.info(f"Checkpoint saved: {path}")

    def log_generation(self, prompt: str, generated: str) -> None:
        """Log text generation."""
        self.logger.info(f"Prompt: {prompt}")
        self.logger.info(f"Generated: {generated}")

    def log_info(self, message: str) -> None:
        """Log general info message."""
        self.logger.info(message)

    def log_warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def log_error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)

    def log_train_end(self, total_steps: int, final_loss: float) -> None:
        """Log training completion."""
        self.logger.info("-" * 60)
        self.logger.info("Training complete!")
        self.logger.info(f"Total steps: {total_steps}")
        self.logger.info(f"Final loss: {final_loss:.4f}")
        self.logger.info("-" * 60)
