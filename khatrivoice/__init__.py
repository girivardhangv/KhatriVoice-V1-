"""
KhatriVoice v1 - A decoder-only autoregressive Transformer language model.

This module implements KhatriVoice from scratch using PyTorch primitives,
designed for the SSK Khatri language.
"""

__version__ = "0.1.0"
__author__ = "KhatriVoice Team"

from khatrivoice.config.model_config import KhatriVoiceConfig
from khatrivoice.model.khatrivoice import KhatriVoice

__all__ = [
    "KhatriVoice",
    "KhatriVoiceConfig",
]
