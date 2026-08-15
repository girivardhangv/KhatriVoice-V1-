"""
KhatriVoice - A transformer-based language model for household/kitchen domain.
"""

__version__ = "1.0.0"
__author__ = "KhatriVoice Team"

from khatrivoice.config.model_config import KhatriVoiceConfig
from khatrivoice.model.khatrivoice import KhatriVoice

__all__ = ["KhatriVoiceConfig", "KhatriVoice"]
