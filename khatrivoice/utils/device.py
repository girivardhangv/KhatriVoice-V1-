"""
Device utilities for KhatriVoice.

Provides automatic device detection and selection for CPU/CUDA training.
"""

from typing import Optional

import torch


def get_device(device: str = "auto") -> torch.device:
    """
    Get the appropriate PyTorch device.

    Args:
        device: One of 'auto', 'cpu', 'cuda', or a specific device like 'cuda:0'

    Returns:
        torch.device instance

    Raises:
        ValueError: If the device string is invalid or device is unavailable
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    # Handle specific device strings
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(
                f"CUDA requested but not available. "
                f"torch.cuda.is_available() returned False."
            )
        return torch.device(device)

    if device == "cpu":
        return torch.device(device)

    raise ValueError(
        f"Invalid device string: '{device}'. "
        f"Must be 'auto', 'cpu', 'cuda', or 'cuda:N'."
    )


def get_device_info(device: Optional[torch.device] = None) -> dict:
    """
    Get information about the available device(s).

    Args:
        device: Device to inspect (None for current device)

    Returns:
        Dictionary with device information
    """
    if device is None:
        device = get_device("auto")

    info = {
        "device": str(device),
        "device_type": device.type,
    }

    if device.type == "cuda":
        info.update({
            "device_name": torch.cuda.get_device_name(device),
            "device_count": torch.cuda.device_count(),
            "device_capability": torch.cuda.get_device_capability(device),
            "memory_allocated": torch.cuda.memory_allocated(device),
            "memory_reserved": torch.cuda.memory_reserved(device),
            "memory_total": torch.cuda.get_device_properties(device).total_memory,
        })
    elif device.type == "cpu":
        import multiprocessing
        info.update({
            "cpu_count": multiprocessing.cpu_count(),
        })

    return info


def to_device(
    obj: torch.Tensor | torch.nn.Module | dict | list,
    device: torch.device,
) -> torch.Tensor | torch.nn.Module | dict | list:
    """
    Recursively move tensors and modules to a device.

    Args:
        obj: Object to move (tensor, module, dict, or list)
        device: Target device

    Returns:
        Object on the target device
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, torch.nn.Module):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_device(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(to_device(item, device) for item in obj)
    return obj


def print_device_info(device: Optional[torch.device] = None) -> str:
    """
    Print formatted information about the device.

    Args:
        device: Device to inspect

    Returns:
        Formatted string with device information
    """
    info = get_device_info(device)
    lines = [
        "=" * 50,
        "Device Information",
        "=" * 50,
        f"Device: {info['device']}",
        f"Type: {info['device_type']}",
    ]

    if info["device_type"] == "cuda":
        lines.extend([
            f"Name: {info['device_name']}",
            f"Capability: {info['device_capability']}",
            f"Total Memory: {info['memory_total'] / 1024**3:.2f} GB",
        ])

    lines.append("=" * 50)
    return "\n".join(lines)


def clear_cuda_cache() -> None:
    """Clear the CUDA cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_memory_stats(device: torch.device) -> dict:
    """
    Get memory statistics for a device.

    Args:
        device: The device to query

    Returns:
        Dictionary with memory statistics in bytes
    """
    if device.type == "cuda":
        return {
            "allocated": torch.cuda.memory_allocated(device),
            "reserved": torch.cuda.memory_reserved(device),
            "peak_allocated": torch.cuda.max_memory_allocated(device),
            "peak_reserved": torch.cuda.max_memory_reserved(device),
        }
    return {
        "allocated": 0,
        "reserved": 0,
        "peak_allocated": 0,
        "peak_reserved": 0,
    }
