#!/usr/bin/env python3
"""
Test script to verify the KhatriVoice configuration system.

Run with: python tests/test_config.py
"""

import sys
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore
    except AttributeError:
        pass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_config_import():
    """Test that configuration can be imported."""
    from khatrivoice.config.model_config import (
        KhatriVoiceConfig,
        get_tiny_config,
        get_small_config,
        get_base_config,
    )
    print("[OK] Configuration imports successful")


def test_config_creation():
    """Test creating a configuration."""
    from khatrivoice.config.model_config import KhatriVoiceConfig

    config = KhatriVoiceConfig(
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=4,
        intermediate_size=256,
        max_sequence_length=128,
    )

    assert config.vocab_size == 1000
    assert config.hidden_size == 64
    assert config.num_layers == 2
    assert config.head_dim == 16  # 64 / 4
    print("[OK] Configuration creation successful")


def test_config_defaults():
    """Test default values in configuration."""
    from khatrivoice.config.model_config import KhatriVoiceConfig

    config = KhatriVoiceConfig()

    assert config.vocab_size > 0
    assert config.hidden_size > 0
    assert config.num_layers > 0
    assert config.dropout < 1.0
    print("[OK] Configuration defaults valid")


def test_config_validation():
    """Test configuration validation."""
    from khatrivoice.config.model_config import KhatriVoiceConfig

    # This should pass
    config = KhatriVoiceConfig(vocab_size=100, hidden_size=64, num_attention_heads=4, num_kv_heads=2)
    assert config.head_dim == 16

    # Check validation: hidden_size must be divisible by num_attention_heads
    try:
        bad_config = KhatriVoiceConfig(
            hidden_size=65,  # Not divisible by 4
            num_attention_heads=4,
        )
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "divisible" in str(e)

    print("[OK] Configuration validation working")


def test_config_parameter_count():
    """Test parameter estimation."""
    from khatrivoice.config.model_config import KhatriVoiceConfig

    # Tiny config ~100K parameters
    config = KhatriVoiceConfig(
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=4,
        intermediate_size=256,
    )

    params = config.total_parameters
    print(f"  Estimated parameters: {params:,}")
    assert params > 0, "Parameter count should be positive"
    print("[OK] Parameter estimation working")


def test_config_serialization():
    """Test YAML serialization."""
    from khatrivoice.config.model_config import KhatriVoiceConfig

    config = KhatriVoiceConfig(
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
    )

    # Test to_dict
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert config_dict["vocab_size"] == 1000

    # Test from_dict
    restored = KhatriVoiceConfig.from_dict(config_dict)
    assert restored.vocab_size == config.vocab_size
    assert restored.hidden_size == config.hidden_size

    print("[OK] Configuration serialization working")


def test_config_yaml_files():
    """Test loading configuration from YAML files."""
    from khatrivoice.config.model_config import KhatriVoiceConfig

    configs_dir = project_root / "configs"

    # Test tiny.yaml
    tiny_path = configs_dir / "tiny.yaml"
    assert tiny_path.exists(), f"Missing {tiny_path}"
    config = KhatriVoiceConfig.load(tiny_path)
    assert config.vocab_size == 1000
    assert config.num_layers == 2
    print(f"  tiny.yaml: {config.total_parameters:,} parameters")

    # Test small.yaml
    small_path = configs_dir / "small.yaml"
    assert small_path.exists(), f"Missing {small_path}"
    config = KhatriVoiceConfig.load(small_path)
    print(f"  small.yaml: {config.total_parameters:,} parameters")

    # Test base.yaml
    base_path = configs_dir / "base.yaml"
    assert base_path.exists(), f"Missing {base_path}"
    config = KhatriVoiceConfig.load(base_path)
    print(f"  base.yaml: {config.total_parameters:,} parameters")

    print("[OK] YAML configuration files loading correctly")


def test_preset_configs():
    """Test preset configuration functions."""
    from khatrivoice.config.model_config import (
        get_tiny_config,
        get_small_config,
        get_base_config,
    )

    tiny = get_tiny_config()
    assert tiny.vocab_size == 1000
    print(f"  tiny: {tiny.total_parameters:,} parameters")

    small = get_small_config()
    print(f"  small: {small.total_parameters:,} parameters")

    base = get_base_config()
    print(f"  base: {base.total_parameters:,} parameters")

    print("[OK] Preset configurations working")


def test_config_str():
    """Test string representation of config."""
    from khatrivoice.config.model_config import get_tiny_config

    config = get_tiny_config()
    config_str = str(config)

    assert "KhatriVoice Configuration" in config_str
    assert "vocab_size" in config_str
    print("\n" + str(config))
    print("\n[OK] Configuration string representation working")


def test_utils_import():
    """Test utility module imports."""
    from khatrivoice.utils.logging import setup_logging, get_logger
    from khatrivoice.utils.seed import set_seed
    from khatrivoice.utils.device import get_device

    # Test device detection
    device = get_device("auto")
    print(f"  Device: {device}")

    # Test seed setting
    set_seed(42)
    print("[OK] Utility modules imported and working")


def main():
    """Run all tests."""
    print("=" * 60)
    print("KhatriVoice Configuration Tests")
    print("=" * 60)
    print()

    tests = [
        test_config_import,
        test_config_creation,
        test_config_defaults,
        test_config_validation,
        test_config_parameter_count,
        test_config_serialization,
        test_config_yaml_files,
        test_preset_configs,
        test_config_str,
        test_utils_import,
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
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
