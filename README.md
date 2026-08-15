# KhatriVoice (V1)

A decoder-only autoregressive Transformer language model for the SSK Khatri language, built from scratch using PyTorch primitives.

## Overview

KhatriVoice is a GPT-style language model designed specifically for the SSK Khatri language. All parameters are initialized from random — no pretrained weights are used. The model architecture follows modern best practices for language modeling:

- **Decoder-only Transformer**: Standard autoregressive architecture
- **Rotary Position Embeddings (RoPE)**: Better length extrapolation than learned position embeddings
- **Grouped Query Attention (GQA)**: Efficient attention mechanism
- **SwiGLU Activation**: Modern activation function for feed-forward networks
- **RMSNorm**: Efficient normalization layer

## Project Structure

```
KhatriVoice(V1)/
├── khatrivoice/
│   ├── config/          # Configuration management
│   ├── tokenizer/       # BPE tokenizer implementation
│   ├── data/            # Data pipeline and preprocessing
│   ├── model/           # Neural network components
│   ├── training/        # Training infrastructure
│   └── utils/           # Utility functions
├── configs/             # YAML configuration files
│   ├── tiny.yaml        # Tiny model (~0.2M params)
│   ├── small.yaml       # Small model (~8M params)
│   └── base.yaml        # Base model (~100M params)
├── scripts/             # Training and inference scripts
├── tests/               # Test scripts
└── pyproject.toml       # Project configuration
```

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/KhatriVoice-V1.git
cd KhatriVoice-V1

# Create virtual environment
python -m venv venv

# Activate (Windows)
./venv/Scripts/activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

### Run Tests

```bash
# Test configuration system
python tests/test_config.py

# Test tokenizer
python tests/test_tokenizer.py

# Test data pipeline
python tests/test_dataset.py

# Test model components
python tests/test_model.py

# Test training pipeline
python tests/test_training.py

# Test overfit (verifies model can memorize tiny dataset)
python tests/test_overfit.py
```

### Train a Model

```bash
# Train with tiny configuration (fast, for testing)
python scripts/train.py --config configs/tiny.yaml

# Train with small configuration
python scripts/train.py --config configs/small.yaml

# Resume training from checkpoint
python scripts/train.py --config configs/tiny.yaml --resume checkpoints/checkpoint_latest.pt
```

## Configuration

Models are configured via YAML files. Three preset configurations are provided:

| Config | Hidden Size | Layers | Heads | Parameters |
|--------|-------------|--------|-------|------------|
| tiny   | 64          | 2      | 4     | ~0.2M      |
| small  | 128         | 4      | 8     | ~8M        |
| base   | 512         | 12     | 16    | ~100M      |

### Configuration Example

```yaml
# configs/custom.yaml
vocab_size: 1000
hidden_size: 128
num_layers: 4
num_attention_heads: 8
num_kv_heads: 8
intermediate_size: 512
max_sequence_length: 256

# Training
batch_size: 8
learning_rate: 0.0003
max_steps: 10000
warmup_steps: 100
```

## Model Architecture

### Core Components

1. **Embedding Layer**
   - Token embeddings with weight tying to output head
   - No learned position embeddings (uses RoPE)

2. **Transformer Blocks**
   - Pre-normalization with RMSNorm
   - Causal self-attention with RoPE
   - SwiGLU feed-forward network
   - Residual connections

3. **Rotary Position Embeddings (RoPE)**
   - Encodes position through rotation of query/key vectors
   - Better length extrapolation than learned embeddings

4. **Grouped Query Attention (GQA)**
   - Multiple query heads share fewer key/value heads
   - Reduces KV cache size for inference

### Parameter Count Estimation

```python
params ≈ vocab_size * hidden_size  # Embeddings
      + 2 * hidden_size^2 * num_layers  # Attention Q,K,V,O
      + hidden_size * intermediate_size * 3 * num_layers  # MLP
      + hidden_size * 4 * num_layers  # LayerNorm
```

## Data Pipeline

### Tokenizer

KhatriVoice uses a custom BPE tokenizer:

```python
from khatrivoice.tokenizer import KhatriTokenizer

# Create and train tokenizer
tokenizer = KhatriTokenizer(lowercase=True)
tokenizer.train(texts, vocab_size=1000)

# Encode
ids = tokenizer.encode("hello world")

# Decode
text = tokenizer.decode(ids)

# Save/load
tokenizer.save("tokenizer.json")
tokenizer.load("tokenizer.json")
```

### Dataset

```python
from khatrivoice.data import KhatriDataset
from khatrivoice.data import DataCollator

# Create dataset
dataset = KhatriDataset(
    tokenizer=tokenizer,
    texts=texts,
    max_length=128,
)

# Create collator
collator = DataCollator(tokenizer=tokenizer)

# Create dataloader
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=4,
    collate_fn=collator,
)
```

## Training

### Training Loop

The training loop includes:
- Gradient accumulation
- Mixed precision training (on CUDA)
- Gradient clipping
- Learning rate scheduling (cosine with warmup)
- Checkpointing
- Validation evaluation

```python
from khatrivoice.training import Trainer

trainer = Trainer(
    model=model,
    config=config,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    device="auto",
)

trainer.train()
```

### Checkpointing

```python
from khatrivoice.training import CheckpointManager

manager = CheckpointManager("checkpoints/")

# Save
manager.save(model, optimizer, scheduler, step, epoch, loss, config)

# Load
checkpoint = manager.load(model=model, optimizer=optimizer)

# Get latest/best checkpoint
latest = manager.get_latest_checkpoint()
best = manager.get_best_checkpoint()
```

## Testing Philosophy

This project follows the principle: **"If you can't overfit a tiny dataset, there's a bug."**

The `test_overfit.py` script trains on 5 repeated sentences and verifies:
- Loss decreases from ~7.0 to <0.01
- Perplexity reaches ~1.0
- Model generates expected patterns

## Development

### Running All Tests

```bash
# Run individual tests
python tests/test_config.py
python tests/test_tokenizer.py
python tests/test_dataset.py
python tests/test_model.py
python tests/test_training.py
python tests/test_overfit.py
```

### Code Style

- Use type hints
- Write docstrings for all public APIs
- Follow PEP 8
- Keep functions focused and testable

## License

[Your License Here]

## Acknowledgments

- Transformer architecture based on "Attention Is All You Need"
- RoPE from "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- GQA from "GQA: Training Generalized Multi-Query Transformer Models"
- SwiGLU from "GLU Variants Improve Transformer"
