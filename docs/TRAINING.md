# KhatriVoice Training Guide

## Quick Reference

```bash
# 1. Prepare your data file (one sample per line)
# data/raw/khatri.txt

# 2. Train the model
python scripts/train_custom.py \
    --data data/raw/khatri.txt \
    --config configs/small.yaml \
    --vocab-size 5000 \
    --output-dir output/khatri_model

# 3. Generate text
python scripts/generate.py \
    --model output/khatri_model/model_final.pt \
    --prompt "your prompt here" \
    --max-new-tokens 100
```

---

## Step-by-Step Training

### Step 1: Prepare Your Data

Create a text file with your Khatri language samples. Each line is one sample:

```
# data/raw/khatri.txt
Your first Khatri text sample here.
Your second Khatri text sample here.
...
```

Or organize as multiple files:
```
data/raw/
├── khatri_001.txt
├── khatri_002.txt
└── ...
```

### Step 2: Choose Configuration

| Configuration | Use Case | Parameters | Training Time |
|---------------|----------|------------|---------------|
| `tiny.yaml`   | Debugging/testing | ~0.2M | Minutes |
| `small.yaml`  | Small datasets | ~8M | Hours |
| `base.yaml`   | Full training | ~100M | Days |

### Step 3: Train

```bash
# Basic training
python scripts/train_custom.py \
    --data data/raw/khatri.txt \
    --config configs/small.yaml \
    --vocab-size 5000

# Resume from checkpoint
python scripts/train_custom.py \
    --data data/raw/khatri.txt \
    --config configs/small.yaml \
    --resume output/checkpoints/checkpoint_latest.pt
```

### Step 4: Monitor Training

Training outputs:
- `output/checkpoints/` - Model checkpoints
- `output/tokenizer.json` - Trained tokenizer
- `output/model_final.pt` - Final model

### Step 5: Generate Text

```bash
python scripts/generate.py \
    --model output/model_final.pt \
    --prompt "Start text here" \
    --max-new-tokens 100 \
    --temperature 0.8
```

---

## Configuration Options

Edit YAML files in `configs/` to customize:

```yaml
# configs/custom.yaml

# Model architecture
vocab_size: 5000              # Vocabulary size
hidden_size: 256              # Hidden dimension
num_layers: 6                 # Number of transformer layers
num_attention_heads: 8        # Number of attention heads
num_kv_heads: 8               # KV heads (for GQA, use same as num_attention_heads for standard)
intermediate_size: 1024       # MLP intermediate size (usually 4x hidden_size)
max_sequence_length: 512      # Maximum sequence length
dropout: 0.1                  # Dropout rate

# Training
batch_size: 16                 # Batch size
learning_rate: 0.0003         # Learning rate
weight_decay: 0.01            # Weight decay
max_grad_norm: 1.0            # Gradient clipping
warmup_steps: 100             # Warmup steps
max_steps: 10000              # Total training steps

# Checkpointing
save_steps: 500               # Save every N steps
eval_steps: 200               # Evaluate every N steps
```

---

## Python API

```python
from khatrivoice import KhatriVoice, KhatriVoiceConfig
from khatrivoice.tokenizer import KhatriTokenizer
from khatrivoice.data import KhatriDataset, DataCollator
from khatrivoice.training import Trainer

# 1. Create configuration
config = KhatriVoiceConfig(
    vocab_size=5000,
    hidden_size=256,
    num_layers=6,
    num_attention_heads=8,
)

# 2. Train tokenizer
tokenizer = KhatriTokenizer(lowercase=True)
tokenizer.train(texts, vocab_size=config.vocab_size)

# 3. Create model
model = KhatriVoice(config)

# 4. Create datasets
train_dataset = KhatriDataset(tokenizer, train_texts, max_length=config.max_sequence_length)
val_dataset = KhatriDataset(tokenizer, val_texts, max_length=config.max_sequence_length)

# 5. Create data loaders
collator = DataCollator(tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=collator)
val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=collator)

# 6. Train
trainer = Trainer(model, config, train_loader, val_loader)
trainer.train()

# 7. Save
model.save("model.pt")
tokenizer.save("tokenizer.json")
```

---

## Generation

```python
import torch
from khatrivoice import KhatriVoice, KhatriVoiceConfig
from khatrivoice.tokenizer import KhatriTokenizer

# Load
config = KhatriVoiceConfig.load("configs/small.yaml")
model = KhatriVoice(config)
model.load_state_dict(torch.load("model.pt")["model_state_dict"])
tokenizer = KhatriTokenizer()
tokenizer.load("tokenizer.json")

# Generate
prompt = "Start here"
input_ids = torch.tensor([tokenizer.encode(prompt)])

with torch.no_grad():
    for _ in range(50):
        output = model(input_ids)
        next_token = output["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)

generated = tokenizer.decode(input_ids[0].tolist())
print(generated)
```

---

## Troubleshooting

### Loss doesn't decrease
- Check that tokenizer encodes your data properly
- Try lowering learning rate (e.g., 1e-4 → 3e-5)
- Verify data loading (print a batch to check)

### Out of memory
- Reduce batch size
- Reduce sequence length
- Use smaller model (tiny → small)

### Slow training
- Use GPU: `--device cuda`
- Increase batch size if memory allows
- Use `torch.compile(model)` for PyTorch 2.0+
