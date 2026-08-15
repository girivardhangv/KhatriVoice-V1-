# KhatriVoice Module Contract

This document defines the authoritative interface between **Module 1: KhatriVoice Brain Core** and **Module 2: Training & Runtime**.

**Version:** 1.0
**Last Updated:** 2024

---

## 1. Overview

### Module 1 (Brain Core) Provides:
- Model architecture (`KhatriVoice`)
- Configuration (`KhatriVoiceConfig`)
- Tokenizer (`KhatriTokenizer`)
- Transformer components (Attention, MLP, RoPE, etc.)

### Module 2 (Training & Runtime) Provides:
- Dataset orchestration
- Training loop
- Optimization
- Checkpointing
- Evaluation
- Generation/Inference

---

## 2. Configuration Contract

### Import Path
```python
from khatrivoice.config.model_config import KhatriVoiceConfig
```

### Required Fields
```python
@dataclass
class KhatriVoiceConfig:
    # Model Architecture (required)
    vocab_size: int           # Vocabulary size
    hidden_size: int          # Hidden dimension
    num_layers: int           # Number of transformer blocks
    num_attention_heads: int  # Number of attention heads
    num_kv_heads: int         # Number of KV heads (GQA)
    intermediate_size: int    # MLP intermediate dimension
    max_sequence_length: int  # Maximum context length
    rope_theta: float         # RoPE base frequency
    dropout: float            # Dropout probability

    # Training Parameters (Module 2 use)
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    max_grad_norm: float
    warmup_steps: int
    max_steps: int
    eval_steps: int
    save_steps: int
    log_steps: int

    # Paths
    checkpoint_dir: str
    resume_from: Optional[str]
    data_path: str
    tokenizer_path: str
    device: str
    seed: int
```

### Methods
```python
# Serialization
config.to_dict() -> dict
KhatriVoiceConfig.from_dict(d: dict) -> KhatriVoiceConfig

# File I/O
config.save(path: str) -> None
KhatriVoiceConfig.load(path: str) -> KhatriVoiceConfig

# Computed properties
config.head_dim -> int  # hidden_size // num_attention_heads
config.total_parameters -> int  # Estimated parameter count
```

### Factory Functions
```python
from khatrivoice.config.model_config import get_tiny_config, get_small_config, get_base_config

tiny_config = get_tiny_config()    # ~100K parameters, for CPU testing
small_config = get_small_config()  # ~10M parameters, for development
base_config = get_base_config()    # ~100M parameters, for production
```

---

## 3. Model Contract

### Import Path
```python
from khatrivoice.model.khatrivoice import KhatriVoice, create_model, load_model
```

### Model Constructor
```python
model = KhatriVoice(config: KhatriVoiceConfig)
```

### Forward Pass
```python
output = model(
    input_ids: Tensor,           # [batch_size, seq_len] - Token IDs
    attention_mask: Tensor,      # [batch_size, seq_len] - Optional, 1 for valid, 0 for padding
    labels: Tensor,              # [batch_size, seq_len] - Optional, -100 for ignored positions
    position_ids: Tensor,        # [batch_size, seq_len] - Optional, auto-generated if None
    past_key_values: List[Tuple], # Optional, KV cache for generation
    use_cache: bool,             # Whether to return KV cache
)
```

### Output Format
```python
{
    "logits": Tensor,            # [batch_size, seq_len, vocab_size]
    "loss": Tensor,              # Scalar, only if labels provided
    "past_key_values": List[Tuple],  # Only if use_cache=True
}
```

### Loss Calculation
The model handles loss computation internally:
- **Shift**: `logits[i]` predicts `labels[i+1]` (causal LM)
- **Padding**: `labels == -100` are ignored in loss
- **Loss type**: Cross-entropy with `ignore_index=-100`

### Model Methods
```python
# Parameter counting
model.count_parameters() -> Dict[str, int]
model.print_parameter_summary() -> None

# Embedding access
model.get_input_embeddings() -> nn.Embedding
model.set_input_embeddings(embedding: nn.Embedding) -> None
model.get_output_embeddings() -> nn.Linear

# Generation helper
model.prepare_inputs_for_generation(
    input_ids: Tensor,
    past_key_values: Optional[List[Tuple]],
    attention_mask: Optional[Tensor],
    **kwargs
) -> Dict[str, Any]
```

### State Dict
```python
# Save
state_dict = model.state_dict()

# Load
model.load_state_dict(state_dict)

# Checkpoint format (Module 2's responsibility)
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "step": int,
    "epoch": int,
    "loss": float,
    "config": dict,
}
```

---

## 4. Tokenizer Contract

### Import Path
```python
from khatrivoice.tokenizer.tokenizer import KhatriTokenizer
```

### Constructor
```python
tokenizer = KhatriTokenizer(
    vocab: Optional[Vocabulary] = None,
    lowercase: bool = False,
    max_token_length: int = 50,
    min_freq: int = 1,
)
```

### Properties
```python
tokenizer.vocab_size -> int
tokenizer.bos_id -> int
tokenizer.eos_id -> int
tokenizer.pad_id -> int
tokenizer.unk_id -> int
```

### Encoding
```python
# Encode text to token IDs
ids = tokenizer.encode(
    text: str,
    add_bos: bool = False,
    add_eos: bool = False,
    mode: str = "hybrid",  # "word", "char", or "hybrid"
) -> List[int]

# Batch encode
batch_ids = tokenizer.encode_batch(texts: List[str], ...) -> List[List[int]]
```

### Decoding
```python
# Decode token IDs to text
text = tokenizer.decode(
    token_ids: List[int],
    skip_special_tokens: bool = True,
    clean_up_tokenization: bool = True,
) -> str

# Batch decode
texts = tokenizer.decode_batch(batch_token_ids: List[List[int]], ...) -> List[str]
```

### Training
```python
tokenizer.train(
    corpus: List[str],
    mode: str = "word",
    vocab_size: Optional[int] = None,
) -> None
```

### Save/Load
```python
tokenizer.save(directory: str) -> None
tokenizer = KhatriTokenizer.load(directory: str) -> KhatriTokenizer
```

### Vocabulary Access
```python
tokenizer.vocab.get_id(token: str) -> int
tokenizer.vocab.get_token(id: int) -> str
```

---

## 5. Tensor Shape Specifications

### Input Tensors
```python
input_ids: [batch_size, seq_len], dtype=torch.long
attention_mask: [batch_size, seq_len], dtype=torch.long
  - 1 for valid tokens
  - 0 for padding tokens
labels: [batch_size, seq_len], dtype=torch.long
  - Token IDs for targets
  - -100 for ignored positions (padding)
position_ids: [batch_size, seq_len], dtype=torch.long
  - Position indices, 0-indexed
```

### Output Tensors
```python
logits: [batch_size, seq_len, vocab_size], dtype=torch.float32
loss: scalar, dtype=torch.float32 (if labels provided)
past_key_values: List of length num_layers
  Each element: (key: Tensor, value: Tensor)
  key/value shape: [batch_size, num_kv_heads, seq_len, head_dim]
```

### Gradient Accumulation
```python
# Module 2's responsibility
scaled_loss = loss / gradient_accumulation_steps
scaled_loss.backward()
# Accumulate gradients for gradient_accumulation_steps
# Then call optimizer.step()
```

---

## 6. Special Token IDs

| Token | ID | Description |
|-------|----|----|
| PAD | 0 | Padding token |
| UNK | 1 | Unknown token |
| BOS | 2 | Beginning of sequence |
| EOS | 3 | End of sequence |

---

## 7. Module Boundaries

### Module 1 Owns:
- All files under `khatrivoice/model/`
- All files under `khatrivoice/config/`
- All files under `khatrivoice/tokenizer/`
- `khatrivoice/__init__.py`

### Module 2 Owns:
- All files under `khatrivoice/data/`
- All files under `khatrivoice/training/`
- All files under `khatrivoice/inference/`
- All files under `khatrivoice/utils/`
- All files under `scripts/`
- All files under `tests/`
- Configuration files under `configs/`
- Checkpoints under `checkpoints/`

### Never Duplicate:
- Transformer blocks (`model/block.py`)
- Attention mechanism (`model/attention.py`)
- RoPE implementation (`model/rope.py`)
- MLP/SwiGLU (`model/mlp.py`)
- Normalization (`model/normalization.py`)
- Embeddings (`model/embeddings.py`)
- Model class (`model/khatrivoice.py`)
- Tokenizer (`tokenizer/tokenizer.py`)
- Vocabulary (`tokenizer/vocabulary.py`)

---

## 8. Training Pipeline Data Flow

```
Raw Text
    ↓
KhatriTokenizer (Module 1)
    ↓
List[int] token_ids
    ↓
KhatriDataset (Module 2)
    ↓
DataLoader (Module 2)
    ↓
batch = {input_ids, labels, attention_mask}
    ↓
KhatriVoice.forward() (Module 1)
    ↓
{logits, loss} or {logits}
    ↓
Trainer (Module 2)
    ↓
backward(), optimizer.step()
    ↓
CheckpointManager.save() (Module 2)
```

---

## 9. Generation Pipeline Data Flow

```
Prompt Text
    ↓
KhatriTokenizer.encode() (Module 1)
    ↓
Tensor input_ids [1, seq_len]
    ↓
KhatriVoiceGenerator (Module 2)
    ↓
Loop:
    KhatriVoice.forward(input_ids, use_cache=True) (Module 1)
    Sample next token
    Append to input_ids
    Until EOS or max_length
    ↓
Tensor generated_ids [1, total_len]
    ↓
KhatriTokenizer.decode() (Module 1)
    ↓
Generated Text
```

---

## 10. Checkpoint Format

Module 2 saves checkpoints in this format:

```python
checkpoint = {
    # Model state (required)
    "model_state_dict": OrderedDict,  # model.state_dict()

    # Training state (required)
    "optimizer_state_dict": dict,      # optimizer.state_dict()
    "scheduler_state_dict": dict,      # scheduler.state_dict()

    # Training progress (required)
    "step": int,                       # Global training step
    "epoch": int,                      # Current epoch
    "loss": float,                     # Current/last loss value

    # Configuration (required)
    "config": dict,                    # config.to_dict()

    # Optional
    "timestamp": str,                  # ISO timestamp
    "extra_state": dict,               # Additional state
}
```

---

## 11. Validation Tests

To verify Module 1/Module 2 integration:

1. **Model Creation**: `model = KhatriVoice(KhatriVoiceConfig())`
2. **Forward Pass**:
   ```python
   output = model(input_ids=torch.randint(0, 1000, (2, 10)))
   assert output["logits"].shape == (2, 10, vocab_size)
   ```
3. **Loss Computation**:
   ```python
   output = model(input_ids=torch.randint(0, 1000, (2, 10)),
                  labels=torch.randint(0, 1000, (2, 10)))
   assert output["loss"].requires_grad
   ```
4. **Tokenizer Roundtrip**:
   ```python
   text = "hello world"
   ids = tokenizer.encode(text)
   decoded = tokenizer.decode(ids)
   assert text in decoded  # May have minor differences
   ```
5. **Overfit Test**: Train on tiny dataset, loss should decrease below 0.5

---

## 12. API Stability Guarantees

### Stable (do not change without coordination):
- Model forward signature
- Tokenizer encode/decode signature
- Config field names
- Output dictionary keys

### Module 2 Can Modify:
- Training loop implementation
- Optimizer settings and defaults
- Scheduler parameters
- Checkpoint saving frequency
- Logging format

### Module 1 Can Modify:
- Internal implementation of components
- Variable names inside functions
- Private methods (prefixed with `_`)

---

## Contact for Interface Changes

If Module 2 requires functionality not in this contract:

1. Document the requirement
2. Do NOT create a duplicate implementation
3. Coordinate with Module 1 developer to extend the interface

---

*This contract is the source of truth. If code and contract disagree, the contract wins.*
