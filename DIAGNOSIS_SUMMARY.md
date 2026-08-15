# KhatriVoice Diagnosis and Fix Summary

## 1. Why the Old Model Produced Gibberish

**Root Cause: Extremely Repetitive Training Data**

The model produced output like `hi it,normally,,with units hosting remove in manager solutions one one message it remove` because:

1. **The dataset had only 108 unique user prompts and 54 unique AI responses** repeated 741-926x each
2. **Vocabulary diversity was extremely limited** - only 437 unique tokens in the entire corpus
3. **No label masking for user tokens** - the model wasted capacity learning to predict user inputs instead of focusing on assistant responses
4. **Missing conversational structure tokens** - no `<|user|>`, `<|assistant|>`, `<|end|>` markers to teach conversation boundaries
5. **No conversation-aware dataset** - lines were tokenized independently without understanding User/AI structure

The model essentially memorized a tiny set of patterns and generated random combinations of its limited vocabulary.

---

## 2. Why the Vocabulary Was Only 421

**Root Cause: Corpus Lacks Token Diversity**

The tokenizer worked correctly. The issue was the corpus:

- **Total lines in training data**: 100,000
- **Unique user prompts**: 108
- **Unique AI responses**: 54
- **Unique tokens found**: 437

The tokenizer's `train()` method correctly counts all unique tokens and limits to the available vocabulary. Since the corpus only contained 437 unique tokens (and 16 are reserved for special tokens), the result was 421 learned tokens.

**This is NOT a bug** - the tokenizer correctly adapted to the corpus. The problem was the synthetic data generation created highly repetitive content.

---

## 3. What Was Changed

### Files Created:
| File | Purpose |
|------|---------|
| `scripts/generate_diverse_data.py` | Generate varied training data with proper diversity |
| `scripts/preflight_check.py` | Comprehensive 10-test diagnostic before training |
| `khatrivoice/data/conversation_dataset.py` | Dataset with proper label masking for assistant responses |

### Files Modified:
| File | Changes |
|------|---------|
| `khatrivoice/tokenizer/vocabulary.py` | Added conversation tokens (`user_token`, `assistant_token`, `end_token`) |
| `khatrivoice/data/preprocessing.py` | Added conversation parsing (multiline format support) |
| `khatrivoice/data/dataset.py` | Added `ConversationDataset` with user token masking |
| `khatrivoice/training/trainer.py` | Fixed deprecated `torch.cuda.amp` → `torch.amp` API |
| `khatrivoice/inference/generator.py` | Added `repetition_penalty`, `stop_tokens` parameters |
| `khatrivoice/model/rope.py` | Fixed device consistency (`freqs_cis.to(q.device)`) |
| `scripts/generate.py` | Added chat mode, interactive mode, proper response extraction |
| `scripts/train.py` | Added `--conversation-mode` flag |
| `configs/small.yaml` | Configuration for ~8M parameter model |

### Key Fixes:
1. **Conversational Tokens**: Added `<|user|>`, `<|assistant|>`, `<|end|>` special tokens
2. **Label Masking**: User tokens masked with -100, loss only on assistant responses
3. **Device Handling**: All tensors created on correct device (CPU/CUDA)
4. **Multiline Parsing**: Support for your format (`User: ...\nAI: ...`)
5. **Modern AMP**: Updated to PyTorch 2.0+ `torch.amp` API

---

## 4. Checkpoint Compatibility

**Existing checkpoints are NOT compatible with the new vocabulary.**

The new vocabulary includes additional special tokens (`user_token`, `assistant_token`, `end_token`) that older checkpoints don't have. The model's embedding matrix dimensions will mismatch.

**Action Required**: You must retrain from scratch after generating diverse data.

---

## 5. Exact Command to Retrain

```bash
# Step 1: Generate diverse training data
python scripts/generate_diverse_data.py --output data/diverse_conversations.txt --count 50000

# Step 2: Run preflight diagnostic (DO NOT SKIP)
python scripts/preflight_check.py --config configs/small.yaml --data data/diverse_conversations.txt

# Step 3: Train with conversation mode
python scripts/train.py --config configs/small.yaml --conversation-mode
```

---

## 6. Exact Command to Test Generation

```bash
# Chat mode (formats prompt with special tokens)
python scripts/generate.py --checkpoint checkpoints/checkpoint_best.pt --chat --prompt "What is Python?"

# Interactive chat
python scripts/generate.py --checkpoint checkpoints/checkpoint_best.pt --interactive
```

---

## 7. Recommended Training Steps for 7.95M Model

### Configuration (configs/small.yaml):
```yaml
vocab_size: 8000          # Will adapt to actual corpus diversity
hidden_size: 256
num_layers: 6
num_attention_heads: 8
num_kv_heads: 4
intermediate_size: 1024
max_sequence_length: 512
dropout: 0.1

batch_size: 8
gradient_accumulation_steps: 2
learning_rate: 0.0003
warmup_steps: 100
max_steps: 5000
eval_steps: 100
save_steps: 500
```

### Training Process:

1. **Generate Quality Data** (min 10,000 unique conversations):
   ```bash
   python scripts/generate_diverse_data.py --count 50000
   ```

2. **Run Preflight Diagnostic**:
   ```bash
   python scripts/preflight_check.py --config configs/small.yaml --data data/diverse_conversations.txt
   ```
   Must pass all 10 tests before proceeding.

3. **Monitor Training**:
   ```bash
   python scripts/train.py --config configs/small.yaml --conversation-mode
   ```
   Watch for:
   - Training loss decreasing steadily
   - Validation loss not increasing (early stopping if needed)
   - No NaN/Inf values
   - Gradient norms < 10.0

4. **Test Generation**:
   ```bash
   python scripts/generate.py --checkpoint checkpoints/checkpoint_best.pt --chat --prompt "Hello"
   ```

### Expected Results After Proper Training:

| Prompt | Expected Output |
|--------|-----------------|
| "Hello" | "Hello! How can I help you today?" |
| "What is Python?" | "Python is a popular programming language known for its simple syntax..." |
| "What is AI?" | "AI stands for Artificial Intelligence, which enables computers to..." |

### Quality Checks:
- Response should directly address the prompt
- No excessive repetition
- Coherent grammar and structure
- Stops at `<|end|>` token
- No random token mixing

---

## Key Takeaway

**The model was not broken - the training data was.** With diverse data and proper conversation formatting, the 7.95M parameter architecture is sufficient for basic conversational responses about programming, AI, and general topics.

The preflight check script will catch these issues before you waste time training.
