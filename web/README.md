# KhatriVoice Web Playground

A lightweight web interface for testing the KhatriVoice model locally.

## Architecture

```
Browser (HTML/CSS/JS)
       ↓
   FastAPI Server
       ↓
KhatriVoice Generator (reuse existing inference)
       ↓
   KhatriTokenizer
       ↓
   KhatriVoice Model
       ↓
   Generated Text
       ↓
   JSON Response
```

## Quick Start in GitHub Codespaces

### 1. Install Dependencies

```bash
pip install fastapi uvicorn jinja2 torch numpy pyyaml tqdm
```

Or use the project requirements:

```bash
pip install -r requirements.txt
pip install fastapi uvicorn jinja2
```

### 2. Ensure Model Checkpoint Exists

The web app looks for a checkpoint at `checkpoints/checkpoint_best.pt`.

If you don't have one, train a tiny model first:

```bash
python scripts/train.py --config configs/tiny.yaml
```

This will create checkpoints in the `checkpoints/` directory.

### 3. Start the Server

```bash
cd web
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Or run directly:

```bash
python web/app.py
```

### 4. Open in Browser

In GitHub Codespaces:
1. Go to the **Ports** tab
2. Find the forwarded port (8000)
3. Click the link to open in browser

Or manually forward:
1. Click **Ports** → **Forward a Port**
2. Enter `8000`
3. Click the URL that appears

### 5. Test the Model

1. Type a prompt in the text box
2. Adjust generation parameters if desired:
   - **Temperature**: Controls randomness (0 = deterministic, 1 = normal)
   - **Max Tokens**: Maximum length of generated text
   - **Top P**: Nucleus sampling threshold
   - **Top K**: Limit sampling to top K tokens
3. Click **Generate**
4. Wait for KhatriVoice's response

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8000 | Server port |
| `KHATRIVOICE_CHECKPOINT` | `checkpoints/checkpoint_best.pt` | Path to model checkpoint |

Example:

```bash
PORT=8080 python web/app.py
```

## API Endpoints

### GET /

Returns the web interface HTML.

### GET /api/status

Returns model loading status:

```json
{
    "model_loaded": true,
    "device": "cpu",
    "checkpoint_path": "checkpoints/checkpoint_best.pt",
    "error": null
}
```

### POST /api/generate

Generate text from a prompt:

**Request:**

```json
{
    "prompt": "Hello, world!",
    "max_new_tokens": 50,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 50
}
```

**Response:**

```json
{
    "response": "Hello, world! ...",
    "prompt": "Hello, world!",
    "max_new_tokens": 50,
    "temperature": 0.8,
    "top_p": 0.95,
    "model_loaded": true
}
```

### POST /api/reload

Reload the model from checkpoint.

### GET /api/health

Health check endpoint.

## Troubleshooting

### "Model Not Loaded" Error

1. Check that a checkpoint exists:
   ```bash
   ls checkpoints/
   ```

2. If missing, train a model first:
   ```bash
   python scripts/train.py --config configs/tiny.yaml
   ```

3. Restart the web server

### "Cannot Connect to Server" Error

1. Ensure the server is running
2. Check that you're using the correct port
3. In Codespaces, verify the port is forwarded

### Slow Generation

- The tiny model (~200K parameters) runs on CPU but is still slow
- Consider reducing `max_new_tokens`
- Set temperature to 0 for faster greedy decoding

### Out of Memory

- The tiny model should work on most machines
- If you get OOM, try restarting the server

## Stopping the Server

Press `Ctrl+C` in the terminal where the server is running.

## Files

```
web/
├── app.py          # FastAPI server
├── templates/
│   └── index.html  # Web interface
├── static/
│   ├── style.css   # Styles
│   └── app.js      # Frontend logic
└── README.md       # This file
```

## Notes

- The model loads once at startup for efficiency
- The web app reuses the existing KhatriVoice inference pipeline
- Works on CPU-only environments (GitHub Codespaces)
- No external API calls - model runs locally
