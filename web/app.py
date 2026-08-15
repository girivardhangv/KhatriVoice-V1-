"""
KhatriVoice Web Playground

A lightweight web interface for testing the KhatriVoice model.
Uses FastAPI and reuses the existing inference pipeline.
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import torch

# KhatriVoice imports
from khatrivoice.config.model_config import KhatriVoiceConfig
from khatrivoice.model.khatrivoice import KhatriVoice
from khatrivoice.tokenizer.tokenizer import KhatriTokenizer
from khatrivoice.utils.device import get_device

# Create FastAPI app
app = FastAPI(
    title="KhatriVoice Playground",
    description="Web interface for testing KhatriVoice model",
    version="1.0.0",
)

# Paths
WEB_DIR = Path(__file__).parent
PROJECT_ROOT = WEB_DIR.parent

# Templates and static files
templates = Jinja2Templates(directory=WEB_DIR / "templates")
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")

# Global model state (loaded once at startup)
class ModelState:
    """Holds the loaded model and generator."""
    model: Optional[KhatriVoice] = None
    tokenizer: Optional[KhatriTokenizer] = None
    device: Optional[torch.device] = None
    checkpoint_path: Optional[str] = None
    error: Optional[str] = None
    loaded: bool = False


model_state = ModelState()


# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50


class GenerateResponse(BaseModel):
    response: str
    prompt: str
    max_new_tokens: int
    temperature: float
    top_p: float
    model_loaded: bool


class StatusResponse(BaseModel):
    model_loaded: bool
    device: str
    checkpoint_path: Optional[str]
    error: Optional[str]


def load_model(checkpoint_path: Optional[str] = None) -> bool:
    """
    Load the KhatriVoice model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        True if successful, False otherwise
    """
    global model_state

    try:
        # Default checkpoint path
        if checkpoint_path is None:
            checkpoint_path = os.environ.get(
                "KHATRIVOICE_CHECKPOINT",
                str(PROJECT_ROOT / "checkpoints" / "checkpoint_best.pt")
            )

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            model_state.error = f"Checkpoint not found: {checkpoint_path}"
            return False

        # Get device
        model_state.device = get_device("auto")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Get config
        if "config" in checkpoint:
            config = KhatriVoiceConfig.from_dict(checkpoint["config"])
        else:
            model_state.error = "Checkpoint missing config"
            return False

        # Create model
        model_state.model = KhatriVoice(config)

        # Load state dict
        if "model_state_dict" in checkpoint:
            model_state.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model_state.error = "Checkpoint missing model_state_dict"
            return False

        # Move to device and set to eval mode
        model_state.model = model_state.model.to(model_state.device)
        model_state.model.eval()

        # Create tokenizer
        # Try to load from checkpoint directory or create a basic one
        tokenizer_path = checkpoint_path.parent / "tokenizer"
        if tokenizer_path.exists():
            model_state.tokenizer = KhatriTokenizer.load(tokenizer_path)
        else:
            # Create a basic tokenizer with common characters
            model_state.tokenizer = KhatriTokenizer(lowercase=True)
            # Add basic vocabulary
            basic_chars = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?;:'\"()-\n"
            model_state.tokenizer.vocab.add_tokens(list(basic_chars))

        model_state.checkpoint_path = str(checkpoint_path)
        model_state.loaded = True
        model_state.error = None

        return True

    except Exception as e:
        model_state.error = str(e)
        model_state.loaded = False
        return False


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main web interface."""
    html_path = WEB_DIR / "templates" / "index.html"
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get model loading status."""
    return StatusResponse(
        model_loaded=model_state.loaded,
        device=str(model_state.device) if model_state.device else "unknown",
        checkpoint_path=model_state.checkpoint_path,
        error=model_state.error,
    )


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate text using KhatriVoice model.

    Args:
        request: GenerateRequest with prompt and parameters

    Returns:
        GenerateResponse with generated text
    """
    # Check if model is loaded
    if not model_state.loaded or model_state.model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. {model_state.error or 'Unknown error'}"
        )

    # Validate prompt
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    # Validate parameters
    if request.max_new_tokens < 1 or request.max_new_tokens > 500:
        raise HTTPException(
            status_code=400,
            detail="max_new_tokens must be between 1 and 500"
        )

    if request.temperature < 0 or request.temperature > 2:
        raise HTTPException(
            status_code=400,
            detail="temperature must be between 0 and 2"
        )

    if request.top_p < 0 or request.top_p > 1:
        raise HTTPException(
            status_code=400,
            detail="top_p must be between 0 and 1"
        )

    try:
        # Use the generator from the inference module
        from khatrivoice.inference.generator import KhatriVoiceGenerator

        generator = KhatriVoiceGenerator(
            model=model_state.model,
            tokenizer=model_state.tokenizer,
            device=model_state.device,
        )

        # Generate text
        do_sample = request.temperature > 0

        results = generator.generate(
            prompt=request.prompt.strip(),
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature if do_sample else 1.0,
            top_k=request.top_k if request.top_k > 0 else None,
            top_p=request.top_p if request.top_p < 1.0 else None,
            do_sample=do_sample,
        )

        generated_text = results[0] if results else ""

        return GenerateResponse(
            response=generated_text,
            prompt=request.prompt.strip(),
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            model_loaded=True,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )


@app.post("/api/reload")
async def reload_model(checkpoint_path: Optional[str] = None):
    """Reload the model from checkpoint."""
    global model_state

    success = load_model(checkpoint_path)

    if success:
        return {"status": "success", "message": "Model reloaded successfully"}
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload model: {model_state.error}"
        )


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_state.loaded,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
