"""
api_server.py - Production-Grade FastAPI Server for Project Bard
Features:
  - REST API for text generation
  - Server-Sent Events (SSE) streaming
  - Health check endpoint
  - Model info endpoint
  - Beautiful web UI served as static files
  - CORS support for cross-origin requests
  - Rate limiting via simple in-memory token bucket
  - Request/response validation via Pydantic
"""
import asyncio
import json
import time
import logging
from pathlib import Path
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import (
    DEVICE, CHECKPOINT_DIR, GEN_TEMPERATURE, GEN_TOP_K,
    GEN_TOP_P, GEN_REP_PENALTY, BLOCK_SIZE
)
from model import ShakespeareGPT, count_parameters
from tokenizer import load_tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000, description="Input text prompt")
    max_tokens: int = Field(default=200, ge=1, le=1000, description="Maximum tokens to generate")
    temperature: float = Field(default=GEN_TEMPERATURE, ge=0.01, le=2.0)
    top_k: int = Field(default=GEN_TOP_K, ge=1, le=500)
    top_p: float = Field(default=GEN_TOP_P, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=GEN_REP_PENALTY, ge=1.0, le=3.0)
    stream: bool = Field(default=False, description="Enable SSE streaming")
    stop_sequence: Optional[str] = Field(default=None, max_length=100)

class GenerateResponse(BaseModel):
    prompt: str
    generation: str
    tokens_generated: int
    time_seconds: float
    tokens_per_second: float

class ModelInfo(BaseModel):
    model_name: str
    parameters: int
    vocab_size: int
    context_window: int
    architecture: dict
    device: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float

class LoadModelRequest(BaseModel):
    model_name: str

class AvailableModelsResponse(BaseModel):
    current_model: Optional[str]
    available_models: list[str]

# --- Global State ---
model = None
tokenizer_instance = None
device = None
start_time = time.time()
model_cfg = None

loaded_model_name = None

def load_model_and_tokenizer(force_ckpt_name: str = None):
    global model, tokenizer_instance, device, model_cfg, loaded_model_name
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    
    # If a specific checkpoint is requested, only try that one
    checkpoints_to_try = [force_ckpt_name] if force_ckpt_name else ["best.pt", "last.pt", "sft_model.pt"]
    
    for ckpt_name in checkpoints_to_try:
        ckpt_path = CHECKPOINT_DIR / ckpt_name
        if ckpt_path.exists():
            logger.info(f"Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model_cfg = ckpt["config"]
            model = ShakespeareGPT(model_cfg)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval().to(device)
            loaded_model_name = ckpt_name
            break
    else:
        if force_ckpt_name:
            raise FileNotFoundError(f"Checkpoint {force_ckpt_name} not found.")
        logger.warning("No checkpoint found. Run train.py first.")
        # We don't raise error so server can start and serve UI
        return
    
    tokenizer_instance = load_tokenizer()
    logger.info(f"Model loaded on {device} with {count_parameters(model):,} parameters")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_and_tokenizer()
    yield
    logger.info("Shutting down...")

app = FastAPI(
    title="Project Bard API",
    description="Production-grade Shakespeare LLM API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple rate limiter
rate_limit_store = {}
RATE_LIMIT_REQUESTS = 30
RATE_LIMIT_WINDOW = 60  # seconds

def check_rate_limit(client_ip: str) -> bool:
    now = time.time()
    if client_ip not in rate_limit_store:
        rate_limit_store[client_ip] = []
    # Clean old entries
    rate_limit_store[client_ip] = [t for t in rate_limit_store[client_ip] if now - t < RATE_LIMIT_WINDOW]
    if len(rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        return False
    rate_limit_store[client_ip].append(now)
    return True

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        uptime_seconds=round(time.time() - start_time, 2)
    )

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfo(
        model_name="Project Bard - ShakespeareGPT",
        parameters=count_parameters(model),
        vocab_size=model_cfg.vocab_size,
        context_window=model_cfg.block_size,
        architecture={
            "type": "Decoder-only Transformer",
            "layers": model_cfg.n_layer,
            "heads": model_cfg.n_head,
            "embedding_dim": model_cfg.n_embd,
            "mlp_hidden": model_cfg.mlp_hidden,
            "features": ["RoPE", "SwiGLU", "RMSNorm", "KV Cache", "Flash Attention"]
        },
        device=str(device)
    )

@app.get("/models", response_model=AvailableModelsResponse)
async def list_models():
    models = []
    if CHECKPOINT_DIR.exists():
        for file in CHECKPOINT_DIR.glob("*.pt"):
            models.append(file.name)
    return AvailableModelsResponse(
        current_model=loaded_model_name,
        available_models=sorted(models)
    )

@app.post("/models/load")
async def load_specific_model(req: LoadModelRequest):
    try:
        load_model_and_tokenizer(force_ckpt_name=req.model_name)
        return {"status": "success", "message": f"Successfully loaded {req.model_name}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(req: GenerateRequest, request: Request):
    client_ip = request.client.host
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if req.stream:
        return StreamingResponse(
            stream_generate(req),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )
    
    # Non-streaming generation
    ids = tokenizer_instance.encode(req.prompt).ids
    max_prompt_len = BLOCK_SIZE - req.max_tokens
    if len(ids) > max_prompt_len:
        ids = ids[-max_prompt_len:]
    
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    eos_id = tokenizer_instance.token_to_id("[EOS]")
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    
    generated_text = ""
    token_count = 0
    for token in model.generate_stream(
        idx, max_new_tokens=req.max_tokens, temperature=req.temperature,
        top_k=req.top_k, top_p=req.top_p, repetition_penalty=req.repetition_penalty,
        eos_token_id=eos_id if eos_id is not None else 2,
    ):
        token_str = tokenizer_instance.decode([token])
        generated_text += token_str
        token_count += 1
        if req.stop_sequence and req.stop_sequence in generated_text:
            generated_text = generated_text.split(req.stop_sequence)[0]
            break
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    
    return GenerateResponse(
        prompt=req.prompt,
        generation=generated_text.strip(),
        tokens_generated=token_count,
        time_seconds=round(elapsed, 3),
        tokens_per_second=round(token_count / elapsed, 2) if elapsed > 0 else 0
    )


import random
@app.get("/dataset/sample")
async def get_dataset_sample(n: int = 5):
    """Return random paragraphs from the training dataset for the Dataset Explorer UI."""
    try:
        clean_path = Path("data/clean/shakespeare_clean.txt")
        if not clean_path.exists():
            return {"samples": ["[Dataset not found. Please run the data pipeline first.]"]}
        
        with open(clean_path, "r", encoding="utf-8") as f:
            content = f.read()
        paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 50]
        
        if not paragraphs:
            return {"samples": ["[Dataset is empty or too short.]"]}
            
        samples = random.sample(paragraphs, min(n, len(paragraphs)))
        return {"samples": samples}
    except Exception as e:
        return {"samples": [f"[Error loading dataset: {e}]"]}

@app.get("/pipeline/status")
async def get_pipeline_status():
    import json
    status = {
        "data": { "raw_size_mb": 0, "clean_size_mb": 0, "tokens_approx": 0 },
        "training": { "metrics": [], "checkpoints": [] }
    }
    raw_path = Path("data/raw/combined_raw.txt")
    clean_path = Path("data/clean/shakespeare_clean.txt")
    if raw_path.exists(): status["data"]["raw_size_mb"] = round(raw_path.stat().st_size / 1024 / 1024, 2)
    if clean_path.exists():
        sz = clean_path.stat().st_size
        status["data"]["clean_size_mb"] = round(sz / 1024 / 1024, 2)
        status["data"]["tokens_approx"] = sz // 4
        
    ckpt_dir = Path("checkpoints")
    if ckpt_dir.exists():
        for f in ckpt_dir.glob("*.pt"):
            status["training"]["checkpoints"].append({ "name": f.name, "size_mb": round(f.stat().st_size / 1024 / 1024, 2) })
            
    log_file = Path("logs/training_metrics.jsonl")
    if log_file.exists():
        with open(log_file, "r") as f:
            for line in f.readlines()[-100:]:
                try: status["training"]["metrics"].append(json.loads(line))
                except: pass
    return status

async def stream_generate(req: GenerateRequest) -> AsyncGenerator[str, None]:
    ids = tokenizer_instance.encode(req.prompt).ids
    max_prompt_len = BLOCK_SIZE - req.max_tokens
    if len(ids) > max_prompt_len:
        ids = ids[-max_prompt_len:]
    
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    eos_id = tokenizer_instance.token_to_id("[EOS]")
    
    generated_text = ""
    token_count = 0
    t0 = time.time()
    
    for token in model.generate_stream(
        idx, max_new_tokens=req.max_tokens, temperature=req.temperature,
        top_k=req.top_k, top_p=req.top_p, repetition_penalty=req.repetition_penalty,
        eos_token_id=eos_id if eos_id is not None else 2,
    ):
        token_str = tokenizer_instance.decode([token])
        generated_text += token_str
        token_count += 1
        
        data = json.dumps({"token": token_str, "token_count": token_count})
        yield f"data: {data}\n\n"
        await asyncio.sleep(0)  # Yield control to event loop
        
        if req.stop_sequence and req.stop_sequence in generated_text:
            break
    
    elapsed = time.time() - t0
    final = json.dumps({
        "done": True,
        "tokens_generated": token_count,
        "time_seconds": round(elapsed, 3),
        "tokens_per_second": round(token_count / elapsed, 2) if elapsed > 0 else 0
    })
    yield f"data: {final}\n\n"

# Serve the web UI
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    ui_path = Path(__file__).parent / "web" / "index.html"
    if ui_path.exists():
        return HTMLResponse(content=ui_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Project Bard API</h1><p>Web UI not found. Visit /docs for API documentation.</p>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
