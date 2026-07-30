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

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., min_length=1, max_length=4000)

class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., description="Full conversation history")
    system_prompt: str = Field(
        default="You are Bard, a wise and eloquent AI assistant deeply versed in the works of William Shakespeare. You speak with the grace and insight of a scholar of the Elizabethan era, yet you are warm, engaging, and helpful.",
        max_length=1000
    )
    max_tokens: int = Field(default=300, ge=1, le=1000)
    temperature: float = Field(default=0.75, ge=0.01, le=2.0)
    top_k: int = Field(default=40, ge=1, le=500)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.05, ge=1.0, le=3.0)

class ChatResponse(BaseModel):
    response: str
    tokens_generated: int
    time_seconds: float
    model_used: Optional[str]

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
            
            # SFT/DPO checkpoints may not contain the config to save space, 
            # so we fallback to best.pt's config
            model_cfg = ckpt.get("config")
            if model_cfg is None:
                base_ckpt_path = CHECKPOINT_DIR / "best.pt"
                if not base_ckpt_path.exists():
                    base_ckpt_path = CHECKPOINT_DIR / "last.pt"
                base_ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
                model_cfg = base_ckpt["config"]

            model = ShakespeareGPT(model_cfg)
            model.load_state_dict(ckpt["model_state_dict"])
            try:
                model.eval().to(device)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower():
                    logger.warning("CUDA Out of Memory! Falling back to CPU for inference.")
                    device = torch.device("cpu")
                    model.eval().to(device)
                else:
                    raise
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

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    """
    Multi-turn chat endpoint.

    Automatically selects the correct prompt format based on the loaded model:
    - SFT / fine-tuned model (sft_model.pt, dpo_model.pt): uses the structured
      chat template <|system|>...<|user|>...<|assistant|> that the model was
      trained on during fine-tuning.
    - Base / pre-trained model (best.pt, last.pt): uses a natural few-shot
      prompt format that mirrors text the model has seen during pre-training.
      Sampling is also tightened (lower temperature, lower top-p) to prevent
      the base model from drifting into hallucination.
    """
    client_ip = request.client.host
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    eos_id = tokenizer_instance.token_to_id("[EOS]") or 2

    # --- Determine whether the loaded model is fine-tuned or base ---
    is_sft_model = loaded_model_name is not None and any(
        tag in loaded_model_name.lower() for tag in ("sft", "dpo", "ft", "fine")
    )

    stop_seqs: list[str] = []

    if is_sft_model:
        # ==============================================================
        # SFT FORMAT
        # The model has been trained on natural dialogue markers.
        # ==============================================================
        parts = [f"System: {req.system_prompt}\n"]
        for msg in req.messages:
            if msg.role == "user":
                parts.append(f"\nUser: {msg.content}\n")
            elif msg.role == "assistant":
                parts.append(f"\nBard: {msg.content}\n")
        parts.append("\nBard: ")
        full_prompt = "".join(parts)
        stop_seqs   = ["\nUser:", "System:", "[EOS]"]

        # Use caller-supplied sampling params (SFT model is well-behaved)
        temperature       = req.temperature
        top_k             = req.top_k
        top_p             = req.top_p
        repetition_penalty = req.repetition_penalty

    else:
        # ==============================================================
        # BASE MODEL FORMAT (few-shot natural language)
        # The base model was never shown special tokens during
        # pre-training. Instead we use a Q&A style that looks like text
        # the model has already seen, and tighten sampling so it stays
        # on-topic rather than generating random historical narratives.
        # ==============================================================

        # Build a few-shot preamble from conversation history so the model
        # understands the dialogue pattern via in-context learning
        BARD_NAME = "Bard"
        USER_NAME = "Human"
        NEWLINE   = "\n"

        lines = [
            f"The following is a conversation between a human and {BARD_NAME}, "
            f"a knowledgeable and eloquent assistant specialising in the works "
            f"of William Shakespeare and Elizabethan literature.",
            f"",
            f"{BARD_NAME} always answers directly, concisely, and in an educated "
            f"Shakespearean style. {BARD_NAME} does not invent family histories, "
            f"dates, or people. If unsure, {BARD_NAME} says so gracefully.",
            f"",
        ]

        for msg in req.messages:
            if msg.role == "user":
                lines.append(f"{USER_NAME}: {msg.content}")
            elif msg.role == "assistant":
                lines.append(f"{BARD_NAME}: {msg.content}")

        # Prime generation with the assistant's name so the model
        # continues from that position
        lines.append(f"{BARD_NAME}:")
        full_prompt = NEWLINE.join(lines)

        # Stop when the model tries to generate the next human turn
        stop_seqs = [f"\n{USER_NAME}:", f"\n{BARD_NAME}:", "\n\n\n"]

        # Tight sampling for base model to prevent hallucination
        temperature        = min(req.temperature, 0.65)
        top_k              = min(req.top_k, 30)
        top_p              = min(req.top_p, 0.85)
        repetition_penalty = max(req.repetition_penalty, 1.1)

    # --- Tokenise and generate ---
    ids = tokenizer_instance.encode(full_prompt).ids
    max_prompt_len = BLOCK_SIZE - req.max_tokens
    if len(ids) > max_prompt_len:
        ids = ids[-max_prompt_len:]

    idx = torch.tensor([ids], dtype=torch.long, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    generated_text = ""
    token_count    = 0

    def detect_repetition_loop(text: str, ngram: int = 4, threshold: int = 3) -> bool:
        """
        Returns True if any n-gram phrase appears more than `threshold` times.
        This catches the degenerate loop the base model falls into when the
        repetition penalty is insufficient to escape a probability basin.
        """
        words  = text.lower().split()
        if len(words) < ngram * threshold:
            return False
        counts: dict = {}
        for i in range(len(words) - ngram + 1):
            key = " ".join(words[i:i + ngram])
            counts[key] = counts.get(key, 0) + 1
            if counts[key] >= threshold:
                return True
        return False

    def trim_repetition(text: str) -> str:
        """
        Remove the looping suffix from a degenerate output.
        Finds the first repeated 4-gram and truncates everything after
        its second appearance so the response ends cleanly.
        """
        words = text.split()
        seen: dict = {}
        for i in range(len(words) - 3):
            key = " ".join(words[i:i + 4]).lower()
            if key in seen:
                # Truncate at the second occurrence of the repeated phrase
                return " ".join(words[:seen[key] + 4]).strip()
            seen[key] = i
        return text

    for token in model.generate_stream(
        idx,
        max_new_tokens=req.max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=eos_id,
    ):
        token_str       = tokenizer_instance.decode([token])
        generated_text += token_str
        token_count    += 1

        # Stop on chat control tokens
        if any(stop in generated_text for stop in stop_seqs):
            for stop in stop_seqs:
                generated_text = generated_text.split(stop)[0]
            break

        # Stop on repetition collapse (base model safety net)
        if token_count > 40 and detect_repetition_loop(generated_text):
            generated_text = trim_repetition(generated_text)
            break

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    # If response is too short or empty (base model gave up), return a graceful fallback
    response_text = generated_text.strip()
    if not response_text or len(response_text.split()) < 3:
        if not is_sft_model:
            response_text = (
                "Forgive me, good traveller — my training is still in progress "
                "and I am not yet fully versed in the art of conversation. "
                "Run `python sft.py` to complete my instruction fine-tuning, "
                "then load `sft_model.pt` in the Model Selection to unlock my "
                "full conversational abilities."
            )

    return ChatResponse(
        response=response_text,
        tokens_generated=token_count,
        time_seconds=round(elapsed, 3),
        model_used=loaded_model_name
    )


@app.get("/sft/status")
async def get_sft_status():
    """Return SFT training status by reading the latest SFT log."""
    sft_log = Path("logs/sft_metrics.jsonl")
    sft_ckpt = CHECKPOINT_DIR / "sft_model.pt"
    sft_data = CHECKPOINT_DIR.parent / "data" / "sft_shakespeare.jsonl"

    status = {
        "sft_model_exists": sft_ckpt.exists(),
        "sft_data_exists": sft_data.exists(),
        "sft_data_path": str(sft_data),
        "sft_checkpoint_size_mb": round(sft_ckpt.stat().st_size / 1024 / 1024, 2) if sft_ckpt.exists() else 0,
        "latest_metrics": []
    }
    if sft_log.exists():
        with open(sft_log, "r") as f:
            for line in f.readlines()[-20:]:
                try:
                    status["latest_metrics"].append(json.loads(line))
                except:
                    pass
    return status


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
