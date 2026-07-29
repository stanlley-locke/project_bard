"""
config.py - Production-Grade T4-Optimized Configuration
Centralized configuration for ~100M parameter model training and inference.

Supports environment variable overrides for production deployments:
  BARD_DEVICE, BARD_DTYPE, BARD_USE_WANDB, BARD_BATCH_SIZE,
  BARD_MAX_STEPS, BARD_LEARNING_RATE, BARD_CHECKPOINT_DIR
"""
import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


# -----------------------------
# Logging Configuration
# -----------------------------
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a configured logger for any module in the project."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
TOKENIZER_DIR = DATA_DIR / "tokenizer"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"

# Ensure all directories exist
for d in [RAW_DIR, CLEAN_DIR, TOKENIZER_DIR, CHECKPOINT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RAW_TEXT_PATH = RAW_DIR / "combined_raw.txt"
SHAKESPEARE_TEXT_PATH = RAW_DIR / "shakespeare.txt"
FINEWEB_TEXT_PATH = RAW_DIR / "fineweb.txt"
CLEAN_TEXT_PATH = CLEAN_DIR / "shakespeare_clean.txt"
TOKEN_IDS_PATH = CLEAN_DIR / "token_ids.bin"
SPLIT_DIR = DATA_DIR / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Data Sources (Expanded for larger token base)
# -----------------------------
DATA_SOURCES = [
    # Classic Literature (Project Gutenberg)
    "https://www.gutenberg.org/cache/epub/100/pg100.txt",   # The Complete Works of William Shakespeare
    "https://www.gutenberg.org/cache/epub/84/pg84.txt",      # Frankenstein by Mary Shelley
    "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",  # Pride and Prejudice by Jane Austen
    "https://www.gutenberg.org/cache/epub/11/pg11.txt",      # Alice's Adventures in Wonderland by Lewis Carroll
    "https://www.gutenberg.org/cache/epub/996/pg996.txt",    # Don Quixote by Miguel de Cervantes
    "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",  # The Adventures of Sherlock Holmes by Arthur Conan Doyle
    "https://www.gutenberg.org/cache/epub/1400/pg1400.txt",  # Great Expectations by Charles Dickens
    "https://www.gutenberg.org/cache/epub/1257/pg1257.txt",  # The Iliad by Homer
    "https://www.gutenberg.org/cache/epub/8800/pg8800.txt",  # The Divine Comedy by Dante Alighieri
    "https://www.gutenberg.org/cache/epub/76/pg76.txt",      # Adventures of Huckleberry Finn by Mark Twain
    "https://www.gutenberg.org/cache/epub/215/pg215.txt",    # The Picture of Dorian Gray by Oscar Wilde
    "https://www.gutenberg.org/cache/epub/98/pg98.txt",      # A Tale of Two Cities by Charles Dickens
    "https://www.gutenberg.org/cache/epub/160/pg160.txt",    # The Awakening by Kate Chopin
    "https://www.gutenberg.org/cache/epub/1260/pg1260.txt",  # Jane Eyre by Charlotte Bronte
    "https://www.gutenberg.org/cache/epub/1727/pg1727.txt",  # The Odyssey by Homer
]

# -----------------------------
# Tokenizer Configuration
# -----------------------------
VOCAB_SIZE = 32768         # Industry-standard size for rich subword patterns
SPECIAL_TOKENS: List[str] = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
PAD_TOKEN = "[PAD]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
UNK_TOKEN = "[UNK]"

# -----------------------------
# Data Splitting & Loading
# -----------------------------
BLOCK_SIZE = 2048          # Increased Context window size for deep reasoning
TRAIN_RATIO = 0.95
VAL_RATIO = 0.025
TEST_RATIO = 0.025
NUM_WORKERS = 2            # DataLoader workers (adjust based on CPU cores)

# -----------------------------
# Model Architecture (~350M parameters + MoE)
# -----------------------------
N_LAYER = 16               # Number of Transformer blocks
N_HEAD = 16                # Number of attention heads (Queries)
N_KV_HEAD = 4              # Number of KV heads (Grouped-Query Attention)
N_EMBD = 1024              # Embedding dimension
MLP_HIDDEN = 4096          # Hidden dimension for FFNs
DROPOUT = 0.1

# -----------------------------
# Feature Flags
# -----------------------------
USE_ROPE = True
USE_RMSNORM = True
USE_SWIGLU = True
USE_FLASH_ATTN = True      # Requires PyTorch 2.0+
USE_GRAD_CHECKPOINT = True # Trade compute for memory
USE_MOE = True             # Mixture of Experts
NUM_EXPERTS = 4            # Reduced from 8 to 4 to fit in 16GB VRAM
NUM_EXPERTS_PER_TOK = 2    # Experts routed to per token
ROPE_THETA = 10000.0       # Base frequency for RoPE

# -----------------------------
# Training Configuration (T4-optimized)
# -----------------------------
BATCH_SIZE = 2             # Reduced to fit the 1GB float32 logits gradient in backward pass
GRAD_ACCUM_STEPS = 32      # Adjusted to maintain effective batch size
BATCH_SIZE_EFFECTIVE = BATCH_SIZE * GRAD_ACCUM_STEPS  # Effective batch size = 64

NUM_EPOCHS = 3
LEARNING_RATE = 1.5e-4     # Conservative LR for ~100M params
WEIGHT_DECAY = 0.1
BETA1 = 0.9
BETA2 = 0.95
GRAD_CLIP = 1.0

WARMUP_STEPS = 200
MAX_STEPS = 2500
MIN_LR_RATIO = 0.1         # Final LR will be LEARNING_RATE * MIN_LR_RATIO

LOG_INTERVAL = 1           # Log every step (since 1 step takes ~48s)
EVAL_INTERVAL = 250
SAVE_INTERVAL = 500

SEED = 42
DTYPE = os.environ.get("BARD_DTYPE", "float16")       # T4 uses fp16 natively
DEVICE = os.environ.get("BARD_DEVICE", "cuda")

# -----------------------------
# Generation Configuration
# -----------------------------
GEN_TEMPERATURE = 0.8
GEN_TOP_K = 50
GEN_TOP_P = 0.95           # Nucleus sampling
GEN_REP_PENALTY = 1.1      # Repetition penalty
GEN_MAX_NEW_TOKENS = 500

# -----------------------------
# Supervised Fine-Tuning (SFT) & DPO
# -----------------------------
SFT_DATA_PATH = DATA_DIR / "sft_shakespeare.jsonl"
DPO_DATA_PATH = DATA_DIR / "dpo_shakespeare.jsonl"

SFT_EPOCHS = 2
SFT_LR = 1e-5
DPO_LR = 5e-6
DPO_BETA = 0.1

# -----------------------------
# Weights & Biases (WandB) Logging
# -----------------------------
USE_WANDB = os.environ.get("BARD_USE_WANDB", "true").lower() == "true"
WANDB_PROJECT = os.environ.get("BARD_WANDB_PROJECT", "project-bard")
WANDB_ENTITY = os.environ.get("BARD_WANDB_ENTITY", "stanlleylocke-ai")
os.environ["WANDB_MODE"] = "online"

# -----------------------------
# API Server Configuration
# -----------------------------
API_HOST = os.environ.get("BARD_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("BARD_API_PORT", "8000"))
API_RATE_LIMIT = int(os.environ.get("BARD_RATE_LIMIT", "30"))  # requests per minute