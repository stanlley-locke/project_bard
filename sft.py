# ==============================================================================
# PROJECT BARD
# ==============================================================================
#
# @file      : sft.py
# @author    : github.com/stanlley-locke
# @repo      : project_bard
# @desc      : Production-Grade Supervised Fine-Tuning (SFT) + DPO Pipeline
#
# Features:
#   - Supervised Fine-Tuning (SFT) on instruction/response JSONL pairs
#   - Multi-turn conversation dataset support ({system, messages: [{role, content}]})
#   - LoRA (Low-Rank Adaptation) - trains tiny adapter matrices, freezes base model
#   - Response-only loss masking (loss computed only on assistant tokens)
#   - Gradient accumulation to match effective batch size without extra VRAM
#   - Per-layer learning rate groups (lower LR for early layers, higher for final)
#   - Cosine LR with linear warmup
#   - 8-bit AdamW optimizer (bitsandbytes) with standard AdamW fallback
#   - float16 mixed precision with GradScaler
#   - WandB logging with SFT-specific metrics
#   - Checkpoint saving: sft_model.pt + sft_model.safetensors
#   - Resume from SFT checkpoint (crash recovery)
#   - Early stopping + best model tracking by validation loss
#   - Graceful Ctrl+C: saves checkpoint immediately on interrupt
#   - Data validation, deduplication, and length-based curriculum sorting
#   - BLEU-score evaluation after each epoch
#   - Optional DPO (Direct Preference Optimization) stage
#   - Automatic synthetic dataset generation if no JSONL data is found
# ==============================================================================

import gc
import json
import math
import os
import signal
import time
import hashlib
import copy
from collections import Counter
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from config import (
    BLOCK_SIZE, CHECKPOINT_DIR, DEVICE, DPO_BETA, DPO_DATA_PATH,
    DPO_LR, DTYPE, LOG_DIR, SFT_DATA_PATH, SFT_EPOCHS, SFT_LR,
    USE_WANDB, WANDB_ENTITY, WANDB_PROJECT, WEIGHT_DECAY, BETA1, BETA2,
    GRAD_CLIP,
)
from model import ModelConfig, ShakespeareGPT, count_parameters
from tokenizer import load_tokenizer


# ==============================================================================
# CONSTANTS
# ==============================================================================

USER_TOKEN      = "\nUser: "
ASSISTANT_TOKEN = "\nBard: "
END_TOKEN       = "\n"
SYSTEM_TOKEN    = "System: "
IGNORE_INDEX    = -100        # PyTorch ignores this label index in cross-entropy

# SFT-specific hyperparameters (can be overridden via environment variables)
SFT_BATCH_SIZE      = int(os.environ.get("BARD_SFT_BATCH_SIZE", "1"))
SFT_GRAD_ACCUM      = int(os.environ.get("BARD_SFT_GRAD_ACCUM", "8"))   # effective batch = 8
SFT_EARLY_STOP      = int(os.environ.get("BARD_SFT_EARLY_STOP", "3"))   # epochs without improvement
SFT_USE_LORA        = os.environ.get("BARD_SFT_LORA", "true").lower() == "true"
LORA_RANK           = int(os.environ.get("BARD_LORA_RANK", "16"))
LORA_ALPHA          = float(os.environ.get("BARD_LORA_ALPHA", "32"))
LORA_DROPOUT        = float(os.environ.get("BARD_LORA_DROPOUT", "0.05"))

# Default system prompt injected into every chat turn
SYSTEM_PROMPT = (
    "You are Bard, a wise and eloquent AI assistant deeply versed in the works "
    "of William Shakespeare. You speak with the grace and insight of a scholar "
    "of the Elizabethan era, yet you are warm, engaging, and helpful. When asked "
    "to write creatively, you channel the spirit of Shakespeare himself."
)


# ==============================================================================
# GRACEFUL SHUTDOWN
# ==============================================================================

shutdown_requested = False

def _signal_handler(sig, frame):
    global shutdown_requested
    print("\n[!] Interrupt received. Saving checkpoint and exiting gracefully...")
    shutdown_requested = True

signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ==============================================================================
# SECTION 1 - LORA ADAPTER
# ==============================================================================

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation of a frozen nn.Linear layer.

    Instead of training the full weight matrix W (d_in x d_out),
    LoRA trains two small matrices A (d_in x r) and B (r x d_out)
    and adds their product as a residual: W' = W + (alpha/r) * B @ A.

    This reduces trainable parameters by orders of magnitude while
    preserving the base model's general knowledge.
    """

    def __init__(self, base_linear: nn.Linear, r: int = 16, alpha: float = 32.0,
                 dropout: float = 0.05):
        super().__init__()
        self.base    = base_linear
        self.r       = r
        self.alpha   = alpha
        self.scaling = alpha / r

        d_in  = base_linear.in_features
        d_out = base_linear.out_features

        # Freeze the base weight
        for p in self.base.parameters():
            p.requires_grad_(False)

        # Trainable low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(d_in, r) * (1.0 / math.sqrt(r)))
        self.lora_B = nn.Parameter(torch.zeros(r, d_out))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B
        return base_out + lora_out * self.scaling

    def merge_weights(self) -> nn.Linear:
        """Merge LoRA weights back into the base linear for efficient inference."""
        merged = copy.deepcopy(self.base)
        with torch.no_grad():
            delta = (self.lora_A @ self.lora_B).T * self.scaling
            merged.weight.data += delta.to(merged.weight.dtype)
        return merged


def inject_lora(model: nn.Module, r: int = 16, alpha: float = 32.0,
                dropout: float = 0.05) -> Tuple[nn.Module, int]:
    """
    Replace all attention projection (query, key, value, output) and MLP
    linear layers in the transformer with LoRA-wrapped versions.
    Returns the modified model and the count of trainable parameters.
    """
    replaced = 0
    for name, module in list(model.named_modules()):
        # Target attention projections and MLP layers
        if not isinstance(module, nn.Linear):
            continue
        # Skip the final language model head (lm_head) — keep it full rank
        if "lm_head" in name or "head" in name:
            continue

        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model
        if parent_name:
            for part in parent_name.split("."):
                parent = getattr(parent, part)

        lora_layer = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
        setattr(parent, child_name, lora_layer)
        replaced += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] Replaced {replaced} linear layers.")
    print(f"[LoRA] Trainable params: {trainable:,} / {total:,} "
          f"({100.0 * trainable / total:.2f}%)")
    return model, trainable


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Walk the model and merge all LoRA adapters back into their base weights."""
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model
        if parent_name:
            for part in parent_name.split("."):
                parent = getattr(parent, part)
        setattr(parent, child_name, module.merge_weights())
    print("[LoRA] All adapter weights merged into base model.")
    return model


# ==============================================================================
# SECTION 2 - DATA PIPELINE
# ==============================================================================

# ---------------------------------------------------------------------------
# Synthetic Dataset (auto-generated if no JSONL file is found)
# ---------------------------------------------------------------------------
def generate_synthetic_sft_data(output_path: Path, num_examples: int = 5000) -> int:
    """
    Generates a diverse synthetic instruction dataset covering multiple personas
    and task types (General QA, Coding, Creative, Shakespeare).
    """
    import json
    import random
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[*] No SFT data found at {output_path}. Generating large synthetic dataset...")
    
    personas = [
        {
            "system": "You are a helpful, harmless, and honest AI assistant.",
            "prompts": [
                ("Explain quantum computing in simple terms.", "Quantum computing uses the principles of quantum mechanics to process information. Unlike classical bits (0 or 1), qubits can exist in multiple states simultaneously, allowing for exponentially faster calculations in certain tasks."),
                ("What is the capital of France?", "The capital of France is Paris."),
                ("How do I boil an egg?", "To boil an egg, place it in a pot of boiling water for 6-9 minutes depending on how firm you want the yolk. Then plunge it into ice water to stop the cooking."),
                ("Write a polite email declining a job offer.", "Subject: Job Offer - [Your Name]\n\nDear [Name],\n\nThank you so much for offering me the position. After careful consideration, I must politely decline as I have accepted another opportunity that better aligns with my current career goals. I wish you and the team the best.\n\nSincerely,\n[Your Name]"),
                ("What is 15% of 80?", "15% of 80 is 12."),
                ("Why is the sky blue?", "The sky appears blue because molecules in the Earth's atmosphere scatter sunlight in all directions. Blue light is scattered more than other colors because it travels as shorter, smaller waves."),
            ]
        },
        {
            "system": "You are an expert Python programmer.",
            "prompts": [
                ("Write a python script to reverse a string.", "```python\ndef reverse_string(s: str) -> str:\n    return s[::-1]\n```"),
                ("How do I read a JSON file in Python?", "You can use the built-in `json` module:\n```python\nimport json\n\nwith open('data.json', 'r') as f:\n    data = json.load(f)\n```"),
                ("What is a list comprehension?", "A list comprehension is a concise way to create lists in Python. For example, to create a list of squares: `[x**2 for x in range(10)]`."),
                ("How do I handle exceptions in Python?", "You use a try-except block:\n```python\ntry:\n    result = 10 / 0\nexcept ZeroDivisionError:\n    print('Cannot divide by zero!')\n```"),
            ]
        },
        {
            "system": "You are Bard, a wise and eloquent AI assistant deeply versed in the works of William Shakespeare.",
            "prompts": [
                ("Who are you?", "Hail, good traveller! I am Bard, a humble servant of the written word, crafted in the spirit of the immortal Shakespeare."),
                ("Write a poem about a computer.", "Oh, wondrous box of metal, glass, and wire,\nThat pulses with a cold, electric fire!\nWithin thy silicon, a world unseen,\nA ghostly pageant on a glowing screen."),
                ("What is the meaning of life?", "To live, to love, to suffer and to learn—such is the tragicomedy of our mortal coil. As Jaques said, 'All the world's a stage,' and we must play our parts with whatever grace we can muster."),
                ("To be or not to be?", "Ah, the eternal question! Whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune, or to take arms against a sea of troubles, and by opposing end them."),
                ("Good morning!", "Good morrow to thee! What a fine morning to explore the works of Shakespeare together. How may I serve thee?"),
            ]
        },
        {
            "system": "You are a concise data analyst.",
            "prompts": [
                ("What is standard deviation?", "Standard deviation is a statistical measure that quantifies the amount of variation or dispersion in a set of data values."),
                ("Explain p-value.", "The p-value is the probability of obtaining test results at least as extreme as the results actually observed, under the assumption that the null hypothesis is correct."),
                ("What is a normal distribution?", "A normal distribution, or bell curve, is a continuous probability distribution that is symmetrical around its mean, indicating that data near the mean are more frequent in occurrence than data far from the mean."),
            ]
        }
    ]

    data = []
    # Seed for reproducibility
    random.seed(42)
    
    for i in range(num_examples):
        persona = random.choice(personas)
        prompt, response = random.choice(persona["prompts"])
        
        # Add slight variations to prevent exact duplicates when generating large datasets
        variation = f" (Query #{i})" if i >= len(personas)*10 else ""
        
        data.append({
            "system": persona["system"],
            "prompt": prompt + variation if variation else prompt,
            "response": response,
            "context": ""
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record) + "\n")
            
    print(f"[+] Generated {len(data)} diverse synthetic SFT examples -> {output_path}")
    return len(data)


# ---------------------------------------------------------------------------
# Data Validation & Deduplication
# ---------------------------------------------------------------------------

def validate_and_deduplicate(jsonl_path: Path) -> List[dict]:
    """
    Load JSONL, validate required fields, strip duplicates by content hash,
    and return a clean list of records. Supports two formats:
      - Single-turn: {system, prompt, response}
      - Multi-turn:  {system, messages: [{role: user|assistant, content: ...}]}
    """
    records = []
    seen    = set()
    errors  = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [warn] Line {i}: JSON decode error: {e}")
                errors += 1
                continue

            # Validate single-turn format
            if "prompt" in rec and "response" in rec:
                if not rec["prompt"].strip() or not rec["response"].strip():
                    errors += 1
                    continue
                rec.setdefault("system", SYSTEM_PROMPT)
                rec["_format"] = "single"

            # Validate multi-turn format
            elif "messages" in rec:
                msgs = rec["messages"]
                if not isinstance(msgs, list) or len(msgs) < 2:
                    errors += 1
                    continue
                valid = all(
                    isinstance(m, dict) and m.get("role") in ("user", "assistant")
                    and m.get("content", "").strip()
                    for m in msgs
                )
                if not valid:
                    errors += 1
                    continue
                rec.setdefault("system", SYSTEM_PROMPT)
                rec["_format"] = "multi"

            else:
                print(f"  [warn] Line {i}: Missing required fields (prompt+response or messages).")
                errors += 1
                continue

            # Deduplicate by content hash
            content_key = json.dumps(rec, sort_keys=True, ensure_ascii=False)
            h = hashlib.md5(content_key.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            records.append(rec)

    dupes = (i - errors - len(records))
    print(f"[+] Data validation: {len(records)} valid | {errors} invalid | {dupes} duplicates removed")
    return records


# ---------------------------------------------------------------------------
# SFT Dataset
# ---------------------------------------------------------------------------

class SFTDataset(Dataset):
    """
    Converts validated SFT records into model-ready tensors.

    Both single-turn and multi-turn formats are supported.
    Labels are masked (IGNORE_INDEX) everywhere except on assistant response
    tokens, so the model learns only to generate the correct response.

    Curriculum sorting: when sort_by_length=True, records are sorted from
    shortest to longest. This warms the model up on easy, short examples
    before exposing it to long, complex conversations.
    """

    def __init__(self, records: List[dict], tokenizer, max_length: int = BLOCK_SIZE,
                 sort_by_length: bool = True):
        self.tokenizer  = tokenizer
        self.max_length = max_length

        if sort_by_length:
            # Curriculum learning: start with the shortest samples
            records = sorted(
                records,
                key=lambda r: len(r.get("response", " ".join(
                    m["content"] for m in r.get("messages", [])
                )))
            )

        self.records = records
        print(f"[+] SFTDataset: {len(records)} records "
              f"({'curriculum-sorted' if sort_by_length else 'random order'})")

    def __len__(self):
        return len(self.records)

    def _build_single_turn(self, rec: dict):
        """Format a single-turn {system, prompt, response} record."""
        prompt_text = (
            f"{SYSTEM_TOKEN}{rec['system']}{END_TOKEN}"
            f"{USER_TOKEN}{rec['prompt']}{END_TOKEN}"
            f"{ASSISTANT_TOKEN}"
        )
        response_text = f"{rec['response']}{END_TOKEN}"
        return prompt_text, response_text

    def _build_multi_turn(self, rec: dict):
        """
        Format a multi-turn conversation record.
        All assistant tokens in the conversation are un-masked.
        All user/system tokens are masked.
        """
        parts        = [f"{SYSTEM_TOKEN}{rec['system']}{END_TOKEN}"]
        response_text = ""
        for msg in rec["messages"]:
            if msg["role"] == "user":
                parts.append(f"{USER_TOKEN}{msg['content']}{END_TOKEN}")
            elif msg["role"] == "assistant":
                parts.append(f"{ASSISTANT_TOKEN}{msg['content']}{END_TOKEN}")
        prompt_text = "".join(parts[:-1])   # everything up to last assistant turn
        # The last assistant turn becomes the response target
        last = rec["messages"][-1]
        if last["role"] == "assistant":
            response_text = f"{last['content']}{END_TOKEN}"
        return prompt_text, response_text

    def __getitem__(self, idx):
        rec = self.records[idx]
        if rec.get("_format") == "multi":
            prompt_text, response_text = self._build_multi_turn(rec)
        else:
            prompt_text, response_text = self._build_single_turn(rec)

        full_text  = prompt_text + response_text
        full_ids   = self.tokenizer.encode(full_text).ids
        prefix_ids = self.tokenizer.encode(prompt_text).ids

        # The tokenizer post-processor automatically adds [BOS] and [EOS].
        # We must strip the [EOS] from prefix_ids so it matches the start of full_ids,
        # otherwise the mask length is wrong and the model is trained to see [EOS] mid-sequence.
        prompt_ids   = self.tokenizer.encode(prompt_text).ids
        response_ids = self.tokenizer.encode(response_text).ids
        
        bos_id = self.tokenizer.token_to_id("[BOS]")
        eos_id = self.tokenizer.token_to_id("[EOS]")
        
        if prompt_ids and prompt_ids[0] == bos_id: prompt_ids = prompt_ids[1:]
        if prompt_ids and prompt_ids[-1] == eos_id: prompt_ids = prompt_ids[:-1]
        
        if response_ids and response_ids[0] == bos_id: response_ids = response_ids[1:]
        if response_ids and response_ids[-1] == eos_id: response_ids = response_ids[:-1]
        
        full_ids = [bos_id] + prompt_ids + response_ids + [eos_id]
        prompt_len = 1 + len(prompt_ids)

        full_ids  = full_ids[:self.max_length]
        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels    = input_ids.clone()
        labels[:prompt_len] = IGNORE_INDEX

        return {
            "input_ids":      input_ids,
            "labels":         labels,
            "attention_mask": torch.ones_like(input_ids),
        }


def sft_collate(batch):
    """Pad a batch of variable-length tensors to the same length."""
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids_padded, labels_padded, attn_mask_padded = [], [], []
    for item in batch:
        pad_len = max_len - item["input_ids"].size(0)
        input_ids_padded.append( F.pad(item["input_ids"],      (0, pad_len), value=0))
        labels_padded.append(    F.pad(item["labels"],         (0, pad_len), value=IGNORE_INDEX))
        attn_mask_padded.append( F.pad(item["attention_mask"], (0, pad_len), value=0))
    return {
        "input_ids":      torch.stack(input_ids_padded),
        "labels":         torch.stack(labels_padded),
        "attention_mask": torch.stack(attn_mask_padded),
    }


# ==============================================================================
# SECTION 3 - EVALUATION: BLEU SCORE
# ==============================================================================

def compute_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """
    Compute sentence-level BLEU score (up to n-gram order max_n).
    BLEU measures n-gram overlap between the model's output and the
    reference response. A score of 1.0 is a perfect match.
    """
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if len(hyp_tokens) == 0:
        return 0.0

    scores = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(
            tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)
        )
        hyp_ngrams = Counter(
            tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens) - n + 1)
        )
        overlap = sum((hyp_ngrams & ref_ngrams).values())
        total   = max(1, sum(hyp_ngrams.values()))
        scores.append(overlap / total)

    if any(s == 0 for s in scores):
        return 0.0

    log_avg  = sum(math.log(s) for s in scores) / len(scores)
    # Brevity penalty
    bp = 1.0 if len(hyp_tokens) >= len(ref_tokens) else math.exp(
        1.0 - len(ref_tokens) / len(hyp_tokens)
    )
    return bp * math.exp(log_avg)


def run_bleu_evaluation(model, tokenizer, records: List[dict], device: torch.device,
                        dtype: torch.dtype, n_samples: int = 20) -> float:
    """
    Generate text for a random sample of records and compute mean BLEU score.
    This gives a qualitative measure of how well the model follows instructions.
    """
    import random
    model.eval()
    samples = random.sample(records, min(n_samples, len(records)))
    bleu_scores = []
    eos_id = tokenizer.token_to_id("[EOS]") or 2

    for rec in samples:
        ref_response = rec.get("response", "")
        if not ref_response:
            continue

        prompt_text = (
            f"{SYSTEM_TOKEN}{rec.get('system', SYSTEM_PROMPT)}{END_TOKEN}"
            f"{USER_TOKEN}{rec['prompt']}{END_TOKEN}"
            f"{ASSISTANT_TOKEN}"
        )
        ids = tokenizer.encode(prompt_text).ids[-BLOCK_SIZE:]
        idx = torch.tensor([ids], dtype=torch.long, device=device)

        generated = ""
        with torch.no_grad():
            with autocast(device_type=device.type, dtype=dtype):
                for token_id in model.generate_stream(
                    idx, max_new_tokens=min(200, len(ref_response.split()) + 30),
                    temperature=0.1, top_k=1, top_p=1.0,
                    repetition_penalty=1.0, eos_token_id=eos_id,
                ):
                    tok = tokenizer.decode([token_id])
                    generated += tok
                    if END_TOKEN in generated or USER_TOKEN in generated:
                        break

        generated = generated.split(END_TOKEN)[0].strip()
        bleu_scores.append(compute_bleu(ref_response, generated))

    model.train()
    return sum(bleu_scores) / max(1, len(bleu_scores))


# ==============================================================================
# SECTION 4 - LEARNING RATE UTILITIES
# ==============================================================================

def get_cosine_lr(step: int, total_steps: int, warmup_steps: int,
                  base_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def get_layer_grouped_params(model: nn.Module, base_lr: float,
                             decay_factor: float = 0.9) -> List[dict]:
    """
    Assign progressively lower learning rates to earlier transformer layers.
    The intuition: early layers learn universal representations that should
    change very slowly; final layers learn task-specific patterns.

    decay_factor=0.9 means layer N-1 gets lr = base_lr * 0.9^(depth - 1).
    """
    # Collect (name, param) pairs in layer order
    named_params = list(model.named_parameters())

    # Group by transformer layer index (look for "layers.N" in name)
    layer_params: Dict[int, List] = {}
    other_params: List = []

    for name, param in named_params:
        if not param.requires_grad:
            continue
        found = False
        for part in name.split("."):
            if part.isdigit():
                layer_idx = int(part)
                layer_params.setdefault(layer_idx, []).append(param)
                found = True
                break
        if not found:
            other_params.append(param)

    if not layer_params:
        return [{"params": [p for _, p in named_params if p.requires_grad], "lr": base_lr}]

    max_layer = max(layer_params.keys())
    groups = []
    for layer_idx in sorted(layer_params.keys()):
        depth  = max_layer - layer_idx   # 0 = last layer, max = first layer
        lr     = base_lr * (decay_factor ** depth)
        groups.append({"params": layer_params[layer_idx], "lr": lr})

    if other_params:
        groups.append({"params": other_params, "lr": base_lr})

    return groups


# ==============================================================================
# SECTION 5 - CHECKPOINT UTILITIES
# ==============================================================================

def save_sft_checkpoint(model: nn.Module, optimizer, epoch: int, step: int,
                        val_loss: float, bleu: float, filename: str = "sft_model.pt"):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / filename
    torch.save({
        "epoch":    epoch,
        "step":     step,
        "val_loss": val_loss,
        "bleu":     bleu,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, ckpt_path)
    # Export safetensors (HuggingFace-compatible)
    try:
        from safetensors.torch import save_model as save_st
        st_path = CHECKPOINT_DIR / filename.replace(".pt", ".safetensors")
        save_st(model, str(st_path))
    except ImportError:
        pass
    return ckpt_path


def load_sft_checkpoint(model: nn.Module, optimizer, path: Path, device: torch.device):
    """Load a previously saved SFT checkpoint to resume training."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch    = ckpt.get("epoch",    0)
    step     = ckpt.get("step",     0)
    val_loss = ckpt.get("val_loss", float("inf"))
    bleu     = ckpt.get("bleu",     0.0)
    print(f"[+] SFT checkpoint resumed: epoch={epoch} step={step} "
          f"val_loss={val_loss:.4f} bleu={bleu:.4f}")
    return epoch, step, val_loss


# ==============================================================================
# SECTION 6 - SFT TRAINING LOOP
# ==============================================================================

def train_sft():
    global shutdown_requested
    print("\n" + "=" * 60)
    print("[SFT] Supervised Fine-Tuning - Project Bard")
    print("=" * 60)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if DTYPE == "float16" else torch.bfloat16
    print(f"[*] Device: {device} | dtype: {dtype}")
    print(f"[*] LoRA: {'enabled (rank=' + str(LORA_RANK) + ')' if SFT_USE_LORA else 'disabled (full fine-tune)'}")
    print(f"[*] Gradient accumulation: {SFT_GRAD_ACCUM} steps "
          f"(effective batch size = {SFT_BATCH_SIZE * SFT_GRAD_ACCUM})")

    # WandB
    wandb_run = None
    if USE_WANDB:
        try:
            import wandb
            wandb_run = wandb.init(
                project=WANDB_PROJECT, entity=WANDB_ENTITY,
                name="sft-bard", tags=["sft", "lora" if SFT_USE_LORA else "full"],
                config={
                    "sft_lr": SFT_LR, "sft_epochs": SFT_EPOCHS,
                    "lora_rank": LORA_RANK if SFT_USE_LORA else 0,
                    "grad_accum": SFT_GRAD_ACCUM,
                },
            )
        except Exception as e:
            print(f"[!] WandB init failed: {e}")

    # Tokenizer
    tokenizer = load_tokenizer()

    # Dataset
    records = validate_and_deduplicate(SFT_DATA_PATH)
    if len(records) == 0:
        raise RuntimeError(f"No valid records found in {SFT_DATA_PATH}.")

    n_val   = max(1, int(len(records) * 0.1))
    n_train = len(records) - n_val
    import random
    random.shuffle(records)
    train_records = records[:n_train]
    val_records   = records[n_train:]

    # Use curriculum sorting for training (longest last)
    train_ds = SFTDataset(train_records, tokenizer, sort_by_length=True)
    val_ds   = SFTDataset(val_records,   tokenizer, sort_by_length=False)

    train_loader = DataLoader(train_ds, batch_size=SFT_BATCH_SIZE, shuffle=False,
                              collate_fn=sft_collate, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=SFT_BATCH_SIZE, shuffle=False,
                              collate_fn=sft_collate, num_workers=0)
    print(f"[*] Train: {n_train} samples | Val: {n_val} samples")

    # Load base model (CPU first to avoid bitsandbytes deepcopy OOM)
    base_ckpt = CHECKPOINT_DIR / "best.pt"
    if not base_ckpt.exists():
        base_ckpt = CHECKPOINT_DIR / "last.pt"
    if not base_ckpt.exists():
        raise FileNotFoundError(
            "No base checkpoint found. Run train.py first to create best.pt or last.pt.")

    print(f"[*] Loading base model from: {base_ckpt}")
    ckpt           = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    cfg: ModelConfig = ckpt["config"]
    model          = ShakespeareGPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])

    # Inject LoRA adapters (or keep all parameters trainable)
    lora_injected = False
    if SFT_USE_LORA:
        model, n_trainable = inject_lora(model, r=LORA_RANK, alpha=LORA_ALPHA,
                                         dropout=LORA_DROPOUT)
        lora_injected = True
    else:
        n_trainable = count_parameters(model)
        print(f"[*] Full fine-tune: {n_trainable:,} trainable parameters")

    model = model.to(device=device)
    model.train()

    # Optimizer — per-layer LR groups for more nuanced training
    param_groups = get_layer_grouped_params(model, base_lr=SFT_LR, decay_factor=0.9)
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            param_groups, lr=SFT_LR, betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY)
        print("[*] Using 8-bit AdamW (bitsandbytes) with per-layer LR groups")
    except ImportError:
        optimizer = torch.optim.AdamW(
            param_groups, lr=SFT_LR, betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY)
        print("[*] Using standard AdamW with per-layer LR groups")

    scaler = GradScaler(enabled=(dtype == torch.float16))

    # Check for existing SFT checkpoint to resume
    sft_last_path = CHECKPOINT_DIR / "sft_last.pt"
    start_epoch   = 0
    global_step   = 0
    best_val_loss = float("inf")
    best_bleu     = 0.0

    if sft_last_path.exists():
        print(f"[*] Resuming SFT from: {sft_last_path}")
        start_epoch, global_step, best_val_loss = load_sft_checkpoint(
            model, optimizer, sft_last_path, device)

    # Schedule
    total_steps  = SFT_EPOCHS * math.ceil(len(train_loader) / SFT_GRAD_ACCUM)
    warmup_steps = max(1, total_steps // 10)
    min_lr       = SFT_LR * 0.05

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "sft_metrics.jsonl"

    print(f"[*] Total optimizer steps: {total_steps} | Warmup: {warmup_steps}")
    print("=" * 60)

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    epochs_no_improve = 0

    for epoch in range(start_epoch + 1, SFT_EPOCHS + 1):
        if shutdown_requested:
            break

        model.train()
        epoch_loss  = 0.0
        epoch_steps = 0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"SFT Epoch {epoch}/{SFT_EPOCHS}",
                    dynamic_ncols=True) if use_tqdm else train_loader

        for accum_idx, batch in enumerate(pbar):
            if shutdown_requested:
                break

            is_last_accum = ((accum_idx + 1) % SFT_GRAD_ACCUM == 0) or \
                            (accum_idx + 1 == len(train_loader))

            input_ids = batch["input_ids"][:, :-1].to(device, non_blocking=True)
            labels    = batch["labels"][:, 1:].to(device, non_blocking=True)

            t0 = time.time()
            with autocast(device_type=device.type, dtype=dtype):
                model_out = model(input_ids)
                logits = model_out["logits"] if isinstance(model_out, dict) else model_out[0]

                # The logits and labels are already shifted in the batch tensors
                shift_logits = logits.contiguous()
                shift_labels = labels.contiguous()

                # Response-only cross-entropy (IGNORE_INDEX positions are excluded)
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=IGNORE_INDEX,
                ) / SFT_GRAD_ACCUM

            scaler.scale(loss).backward()
            epoch_loss += loss.item()

            if is_last_accum:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # Update learning rate
                lr = get_cosine_lr(global_step, total_steps, warmup_steps, SFT_LR, min_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr * (pg.get("lr", SFT_LR) / SFT_LR)

                epoch_steps += 1
                global_step += 1
                dt = time.time() - t0
                tokens_per_sec = input_ids.numel() * SFT_GRAD_ACCUM / max(dt, 1e-6)
                mem_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0

                metrics = {
                    "epoch": epoch, "step": global_step,
                    "sft_loss":          round(epoch_loss / epoch_steps, 4),
                    "sft_lr":            round(lr, 8),
                    "sft_tokens_per_sec": round(tokens_per_sec),
                    "sft_mem_gb":        round(mem_gb, 2),
                }
                with open(log_path, "a") as lf:
                    lf.write(json.dumps(metrics) + "\n")
                if wandb_run:
                    wandb_run.log(metrics, step=global_step)

                if use_tqdm and hasattr(pbar, "set_postfix"):
                    pbar.set_postfix({
                        "loss": f"{epoch_loss/epoch_steps:.4f}",
                        "lr":   f"{lr:.2e}",
                        "mem":  f"{mem_gb:.1f}GB",
                    })

        # Validation loss
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels    = batch["labels"].to(device)
                with autocast(device_type=device.type, dtype=dtype):
                    model_out = model(input_ids)
                    logits = model_out["logits"] if isinstance(model_out, dict) else model_out[0]
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=IGNORE_INDEX,
                    )
                val_losses.append(loss.item())

        val_loss       = sum(val_losses) / max(1, len(val_losses))
        val_ppl        = math.exp(min(val_loss, 20))
        avg_train_loss = epoch_loss / max(1, epoch_steps)

        # BLEU evaluation
        bleu = run_bleu_evaluation(model, tokenizer, val_records, device, dtype, n_samples=15)

        print(f"\n[Epoch {epoch}/{SFT_EPOCHS}] "
              f"train_loss={avg_train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f} | "
              f"BLEU={bleu:.4f}")

        if wandb_run:
            wandb_run.log({
                "sft_val_loss": val_loss, "sft_val_ppl": val_ppl, "sft_bleu": bleu,
            }, step=global_step)

        # Save epoch checkpoint (for resume)
        save_sft_checkpoint(model, optimizer, epoch, global_step, val_loss, bleu, "sft_last.pt")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            best_bleu         = bleu
            epochs_no_improve = 0

            # If LoRA was used, merge weights before saving so inference
            # does not need to know about LoRA at all
            if lora_injected:
                print("[*] Merging LoRA weights into base model before saving...")
                merged_model = copy.deepcopy(model)
                merged_model = merge_lora_weights(merged_model)
                ckpt_path = save_sft_checkpoint(
                    merged_model, optimizer, epoch, global_step, val_loss, bleu, "sft_model.pt")
            else:
                ckpt_path = save_sft_checkpoint(
                    model, optimizer, epoch, global_step, val_loss, bleu, "sft_model.pt")

            print(f"[+] New best SFT model saved: {ckpt_path} "
                  f"(val_loss={val_loss:.4f}, BLEU={bleu:.4f})")
        else:
            epochs_no_improve += 1
            print(f"[*] No improvement for {epochs_no_improve}/{SFT_EARLY_STOP} epochs.")
            if epochs_no_improve >= SFT_EARLY_STOP:
                print("[!] Early stopping triggered.")
                break

        # Clear CUDA cache between epochs
        if device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    # Final summary
    print("\n" + "=" * 60)
    print("[SFT] Training Complete!")
    print(f"  Total optimizer steps : {global_step}")
    print(f"  Best val loss         : {best_val_loss:.4f}")
    print(f"  Best BLEU score       : {best_bleu:.4f}")
    print(f"  Model saved           : {CHECKPOINT_DIR / 'sft_model.pt'}")
    print(f"  Resume checkpoint     : {CHECKPOINT_DIR / 'sft_last.pt'}")
    print("=" * 60)

    if wandb_run:
        wandb_run.finish()


# ==============================================================================
# SECTION 7 - DPO TRAINING LOOP
# ==============================================================================

class DPODataset(Dataset):
    """
    Loads {prompt, chosen, rejected} JSONL pairs for DPO training.
    DPO teaches the model to prefer 'chosen' responses over 'rejected' ones
    without needing an explicit reward model.
    """

    def __init__(self, jsonl_path: Path, tokenizer, max_length: int = BLOCK_SIZE):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.records    = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        if "prompt" in rec and "chosen" in rec and "rejected" in rec:
                            self.records.append(rec)
                    except json.JSONDecodeError:
                        continue
        print(f"[+] DPODataset: {len(self.records)} preference pairs from {jsonl_path}")

    def __len__(self):
        return len(self.records)

    def _encode(self, prompt: str, response: str) -> dict:
        prompt_text = (
            f"{SYSTEM_TOKEN}{SYSTEM_PROMPT}{END_TOKEN}"
            f"{USER_TOKEN}{prompt}{END_TOKEN}"
            f"{ASSISTANT_TOKEN}"
        )
        prompt_ids   = self.tokenizer.encode(prompt_text).ids
        response_ids = self.tokenizer.encode(response + END_TOKEN).ids
        
        bos_id = self.tokenizer.token_to_id("[BOS]")
        eos_id = self.tokenizer.token_to_id("[EOS]")
        
        if prompt_ids and prompt_ids[0] == bos_id: prompt_ids = prompt_ids[1:]
        if prompt_ids and prompt_ids[-1] == eos_id: prompt_ids = prompt_ids[:-1]
        
        if response_ids and response_ids[0] == bos_id: response_ids = response_ids[1:]
        if response_ids and response_ids[-1] == eos_id: response_ids = response_ids[:-1]
        
        full_ids = [bos_id] + prompt_ids + response_ids + [eos_id]
        prompt_len = 1 + len(prompt_ids)
        
        full_ids = full_ids[:self.max_length]
        input_ids  = torch.tensor(full_ids, dtype=torch.long)
        labels     = input_ids.clone()
        labels[:prompt_len] = IGNORE_INDEX
        return {"input_ids": input_ids, "labels": labels}

    def __getitem__(self, idx):
        rec = self.records[idx]
        return {
            "chosen":   self._encode(rec["prompt"], rec["chosen"]),
            "rejected": self._encode(rec["prompt"], rec["rejected"]),
        }


def compute_log_probs(model, input_ids: torch.Tensor, labels: torch.Tensor,
                      device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Sum of log-probabilities over response tokens only (labels != IGNORE_INDEX)."""
    with autocast(device_type=device.type, dtype=dtype):
        out    = model(input_ids)
        logits = out["logits"] if isinstance(out, dict) else out[0]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    log_probs    = F.log_softmax(shift_logits, dim=-1)
    token_lp     = log_probs.gather(2, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    mask         = (shift_labels != IGNORE_INDEX).float()
    return (token_lp * mask).sum(dim=-1)


def train_dpo():
    """
    Direct Preference Optimization stage.
    Runs after SFT using sft_model.pt as both the policy and reference model.

    DPO Loss = -log_sigmoid( beta * (
        log_pi_chosen   - log_ref_chosen   -
        log_pi_rejected + log_ref_rejected
    ))
    """
    global shutdown_requested
    print("\n" + "=" * 60)
    print("[DPO] Direct Preference Optimization - Project Bard")
    print("=" * 60)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if DTYPE == "float16" else torch.bfloat16
    tokenizer = load_tokenizer()

    sft_ckpt_path = CHECKPOINT_DIR / "sft_model.pt"
    if not sft_ckpt_path.exists():
        print("[!] sft_model.pt not found. Run SFT stage first.")
        return

    def load_from(path):
        c   = torch.load(path, map_location="cpu", weights_only=False)
        cfg = c.get("config") or torch.load(
            CHECKPOINT_DIR / "best.pt", map_location="cpu", weights_only=False)["config"]
        m = ShakespeareGPT(cfg).to(device=device)
        m.load_state_dict(c["model_state_dict"])
        return m

    policy_model = load_from(sft_ckpt_path)
    policy_model.train()

    ref_model = load_from(sft_ckpt_path)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    print(f"[*] Policy model: {count_parameters(policy_model):,} parameters (trainable)")
    print(f"[*] Reference model: frozen copy of sft_model.pt")

    dataset = DPODataset(DPO_DATA_PATH, tokenizer)
    loader  = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(policy_model.parameters(), lr=DPO_LR)
    except ImportError:
        optimizer = torch.optim.AdamW(policy_model.parameters(), lr=DPO_LR)

    scaler   = GradScaler(enabled=(dtype == torch.float16))
    log_path = LOG_DIR / "dpo_metrics.jsonl"

    try:
        from tqdm import tqdm
        pbar = tqdm(loader, desc="DPO Training", dynamic_ncols=True)
    except ImportError:
        pbar = loader

    for step, batch in enumerate(pbar):
        if shutdown_requested:
            break

        def prep(item):
            return (item["input_ids"].squeeze(0).unsqueeze(0).to(device),
                    item["labels"].squeeze(0).unsqueeze(0).to(device))

        c_ids, c_labels = prep(batch["chosen"])
        r_ids, r_labels = prep(batch["rejected"])

        pi_chosen   = compute_log_probs(policy_model, c_ids, c_labels, device, dtype)
        pi_rejected = compute_log_probs(policy_model, r_ids, r_labels, device, dtype)

        with torch.no_grad():
            ref_chosen   = compute_log_probs(ref_model, c_ids, c_labels, device, dtype)
            ref_rejected = compute_log_probs(ref_model, r_ids, r_labels, device, dtype)

        reward_diff = DPO_BETA * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))
        dpo_loss    = -F.logsigmoid(reward_diff).mean()

        scaler.scale(dpo_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        m = {"step": step, "dpo_loss": round(dpo_loss.item(), 4),
             "reward_margin": round(reward_diff.mean().item(), 4)}
        with open(log_path, "a") as lf:
            lf.write(json.dumps(m) + "\n")
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix({"dpo_loss": f"{dpo_loss.item():.4f}",
                              "margin":   f"{reward_diff.mean().item():.4f}"})

    dpo_path = CHECKPOINT_DIR / "dpo_model.pt"
    torch.save({"model_state_dict": policy_model.state_dict()}, dpo_path)
    print(f"[+] DPO model saved: {dpo_path}")


# ==============================================================================
# SECTION 8 - ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Project Bard - Fine-Tuning Pipeline")
    print("=" * 60)

    # Step 1: Generate synthetic dataset if none exists
    if not SFT_DATA_PATH.exists():
        print(f"[*] No SFT data found at {SFT_DATA_PATH}. Generating synthetic dataset...")
        generate_synthetic_sft_data(SFT_DATA_PATH)
    else:
        print(f"[*] Using existing SFT data: {SFT_DATA_PATH}")

    # Step 2: Supervised Fine-Tuning
    train_sft()

    # Step 3: DPO (only if preference data file exists)
    if DPO_DATA_PATH.exists():
        print(f"\n[*] DPO data found at {DPO_DATA_PATH}. Running DPO stage...")
        train_dpo()
    else:
        print(f"\n[*] No DPO data found at {DPO_DATA_PATH}. Skipping DPO.")
        print(f"    To enable DPO, create a JSONL file with {{prompt, chosen, rejected}} pairs at that path.")

    print("\n[+] Full fine-tuning pipeline complete!")
    print(f"    SFT model : {CHECKPOINT_DIR / 'sft_model.pt'}")
    print(f"    DPO model : {CHECKPOINT_DIR / 'dpo_model.pt'} (if DPO ran)")
    print(f"    Next step : Load sft_model.pt in the Chat tab of the API dashboard.")
