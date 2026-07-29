"""
train.py - Production-Grade T4-optimized training
Features:
  - float16 mixed precision (T4-native)
  - Gradient accumulation (effective batch = 64)
  - WandB logging with resume capability
  - Cosine LR with warmup
  - Checkpoint resumption (crash recovery)
  - Early stopping to prevent overfitting
  - Graceful interruption handling (Ctrl+C saves checkpoint)
  - Memory management (GPU cache clearing during eval)
"""
import math
import time
import torch
import signal
import gc
from torch.amp import autocast, GradScaler

from config import (
    DEVICE, DTYPE, LEARNING_RATE, WEIGHT_DECAY, BETA1, BETA2,
    GRAD_CLIP, WARMUP_STEPS, MAX_STEPS, MIN_LR_RATIO,
    LOG_INTERVAL, EVAL_INTERVAL, SAVE_INTERVAL, CHECKPOINT_DIR, SEED,
    BATCH_SIZE, GRAD_ACCUM_STEPS, VOCAB_SIZE, USE_WANDB,
    WANDB_PROJECT, WANDB_ENTITY
)
from model import ShakespeareGPT, ModelConfig, count_parameters
from dataset import get_dataloader, split_data

# Early stopping patience (in steps)
EARLY_STOPPING_PATIENCE = 1000 

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    print("\n[!] Shutdown signal received. Saving final checkpoint and exiting gracefully...")
    shutdown_requested = True

# Register signal handlers for Ctrl+C (SIGINT) and termination (SIGTERM)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def get_lr(step: int) -> float:
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS
    if step >= MAX_STEPS:
        return LEARNING_RATE * MIN_LR_RATIO
    progress = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return LEARNING_RATE * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * cosine)


@torch.no_grad()
def evaluate(model: ShakespeareGPT, device: torch.device, dtype: torch.dtype) -> dict:
    model.eval()
    out = {}
    for split in ("train", "val"):
        dl = get_dataloader(split, batch_size=BATCH_SIZE, shuffle=False)
        losses = []
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast(device_type=device.type, dtype=dtype):
                model_out = model(x, y)
                # Handle both dict (new) and tuple (legacy) return types
                loss = model_out["loss"] if isinstance(model_out, dict) else model_out[1]
            losses.append(loss.item())
        mean_loss = sum(losses) / max(1, len(losses))
        out[f"{split}_loss"] = mean_loss
        out[f"{split}_perplexity"] = math.exp(min(mean_loss, 20))  # cap for safety
    
    # Prevent memory leaks during long training runs
    if device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
        
    model.train()
    return out


def save_checkpoint(model, optimizer, step, val_loss, cfg, filename="last.pt"):
    """Centralized checkpoint saving to ensure consistency."""
    ckpt_path = CHECKPOINT_DIR / filename
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "config": cfg,
    }, ckpt_path)
    return ckpt_path


def train():
    global shutdown_requested
    print("=" * 60)
    print("[PHASE 5] Pre-Training (Production-Grade)")
    print("=" * 60)

    # Initialize WandB
    wandb_run = None
    if USE_WANDB:
        try:
            import wandb
            wandb_run = wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                resume="allow",  # Allow resuming the wandb run if it crashed
                config={k: v for k, v in locals().items() if isinstance(v, (int, float, str, bool))},
            )
        except Exception as e:
            print(f"[!] WandB init failed: {e}. Continuing without WandB.")

    torch.manual_seed(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if DTYPE == "float16" else torch.bfloat16
    print(f"[*] Device: {device}, dtype: {dtype}")
    print(f"[*] Effective batch size: {BATCH_SIZE} * {GRAD_ACCUM_STEPS} = {BATCH_SIZE * GRAD_ACCUM_STEPS}")

    if device.type == "cuda":
        print(f"[*] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[*] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    split_data()

    cfg = ModelConfig(vocab_size=VOCAB_SIZE)
    model = ShakespeareGPT(cfg).to(device)
    print(f"[+] Model parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2),
        weight_decay=WEIGHT_DECAY,
        fused=True,  # Faster on modern PyTorch
    )

    scaler = GradScaler(enabled=(dtype == torch.float16))

    # --- RESUME LOGIC ---
    start_step = 0
    best_val_loss = float("inf")
    steps_without_improvement = 0
    resume_path = CHECKPOINT_DIR / "last.pt"
    
    if resume_path.exists():
        print(f"[*] Found existing checkpoint at {resume_path}. Resuming training...")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint["step"]
        best_val_loss = checkpoint.get("val_loss", float("inf"))
        print(f"[+] Resumed from step {start_step} with best_val_loss: {best_val_loss:.4f}")
    else:
        print("[*] No previous checkpoint found. Starting fresh training.")

    train_loader = get_dataloader("train", batch_size=BATCH_SIZE, shuffle=True)
    data_iter = iter(train_loader)

    model.train()
    step = start_step
    t0 = time.time()
    accumulated_loss = 0.0

    while step < MAX_STEPS and not shutdown_requested:
        # Update LR per optimizer step
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation loop
        for accum_step in range(GRAD_ACCUM_STEPS):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            is_last_accum = (accum_step == GRAD_ACCUM_STEPS - 1)

            with autocast(device_type=device.type, dtype=dtype):
                model_out = model(x, y)
                # Handle both dict (new) and tuple (legacy) return types
                loss = model_out["loss"] if isinstance(model_out, dict) else model_out[1]
                loss = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()
            accumulated_loss += loss.item()

            if is_last_accum:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        # Logging
        if step % LOG_INTERVAL == 0:
            dt = time.time() - t0
            t0 = time.time()
            avg_loss = accumulated_loss / GRAD_ACCUM_STEPS
            accumulated_loss = 0.0
            mem_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
            
            # Reset max memory tracking for accurate per-interval measurement
            if device.type == "cuda":
                torch.cuda.reset_max_memory_allocated()

            log_data = {
                "step": step,
                "loss": avg_loss,
                "lr": lr,
                "iter_time_ms": dt * 1000 / LOG_INTERVAL,
                "gpu_mem_gb": mem_gb,
            }
            print(
                f"[step {step:04d}] loss={avg_loss:.4f} | lr={lr:.2e} | "
                f"iter_time={dt * 1000 / LOG_INTERVAL:.1f}ms | "
                f"gpu_mem={mem_gb:.2f}GB"
            )
            if wandb_run:
                wandb_run.log(log_data, step=step)

        # Evaluation
        if step > 0 and step % EVAL_INTERVAL == 0:
            metrics = evaluate(model, device, dtype)
            print(
                f"[eval step {step}] "
                f"train_loss={metrics['train_loss']:.4f} train_ppl={metrics['train_perplexity']:.2f} | "
                f"val_loss={metrics['val_loss']:.4f} val_ppl={metrics['val_perplexity']:.2f}"
            )
            if wandb_run:
                wandb_run.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)

            # Early Stopping & Best Model Check
            if metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                steps_without_improvement = 0
                ckpt_path = save_checkpoint(model, optimizer, step, best_val_loss, cfg, "best.pt")
                print(f"[+] New best model saved: {ckpt_path}")
            else:
                steps_without_improvement += EVAL_INTERVAL
                if steps_without_improvement >= EARLY_STOPPING_PATIENCE:
                    print(f"[!] Early stopping triggered. No improvement in {EARLY_STOPPING_PATIENCE} steps.")
                    shutdown_requested = True

        # Periodic Checkpoint (acts as the resume point)
        if step > 0 and step % SAVE_INTERVAL == 0:
            ckpt_path = save_checkpoint(model, optimizer, step, best_val_loss, cfg, "last.pt")
            print(f"[*] Checkpoint saved: {ckpt_path}")

        step += 1

    # Final save on exit (normal completion or interrupted)
    if step > start_step:
        final_ckpt = save_checkpoint(model, optimizer, step, best_val_loss, cfg, "last.pt")
        print(f"[*] Final checkpoint saved: {final_ckpt}")

    if wandb_run:
        wandb_run.finish()
    print("[+] Training complete.")
    return model


if __name__ == "__main__":
    train()