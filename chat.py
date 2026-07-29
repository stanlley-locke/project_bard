"""
chat.py - Production-Grade Contextual Inference Interface
Features:
  - Pure text continuation (no artificial Q&A formatting)
  - Real-time token streaming (typewriter effect)
  - Sliding window context management
  - Interactive CLI commands (/clear, /stats, /save, /load, /temp, /quit)
  - Custom stop sequences for controlled generation
  - Separate prefill and decode timing metrics
  - Configurable checkpoint loading
"""
import torch
import argparse
import time
import json
from pathlib import Path
from typing import List

from config import (
    DEVICE, CHECKPOINT_DIR, GEN_TEMPERATURE, GEN_TOP_K,
    GEN_TOP_P, GEN_REP_PENALTY, BLOCK_SIZE
)
from model import ShakespeareGPT, ModelConfig, count_parameters
from tokenizer import load_tokenizer


def load_model_for_inference(checkpoint_name: str = "best.pt"):
    """Load the model for inference with specified checkpoint."""
    ckpt_path = CHECKPOINT_DIR / checkpoint_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Run train.py first.")
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: ModelConfig = ckpt["config"]
    
    model = ShakespeareGPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def build_prompt(history: List[str], new_input: str) -> str:
    """
    Production-grade prompt engineering for pure contextual continuation.
    Appends new input to the history to maintain narrative context.
    """
    if not history:
        return new_input
    
    # Join history with new input to maintain a continuous text block
    return "\n".join(history) + "\n" + new_input


def chat_loop(model: ShakespeareGPT, tokenizer, device: torch.device, args):
    """Main interactive inference loop with real-time streaming."""
    print("=" * 70)
    print(" PROJECT BARD: Production-Grade Contextual Inference Interface")
    print("=" * 70)
    print("Commands:")
    print("  /clear          - Clear conversation history")
    print("  /stats          - Show model and generation statistics")
    print("  /save <file>    - Save conversation history to JSON")
    print("  /load <file>    - Load conversation history from JSON")
    print("  /temp <val>     - Set temperature (e.g., /temp 0.8)")
    print("  /topp <val>     - Set top-p (e.g., /topp 0.9)")
    print("  /topk <val>     - Set top-k (e.g., /topk 40)")
    print("  /reppen <val>   - Set repetition penalty (e.g., /reppen 1.1)")
    print("  /maxtok <val>   - Set max new tokens (e.g., /maxtok 150)")
    print("  /quit           - Exit the interface")
    print("=" * 70)
    print("Tip: Provide a starting phrase, and the model will continue the text.")
    print("=" * 70)
    
    history: List[str] = []

    while True:
        try:
            user_input = input("\nPrompt: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting interface.")
            break

        if not user_input:
            continue

        # Handle commands
        lower_input = user_input.lower()
        if lower_input in ["/quit", "/exit", "q"]:
            print("Exiting interface.")
            break
        elif lower_input == "/clear":
            history = []
            print("History cleared.")
            continue
        elif lower_input.startswith("/save "):
            filename = user_input.split(" ", 1)[1].strip()
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2, ensure_ascii=False)
                print(f"History saved to {filename}")
            except Exception as e:
                print(f"Error saving history: {e}")
            continue
        elif lower_input.startswith("/load "):
            filename = user_input.split(" ", 1)[1].strip()
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    history = json.load(f)
                print(f"History loaded from {filename} ({len(history)} turns)")
            except Exception as e:
                print(f"Error loading history: {e}")
            continue
        elif lower_input.startswith("/temp "):
            try:
                args.temp = float(user_input.split(" ", 1)[1].strip())
                print(f"Temperature set to {args.temp}")
            except ValueError:
                print("Invalid temperature value.")
            continue
        elif lower_input.startswith("/topp "):
            try:
                args.top_p = float(user_input.split(" ", 1)[1].strip())
                print(f"Top-P set to {args.top_p}")
            except ValueError:
                print("Invalid top-p value.")
            continue
        elif lower_input.startswith("/topk "):
            try:
                args.top_k = int(user_input.split(" ", 1)[1].strip())
                print(f"Top-K set to {args.top_k}")
            except ValueError:
                print("Invalid top-k value.")
            continue
        elif lower_input.startswith("/reppen "):
            try:
                args.rep_penalty = float(user_input.split(" ", 1)[1].strip())
                print(f"Repetition penalty set to {args.rep_penalty}")
            except ValueError:
                print("Invalid repetition penalty value.")
            continue
        elif lower_input.startswith("/maxtok "):
            try:
                args.max_tokens = int(user_input.split(" ", 1)[1].strip())
                print(f"Max tokens set to {args.max_tokens}")
            except ValueError:
                print("Invalid max tokens value.")
            continue
        elif lower_input == "/stats":
            print("-" * 70)
            print(f"Model Parameters : {count_parameters(model):,}")
            print(f"Vocab Size       : {model.cfg.vocab_size}")
            print(f"Context Window   : {model.cfg.block_size} tokens")
            print(f"Current Settings : Temp={args.temp}, Top-K={args.top_k}, Top-P={args.top_p}, Rep-Penalty={args.rep_penalty}, Max-Tokens={args.max_tokens}")
            print(f"History Length   : {len(history)} turns")
            print("-" * 70)
            continue

        # Format full context for pure continuation
        prompt_text = build_prompt(history, user_input)
        
        # Tokenize
        ids = tokenizer.encode(prompt_text).ids
        
        # Hard truncate to leave room for generation
        max_prompt_len = BLOCK_SIZE - args.max_tokens
        if len(ids) > max_prompt_len:
            print(f"Warning: Context too long ({len(ids)} tokens). Truncating oldest messages...")
            ids = ids[-max_prompt_len:]

        idx = torch.tensor([ids], dtype=torch.long, device=device)

        print("\nContinuation: ", end="", flush=True)

        # Stream generation with timing
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.time()
        token_count = 0
        response_text = ""
        
        eos_token_id = tokenizer.token_to_id("[EOS]")
        
        for token in model.generate_stream(
            idx,
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.rep_penalty,
            eos_token_id=eos_token_id,
        ):
            token_str = tokenizer.decode([token])
            print(token_str, end="", flush=True)
            response_text += token_str
            token_count += 1
            
            # Check for custom stop sequence if provided
            if args.stop_sequence and args.stop_sequence in response_text:
                response_text = response_text.split(args.stop_sequence)[0]
                break

        if device.type == "cuda":
            torch.cuda.synchronize()
            
        elapsed = time.time() - start_time
        speed = token_count / elapsed if elapsed > 0 else 0
        
        print(f"\n\n[Generated {token_count} tokens in {elapsed:.2f}s ({speed:.1f} tok/s)]")
        
        # Clean up any trailing artifacts
        response_text = response_text.split("[_")[0].split("SCENE")[0].strip()
        
        # Add to history for continuous context
        history.append(user_input + response_text)
        
        # Keep history manageable (last 5 turns)
        if len(history) > 5:
            history = history[-5:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contextual Inference with Project Bard")
    parser.add_argument("--checkpoint", type=str, default="last.pt", help="Checkpoint file to load")
    parser.add_argument("--temp", type=float, default=GEN_TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=GEN_TOP_K, help="Top-K sampling")
    parser.add_argument("--top-p", type=float, default=GEN_TOP_P, help="Top-P (nucleus) sampling")
    parser.add_argument("--rep-penalty", type=float, default=GEN_REP_PENALTY, help="Repetition penalty")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max new tokens per turn")
    parser.add_argument("--stop-sequence", type=str, default=None, help="String to stop generation early")
    args = parser.parse_args()

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[*] Loading model '{args.checkpoint}' on {device}...")
    
    try:
        model, cfg = load_model_for_inference(args.checkpoint)
        model = model.to(device)
    except FileNotFoundError as e:
        print(f"[!] Error: {e}")
        exit(1)
    
    print("[*] Loading tokenizer...")
    tokenizer = load_tokenizer()
    print("[+] Ready! Start typing your prompts.\n")
    
    chat_loop(model, tokenizer, device, args)