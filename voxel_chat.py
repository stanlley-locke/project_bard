"""
voxel_chat.py - VoxelPath Inference Interface
Uses the distilled voxel model for geometric text generation.
"""
import argparse
import time
import torch
from pathlib import Path
from typing import List

try:
    from config import DEVICE, CHECKPOINT_DIR
except ImportError:
    DEVICE = "cuda"
    CHECKPOINT_DIR = Path("checkpoints")

from tokenizer import load_tokenizer
from voxel_engine import VoxelEngine


def load_voxel_system():
    """Load the distilled voxel model."""
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    
    voxel_path = CHECKPOINT_DIR / "voxel_model"
    if not voxel_path.with_suffix('.grid.npy').exists():
        raise FileNotFoundError(f"Voxel model not found at {voxel_path}. Run voxel_train.py first.")
    
    print("[*] Loading distilled voxel model...")
    engine = VoxelEngine()
    engine.load(voxel_path)
    
    print("[*] Loading tokenizer...")
    tokenizer = load_tokenizer()
    
    stats = engine.get_stats()
    print(f"[+] Voxel model loaded. Memory: {stats['memory_mb']:.2f} MB")
    
    return engine, tokenizer, device


def voxel_chat_loop(engine: VoxelEngine, tokenizer, device: torch.device, args):
    """Main interactive chat loop for the Voxel Engine."""
    print("=" * 70)
    print(" VOXEL LM: Geometric Inference Interface")
    print("=" * 70)
    print("Commands:")
    print("  /clear       - Clear conversation history")
    print("  /temp <val>  - Set temperature (e.g., /temp 0.8)")
    print("  /stats       - Show voxel model statistics")
    print("  /quit        - Exit the interface")
    print("=" * 70)
    print("Note: This uses the distilled voxel model (no GPU required for inference).")
    print("=" * 70)
    
    history: List[str] = []

    while True:
        try:
            user_input = input("\nPrompt: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting Voxel Chat.")
            break

        if not user_input:
            continue

        lower_input = user_input.lower()
        if lower_input in ["/quit", "/exit", "q"]:
            print("Exiting Voxel Chat.")
            break
        elif lower_input == "/clear":
            history = []
            print("History cleared.")
            continue
        elif lower_input == "/stats":
            stats = engine.get_stats()
            print("\n--- Voxel Model Statistics ---")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            print("----------------------------\n")
            continue
        elif lower_input.startswith("/temp "):
            try:
                args.temp = float(user_input.split(" ", 1)[1].strip())
                print(f"Temperature set to {args.temp}")
            except ValueError:
                print("Invalid temperature value.")
            continue

        # Encode prompt
        encoded = tokenizer.encode(user_input)
        prompt_ids = encoded.ids
        
        # Strip trailing EOS
        eos_id = tokenizer.token_to_id("[EOS]")
        if prompt_ids and prompt_ids[-1] == eos_id:
            prompt_ids = prompt_ids[:-1]

        print("\nVoxel: ", end="", flush=True)
        
        start_time = time.time()
        
        # Generate using voxel geometric pathfinding
        generated_ids = engine.generate(
            prompt_tokens=prompt_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
            repetition_penalty=args.rep_penalty,
            top_k=args.top_k,
            top_p=args.top_p
        )
        
        # Decode only new tokens
        new_ids = generated_ids[len(prompt_ids):]
        response = tokenizer.decode(new_ids).strip()
        
        print(response, end="", flush=True)
        
        elapsed = time.time() - start_time
        tokens_generated = len(new_ids)
        speed = tokens_generated / elapsed if elapsed > 0 else 0
        
        print(f"\n\n[Generated {tokens_generated} tokens in {elapsed:.2f}s ({speed:.1f} tok/s)]")
        
        history.append(user_input + " " + response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoxelPath Chat")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=150, help="Max tokens to generate")
    parser.add_argument("--rep-penalty", type=float, default=1.2, help="Repetition penalty")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-P sampling")
    args = parser.parse_args()

    print("[*] Initializing VoxelPath Engine...")
    try:
        engine, tokenizer, device = load_voxel_system()
        print("[+] Ready! Start typing your prompts.\n")
        voxel_chat_loop(engine, tokenizer, device, args)
    except FileNotFoundError as e:
        print(f"[!] Error: {e}")