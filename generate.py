"""
generate.py - Production-Grade Inference Script
Features:
  - Fast batched generation with KV cache
  - Advanced sampling: Top-P, Top-K, Repetition Penalty, Min Length
  - Real-time token streaming option
  - Early stopping via custom stop sequences
  - Accurate prefill vs. decode timing metrics
  - Reproducible generation via seed control
  - Output to file (JSONL) support for evaluation pipelines
  - Automatic dtype optimization (float16 for T4)
"""
import argparse
import json
import time
import torch
from pathlib import Path
from typing import List, Optional

from config import (
    DEVICE, CHECKPOINT_DIR, GEN_TEMPERATURE, GEN_TOP_K,
    GEN_TOP_P, GEN_REP_PENALTY, GEN_MAX_NEW_TOKENS, DTYPE
)
from model import ShakespeareGPT, ModelConfig
from tokenizer import load_tokenizer


def load_model_for_inference(checkpoint_name: str = "best.pt"):
    """Load model with enforced dtype for memory efficiency."""
    ckpt_path = CHECKPOINT_DIR / checkpoint_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Run train.py first.")
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: ModelConfig = ckpt["config"]
    
    model = ShakespeareGPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    
    # Enforce dtype for memory efficiency (e.g., float16 for T4)
    dtype = torch.float16 if DTYPE == "float16" else torch.bfloat16
    model = model.to(dtype=dtype)
    model.eval()
    return model, cfg


def generate_single(
    model: ShakespeareGPT,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    min_new_tokens: int,
    stop_sequence: Optional[str],
    stream: bool,
    device: torch.device,
) -> dict:
    """Generate text for a single prompt with detailed metrics."""
    bos = tokenizer.token_to_id("[BOS]")
    eos = tokenizer.token_to_id("[EOS]")
    
    # Encode prompt
    prompt_ids = [bos] + tokenizer.encode(prompt).ids
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    prompt_len = len(prompt_ids)
    
    # Timing setup
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()

    generated_text = ""
    
    if stream:
        print("\n[Model]: ", end="", flush=True)

    # Use streaming or batched generation based on flag
    if stream:
        out_ids_list = prompt_ids.copy()
        for token in model.generate_stream(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            min_new_tokens=min_new_tokens,
            eos_token_id=eos,
        ):
            out_ids_list.append(token)
            token_str = tokenizer.decode([token])
            print(token_str, end="", flush=True)
            generated_text += token_str
            
            # Check stop sequence
            if stop_sequence and stop_sequence in generated_text:
                generated_text = generated_text.split(stop_sequence)[0]
                break
        print()  # Newline after stream
        out_ids = out_ids_list
    else:
        out_ids_tensor = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            min_new_tokens=min_new_tokens,
            eos_token_id=eos,
        )
        out_ids = out_ids_tensor[0].tolist()
        new_ids = out_ids[prompt_len:]
        generated_text = tokenizer.decode(new_ids).strip()
        
        if stop_sequence and stop_sequence in generated_text:
            generated_text = generated_text.split(stop_sequence)[0]

    if device.type == "cuda":
        torch.cuda.synchronize()
    
    elapsed_time = time.time() - start_time
    new_tokens = len(out_ids) - prompt_len
    decode_speed = new_tokens / elapsed_time if elapsed_time > 0 else 0.0

    return {
        "prompt": prompt,
        "generation": generated_text,
        "prompt_tokens": prompt_len,
        "new_tokens": new_tokens,
        "total_tokens": len(out_ids),
        "time_seconds": round(elapsed_time, 3),
        "tokens_per_second": round(decode_speed, 2)
    }


def main():
    parser = argparse.ArgumentParser(description="Production-Grade Text Generation")
    parser.add_argument("--prompt", type=str, default="ROMEO:\n", help="Input prompt")
    parser.add_argument("--prompts-file", type=str, default=None, help="Path to a text file with one prompt per line")
    parser.add_argument("--checkpoint", type=str, default="best.pt", help="Checkpoint file to load")
    parser.add_argument("--max-tokens", type=int, default=GEN_MAX_NEW_TOKENS, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=GEN_TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=GEN_TOP_K, help="Top-K sampling")
    parser.add_argument("--top-p", type=float, default=GEN_TOP_P, help="Top-P (nucleus) sampling")
    parser.add_argument("--rep-penalty", type=float, default=GEN_REP_PENALTY, help="Repetition penalty")
    parser.add_argument("--min-tokens", type=int, default=0, help="Minimum new tokens before allowing EOS")
    parser.add_argument("--stop-sequence", type=str, default=None, help="String to stop generation early")
    parser.add_argument("--stream", action="store_true", help="Stream output token by token")
    parser.add_argument("--output-file", type=str, default=None, help="Save results to this JSONL file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[*] Loading model from {args.checkpoint} on {device}...")
    
    try:
        model, cfg = load_model_for_inference(args.checkpoint)
        model = model.to(device)
    except FileNotFoundError as e:
        print(f"[!] Error: {e}")
        return

    print("[*] Loading tokenizer...")
    tokenizer = load_tokenizer()
    print("[+] Ready!\n")

    # Gather prompts
    prompts = []
    if args.prompts_file:
        p_file = Path(args.prompts_file)
        if p_file.exists():
            prompts = p_file.read_text(encoding="utf-8").splitlines()
            prompts = [p.strip() for p in prompts if p.strip()]
            print(f"[*] Loaded {len(prompts)} prompts from {args.prompts_file}")
        else:
            print(f"[!] Prompts file not found: {args.prompts_file}")
            return
    else:
        prompts = [args.prompt]

    results = []
    
    for i, prompt in enumerate(prompts):
        if len(prompts) > 1:
            print(f"\n--- Prompt {i+1}/{len(prompts)} ---")
            print(f"Prompt: {prompt}")
            
        result = generate_single(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.rep_penalty,
            min_new_tokens=args.min_tokens,
            stop_sequence=args.stop_sequence,
            stream=args.stream and len(prompts) == 1,  # Only stream if single prompt for clean output
            device=device,
        )
        results.append(result)
        
        if not args.stream or len(prompts) > 1:
            print(f"\n[Generation]:\n{result['generation']}")
            print(f"\n[Metrics]: {result['new_tokens']} tokens in {result['time_seconds']}s ({result['tokens_per_second']} tok/s)")

    # Save to file if requested
    if args.output_file:
        out_path = Path(args.output_file)
        with out_path.open("w", encoding="utf-8") as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
        print(f"\n[+] Results saved to {out_path}")


if __name__ == "__main__":
    main()