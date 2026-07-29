"""
explore_data.py - Dataset exploration and validation utility.
Inspects token distributions, sequence lengths, and decodes sample batches.
"""
import argparse
import sys
import numpy as np
import torch
from config import SPLIT_DIR, BLOCK_SIZE, TOKEN_IDS_PATH
from tokenizer import load_tokenizer


def explore_dataset(target_split=None, num_samples=1):
    print("=" * 70)
    print(" 🔍 PROJECT BARD: Dataset Explorer 🔍")
    print("=" * 70)

    # 1. Load Tokenizer
    print("[*] Loading tokenizer...")
    try:
        tokenizer = load_tokenizer()
        vocab_size = tokenizer.get_vocab_size()
        print(f"[+] Vocabulary size: {vocab_size}")
    except Exception as e:
        print(f"    [!] Failed to load tokenizer: {e}")
        sys.exit(1)

    # 2. Inspect Raw Token File
    if not target_split:
        print(f"\n[*] Inspecting raw token file: {TOKEN_IDS_PATH}")
        if TOKEN_IDS_PATH.exists():
            try:
                raw_data = np.memmap(str(TOKEN_IDS_PATH), dtype=np.uint16, mode="r")
                print(f"    Total tokens: {len(raw_data):,}")
                
                # Show random snippets
                for i in range(num_samples):
                    if len(raw_data) > 100:
                        start_idx = np.random.randint(0, len(raw_data) - 100)
                        snippet_ids = raw_data[start_idx : start_idx + 100].astype(np.int64)
                        snippet_text = tokenizer.decode(snippet_ids.tolist())
                        print(f"\n    Random 100-token snippet #{i+1}:")
                        print(f"    ---")
                        print(f"    {snippet_text}")
                        print(f"    ---")
            except Exception as e:
                print(f"    [!] Error reading raw tokens: {e}")
        else:
            print("    [!] Raw token file not found. Run tokenizer.py first.")

    # 3. Inspect Splits
    print("\n[*] Inspecting data splits:")
    splits_to_check = [target_split] if target_split else ["train", "val", "test"]
    
    for split in splits_to_check:
        split_path = SPLIT_DIR / f"{split}.bin"
        if split_path.exists():
            try:
                data = np.memmap(str(split_path), dtype=np.uint16, mode="r")
                num_sequences = len(data) // BLOCK_SIZE
                print(f"    {split.ljust(5)}: {len(data):>8,} tokens | {num_sequences:>5,} sequences of length {BLOCK_SIZE}")
                
                for i in range(num_samples):
                    if num_sequences > 0:
                        seq_idx = np.random.randint(0, num_sequences)
                        start = seq_idx * BLOCK_SIZE
                        end = start + BLOCK_SIZE
                        seq_ids = data[start:end].astype(np.int64)
                        seq_text = tokenizer.decode(seq_ids.tolist())
                        
                        print(f"    Sample {split} sequence #{i+1} (first 150 chars):")
                        print(f"    > {seq_text[:150].replace(chr(10), ' ')}...")
            except Exception as e:
                print(f"    [!] Error reading {split}.bin: {e}")
        else:
            print(f"    [!] {split}.bin not found. Run dataset.py first.")

    # 4. Token Frequency Analysis
    analysis_split = target_split if target_split else "val"
    print(f"\n[*] Running token frequency analysis on {analysis_split} set...")
    val_path = SPLIT_DIR / f"{analysis_split}.bin"
    if val_path.exists():
        try:
            val_data = np.memmap(str(val_path), dtype=np.uint16, mode="r")
            # Sample 100k tokens for speed
            sample_size = min(100000, len(val_data))
            sample = val_data[:sample_size]
            
            unique_tokens = np.unique(sample)
            print(f"    Unique tokens in sample: {len(unique_tokens)} / {vocab_size}")
            
            # Find most common tokens
            counts = np.bincount(sample, minlength=vocab_size)
            top_indices = np.argsort(counts)[-10:][::-1]
            print("    Top 10 most common tokens:")
            
            max_count = counts[top_indices[0]] if len(top_indices) > 0 else 1
            
            for idx in top_indices:
                if counts[idx] == 0:
                    continue
                token_str = tokenizer.decode([idx]).replace('\n', '\\n').replace('\r', '\\r')
                count = counts[idx]
                bar_len = int(20 * count / max_count)
                bar = '█' * bar_len
                print(f"      - ID {idx:4d} | {bar:<20} | count: {count:>6,} | '{token_str}'")
        except Exception as e:
            print(f"    [!] Error during token analysis: {e}")
    else:
        print(f"    [!] {analysis_split}.bin not found for frequency analysis.")

    print("\n[+] Exploration complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore and validate dataset splits.")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], help="Specific split to explore")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of random samples to show")
    
    args = parser.parse_args()
    
    explore_dataset(target_split=args.split, num_samples=args.num_samples)