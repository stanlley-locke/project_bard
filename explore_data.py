"""
explore_data.py - Dataset exploration and validation utility.
Inspects token distributions, sequence lengths, and decodes sample batches.
"""
import numpy as np
import torch
from config import SPLIT_DIR, BLOCK_SIZE, TOKEN_IDS_PATH
from tokenizer import load_tokenizer


def explore_dataset():
    print("=" * 70)
    print(" 🔍 PROJECT BARD: Dataset Explorer 🔍")
    print("=" * 70)

    # 1. Load Tokenizer
    print("[*] Loading tokenizer...")
    tokenizer = load_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    print(f"[+] Vocabulary size: {vocab_size}")

    # 2. Inspect Raw Token File
    print(f"\n[*] Inspecting raw token file: {TOKEN_IDS_PATH}")
    if TOKEN_IDS_PATH.exists():
        raw_data = np.memmap(str(TOKEN_IDS_PATH), dtype=np.uint16, mode="r")
        print(f"    Total tokens: {len(raw_data):,}")
        
        # Show a random snippet
        start_idx = np.random.randint(0, len(raw_data) - 100)
        snippet_ids = raw_data[start_idx : start_idx + 100].astype(np.int64)
        snippet_text = tokenizer.decode(snippet_ids.tolist())
        print(f"\n    Random 100-token snippet:")
        print(f"    ---")
        print(f"    {snippet_text}")
        print(f"    ---")
    else:
        print("    [!] Raw token file not found. Run tokenizer.py first.")

    # 3. Inspect Splits
    print("\n[*] Inspecting data splits:")
    for split in ["train", "val", "test"]:
        split_path = SPLIT_DIR / f"{split}.bin"
        if split_path.exists():
            data = np.memmap(str(split_path), dtype=np.uint16, mode="r")
            num_sequences = len(data) // BLOCK_SIZE
            print(f"    {split.ljust(5)}: {len(data):>8,} tokens | {num_sequences:>5,} sequences of length {BLOCK_SIZE}")
            
            # Decode one random sequence from the split
            seq_idx = np.random.randint(0, num_sequences)
            start = seq_idx * BLOCK_SIZE
            end = start + BLOCK_SIZE
            seq_ids = data[start:end].astype(np.int64)
            seq_text = tokenizer.decode(seq_ids.tolist())
            
            print(f"    Sample {split} sequence (first 150 chars):")
            print(f"    > {seq_text[:150].replace(chr(10), ' ')}...")
        else:
            print(f"    [!] {split}.bin not found. Run dataset.py first.")

    # 4. Token Frequency Analysis (Optional but useful)
    print("\n[*] Running quick token frequency analysis on validation set...")
    val_path = SPLIT_DIR / "val.bin"
    if val_path.exists():
        val_data = np.memmap(str(val_path), dtype=np.uint16, mode="r")
        # Sample 100k tokens for speed
        sample_size = min(100000, len(val_data))
        sample = val_data[:sample_size]
        
        unique_tokens = np.unique(sample)
        print(f"    Unique tokens in sample: {len(unique_tokens)} / {vocab_size}")
        
        # Find most common tokens
        counts = np.bincount(sample, minlength=vocab_size)
        top_5_indices = np.argsort(counts)[-5:][::-1]
        print("    Top 5 most common tokens:")
        for idx in top_5_indices:
            token_str = tokenizer.decode([idx]).replace('\n', '\\n').replace('\r', '\\r')
            print(f"      - ID {idx:4d}: '{token_str}' (count: {counts[idx]:,})")

    print("\n[+] Exploration complete!")


if __name__ == "__main__":
    explore_dataset()