"""
dataset.py - Production-Grade Phase 3: Dataset Splitting & Structuring
Features:
  - Configurable stride/overlap for sliding window context (better learning)
  - Data integrity validation (checks for out-of-bounds tokens)
  - Detailed split statistics and verification
  - Optimized memory-mapped reading with efficient PyTorch tensor conversion
  - Robust DataLoader with persistent workers for faster epoch transitions
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import (
    TOKEN_IDS_PATH, SPLIT_DIR, BLOCK_SIZE,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, BATCH_SIZE, SEED, NUM_WORKERS, VOCAB_SIZE
)


def split_data():
    """Split the token ID binary file into train/val/test memmaps with verification."""
    print("=" * 70)
    print("[PHASE 3] Dataset Splitting & Structuring")
    print("=" * 70)

    if not TOKEN_IDS_PATH.exists():
        raise FileNotFoundError(f"Tokenized data not found at {TOKEN_IDS_PATH}. Run tokenizer.py first.")

    # Load as read-only memmap
    ids = np.memmap(str(TOKEN_IDS_PATH), dtype=np.uint16, mode="r")
    n = len(ids)
    
    if n == 0:
        raise ValueError("Tokenized data is empty.")

    # Verify ratios sum to ~1.0
    total_ratio = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if not (0.99 <= total_ratio <= 1.01):
        raise ValueError(f"Split ratios must sum to 1.0. Current sum: {total_ratio}")

    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    # Test gets the remainder to ensure no tokens are lost
    
    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]

    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids
    }

    print(f"[*] Total tokens: {n:,}")
    for name, arr in splits.items():
        path = SPLIT_DIR / f"{name}.bin"
        # Create writable memmap, copy data, and flush to disk
        mm = np.memmap(str(path), dtype=np.uint16, mode="w+", shape=arr.shape)
        mm[:] = arr[:]
        mm.flush()
        
        ratio = len(arr) / n * 100
        print(f"[+] {name.ljust(5)}: {len(arr):>10,} tokens ({ratio:>5.1f}%) -> {path}")

    print("[+] Dataset splitting complete.")


class LMDataset(Dataset):
    """
    Production-grade Language Model Dataset.
    Streams chunks from a memmap file with configurable stride (overlap).
    x = tokens[0:T], y = tokens[1:T+1] (next-token prediction).
    """

    def __init__(self, split: str, block_size: int = BLOCK_SIZE, stride: int = None):
        assert split in ("train", "val", "test"), f"Invalid split: {split}"
        self.split = split
        self.path = SPLIT_DIR / f"{split}.bin"
        
        if not self.path.exists():
            raise FileNotFoundError(f"Split file not found: {self.path}")
            
        self.data = np.memmap(str(self.path), dtype=np.uint16, mode="r")
        self.block_size = block_size
        
        # Default stride to block_size (non-overlapping) if not specified.
        # For training, stride = block_size // 2 is recommended for better context learning.
        self.stride = stride if stride is not None else block_size
        
        if self.stride <= 0 or self.stride > self.block_size:
            raise ValueError("Stride must be > 0 and <= block_size")

        # Calculate number of valid chunks
        # We need block_size + 1 tokens to generate x (block_size) and y (block_size)
        self.num_chunks = max(1, (len(self.data) - self.block_size) // self.stride)

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        end = start + self.block_size + 1
        
        # Slice from memmap and convert to int64 for PyTorch cross_entropy compatibility
        chunk = self.data[start:end].astype(np.int64)
        
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        
        return x, y

    def validate_integrity(self, max_checks: int = 1000) -> dict:
        """Randomly sample chunks to verify data integrity (no out-of-bounds tokens)."""
        print(f"[*] Validating {self.split} dataset integrity...")
        errors = 0
        max_token = 0
        
        # Check random indices
        indices = np.random.randint(0, len(self), size=min(max_checks, len(self)))
        for idx in indices:
            start = idx * self.stride
            end = start + self.block_size + 1
            chunk = self.data[start:end]
            
            current_max = np.max(chunk)
            max_token = max(max_token, current_max)
            
            if current_max >= VOCAB_SIZE:
                errors += 1
                
        if errors > 0:
            print(f"[!] WARNING: Found {errors} chunks with out-of-bounds tokens (>= {VOCAB_SIZE})")
            print(f"    Max token ID found: {max_token}")
        else:
            print(f"[+] {self.split} dataset integrity verified. Max token ID: {max_token}")
            
        return {"errors": errors, "max_token": max_token}


def get_dataloader(
    split: str, 
    block_size: int = BLOCK_SIZE, 
    batch_size: int = BATCH_SIZE, 
    stride: int = None,
    shuffle: bool = True, 
    num_workers: int = NUM_WORKERS
) -> DataLoader:
    """
    Create a DataLoader for the specified split.
    """
    ds = LMDataset(split, block_size=block_size, stride=stride)
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0), # Keeps workers alive between epochs
    )


if __name__ == "__main__":
    # 1. Split the data
    split_data()
    
    # 2. Initialize dataloaders
    print("\n[*] Initializing DataLoaders...")
    # Use stride=BLOCK_SIZE//2 for training to maximize context exposure
    train_dl = get_dataloader("train", batch_size=4, stride=BLOCK_SIZE // 2)
    val_dl = get_dataloader("val", batch_size=4, stride=BLOCK_SIZE)
    test_dl = get_dataloader("test", batch_size=4, stride=BLOCK_SIZE)
    
    # 3. Sanity check
    x, y = next(iter(train_dl))
    print(f"\n[+] Sanity check passed:")
    print(f"    Batch shape: x={tuple(x.shape)}, y={tuple(y.shape)}")
    print(f"    x dtype: {x.dtype}, y dtype: {y.dtype}")
    print(f"    x sample: {x[0, :10].tolist()}")
    print(f"    y sample: {y[0, :10].tolist()}")
    
    # 4. Validate integrity
    print("\n[*] Running integrity checks...")
    for split in ["train", "val", "test"]:
        ds = LMDataset(split, block_size=BLOCK_SIZE)
        ds.validate_integrity()