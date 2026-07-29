"""
tokenizer.py - Production-Grade Phase 2: Tokenization
Features:
  - Large vocabulary (32K) for richer subword patterns
  - Byte fallback (eliminates UNK tokens entirely)
  - Unicode NFKC normalization
  - Enhanced pre-tokenization (digits, punctuation, whitespace)
  - BOS/EOS post-processing
  - Comprehensive vocabulary analysis and statistics
  - Vocabulary export for inspection
  - Tokenization coverage metrics
"""
import numpy as np
from pathlib import Path
from typing import Dict
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors, decoders, normalizers

from config import (
    CLEAN_TEXT_PATH, TOKENIZER_DIR, TOKEN_IDS_PATH,
    VOCAB_SIZE, SPECIAL_TOKENS, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN
)


def train_bpe_tokenizer() -> Tokenizer:
    """Train a production-grade BPE tokenizer with byte fallback and large vocabulary."""
    print("=" * 70)
    print("[PHASE 2] Tokenization (Production-Grade)")
    print("=" * 70)

    # Initialize BPE model with byte fallback (critical for robustness)
    tokenizer = Tokenizer(models.BPE(
        unk_token=None,  # No UNK - we use byte fallback instead
        fuse_unk=False,
        byte_fallback=True  # Decomposes unknown chars into UTF-8 bytes
    ))

    # Unicode normalization (NFKC handles full-width chars, accents, etc.)
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.Replace(" ", " "),  # Normalize whitespace
    ])

    # Enhanced pre-tokenization
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=False),  # Keep numbers together
        pre_tokenizers.Punctuation(),  # Separate punctuation
        pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True),
    ])

    # Trainer with larger vocabulary and better defaults
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,  # Minimum 2 occurrences to be included
        limit_alphabet=1000,  # Limit initial character-level tokens
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # Include all bytes
        show_progress=True,
    )

    print(f"[*] Training BPE tokenizer...")
    print(f"    Vocabulary size: {VOCAB_SIZE}")
    print(f"    Byte fallback: Enabled")
    print(f"    Unicode normalization: NFKC")
    print(f"    Training file: {CLEAN_TEXT_PATH}")
    
    tokenizer.train(files=[str(CLEAN_TEXT_PATH)], trainer=trainer)

    # Post-processor: Add BOS/EOS tokens automatically
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        pair=f"{BOS_TOKEN} $A {EOS_TOKEN} {BOS_TOKEN} $B {EOS_TOKEN}",
        special_tokens=[
            (BOS_TOKEN, tokenizer.token_to_id(BOS_TOKEN)),
            (EOS_TOKEN, tokenizer.token_to_id(EOS_TOKEN)),
        ],
    )
    
    # Decoder: Convert tokens back to UTF-8 text
    tokenizer.decoder = decoders.ByteLevel()

    # Save tokenizer
    tokenizer_path = TOKENIZER_DIR / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    
    print(f"[+] Tokenizer saved: {tokenizer_path}")
    print(f"[+] Final vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Run analysis
    analyze_vocabulary(tokenizer)
    
    return tokenizer


def analyze_vocabulary(tokenizer: Tokenizer) -> Dict:
    """Comprehensive analysis of the trained vocabulary."""
    print("\n[*] Analyzing vocabulary...")
    
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    
    # Categorize tokens
    special_count = 0
    single_char_count = 0
    multi_char_count = 0
    byte_tokens = 0
    
    token_lengths = []
    
    for token, idx in vocab.items():
        if token in SPECIAL_TOKENS:
            special_count += 1
        elif len(token) == 1:
            single_char_count += 1
        elif token.startswith("<0x") and token.endswith(">"):
            byte_tokens += 1
        else:
            multi_char_count += 1
            token_lengths.append(len(token))
    
    avg_length = sum(token_lengths) / len(token_lengths) if token_lengths else 0
    max_length = max(token_lengths) if token_lengths else 0
    
    stats = {
        "vocab_size": vocab_size,
        "special_tokens": special_count,
        "single_char_tokens": single_char_count,
        "multi_char_tokens": multi_char_count,
        "byte_tokens": byte_tokens,
        "avg_token_length": avg_length,
        "max_token_length": max_length,
    }
    
    print(f"    Vocabulary breakdown:")
    print(f"      Total tokens: {vocab_size:,}")
    print(f"      Special tokens: {special_count}")
    print(f"      Single character tokens: {single_char_count:,}")
    print(f"      Multi-character tokens: {multi_char_count:,}")
    print(f"      Byte fallback tokens: {byte_tokens}")
    print(f"      Average token length: {avg_length:.2f} chars")
    print(f"      Maximum token length: {max_length} chars")
    
    # Export vocabulary for inspection
    export_vocabulary(tokenizer)
    
    return stats


def export_vocabulary(tokenizer: Tokenizer, max_tokens: int = 1000):
    """Export vocabulary to a human-readable file for inspection."""
    vocab_path = TOKENIZER_DIR / "vocabulary.txt"
    vocab = tokenizer.get_vocab()
    
    # Sort by token ID
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    
    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write("# Project Bard Vocabulary\n")
        f.write(f"# Total size: {len(vocab)}\n")
        f.write("# Format: ID | Token\n")
        f.write("#" + "=" * 60 + "\n\n")
        
        for token, idx in sorted_vocab[:max_tokens]:
            # Escape special characters for readability
            display_token = token.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
            f.write(f"{idx:5d} | {display_token}\n")
        
        if len(vocab) > max_tokens:
            f.write(f"\n# ... and {len(vocab) - max_tokens} more tokens\n")
    
    print(f"[+] Vocabulary exported: {vocab_path} (showing first {max_tokens} tokens)")


def calculate_coverage(tokenizer: Tokenizer, sample_size: int = 100000) -> Dict:
    """Calculate tokenization efficiency metrics."""
    print("\n[*] Calculating tokenization coverage...")
    
    text = CLEAN_TEXT_PATH.read_text(encoding="utf-8")
    
    # Sample for speed
    if len(text) > sample_size:
        sample_text = text[:sample_size]
    else:
        sample_text = text
    
    # Encode
    encoding = tokenizer.encode(sample_text)
    token_ids = encoding.ids
    
    # Calculate metrics
    char_count = len(sample_text)
    token_count = len(token_ids)
    compression_ratio = char_count / token_count if token_count > 0 else 0
    
    # Count unique tokens used
    unique_tokens = len(set(token_ids))
    vocab_utilization = unique_tokens / tokenizer.get_vocab_size() * 100
    
    metrics = {
        "characters": char_count,
        "tokens": token_count,
        "compression_ratio": compression_ratio,
        "unique_tokens_used": unique_tokens,
        "vocab_utilization_percent": vocab_utilization,
    }
    
    print(f"    Coverage metrics (sample of {char_count:,} chars):")
    print(f"      Total tokens: {token_count:,}")
    print(f"      Compression ratio: {compression_ratio:.2f} chars/token")
    print(f"      Unique tokens used: {unique_tokens:,}")
    print(f"      Vocabulary utilization: {vocab_utilization:.1f}%")
    
    return metrics


def digitize_and_save(tokenizer: Tokenizer) -> np.memmap:
    """Convert the cleaned corpus to token IDs. (Threading is handled automatically by the Rust backend)."""
    print("\n[*] Digitizing corpus...")
    
    text = CLEAN_TEXT_PATH.read_text(encoding="utf-8")
    print(f"    Corpus size: {len(text):,} characters")
    
    print("[*] Encoding text to token IDs...")
    encoding = tokenizer.encode(text)
    ids = np.array(encoding.ids, dtype=np.uint16)  # uint16 supports up to 65535 tokens
    
    # Save as raw binary for fast memmap reads
    ids.tofile(str(TOKEN_IDS_PATH))
    print(f"[+] Token IDs saved: {TOKEN_IDS_PATH}")
    print(f"    Total tokens: {ids.shape[0]:,}")
    print(f"    File size: {ids.nbytes / 1024 / 1024:.2f} MB")
    
    # Calculate coverage metrics
    calculate_coverage(tokenizer)
    
    # Return a memory-mapped view (zero-copy, disk-backed)
    mmap = np.memmap(str(TOKEN_IDS_PATH), dtype=np.uint16, mode="r", shape=ids.shape)
    return mmap


def load_tokenizer() -> Tokenizer:
    """Load the trained tokenizer from disk."""
    return Tokenizer.from_file(str(TOKENIZER_DIR / "tokenizer.json"))


def test_tokenizer(tokenizer: Tokenizer):
    """Validate the tokenizer with test cases."""
    print("\n[*] Running tokenizer validation tests...")
    
    test_cases = [
        "To be, or not to be, that is the question.",
        "Hello, world! 123",
        "Special chars: àáâãäå æ ç èéêë",
        "Emojis and symbols: ",
        "Mixed: The price is $1,234.56 for 50% off!",
    ]
    
    for test in test_cases:
        encoding = tokenizer.encode(test)
        decoded = tokenizer.decode(encoding.ids)
        
        # Check if decode matches (allowing for BOS/EOS)
        decoded_clean = decoded.replace(BOS_TOKEN, "").replace(EOS_TOKEN, "").strip()
        
        match = "OK" if decoded_clean == test else "MISMATCH"
        print(f"    [{match}] '{test[:40]}...' -> {len(encoding.ids)} tokens")
        
        if match == "MISMATCH":
            print(f"         Original:  {test}")
            print(f"         Decoded:   {decoded_clean}")
    
    print("[+] Validation complete.")


if __name__ == "__main__":
    # Train tokenizer
    tok = train_bpe_tokenizer()
    
    # Validate
    test_tokenizer(tok)
    
    # Digitize corpus
    digitize_and_save(tok)
    
    print("\n" + "=" * 70)
    print("[+] Tokenization pipeline complete!")
    print("=" * 70)