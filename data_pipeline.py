"""
data_pipeline.py - Production-Grade Multi-Source Data Collection and Curation
Features:
  - Expanded data sources (diverse classic literature and reliable raw text)
  - Parallel downloads with retry logic and incremental resumption
  - Advanced quality filtering (preserves paragraph structure)
  - Enhanced PII scrubbing (emails, phones, financial data, IPs, dates)
  - Document-level and paragraph-level deduplication
  - Metadata tracking and comprehensive data quality reporting
"""
import re
import hashlib
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Iterator, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import time

from datasketch import MinHash, MinHashLSH

from config import (
    DATA_SOURCES, RAW_DIR, CLEAN_TEXT_PATH, RAW_TEXT_PATH, DATA_DIR
)

# Expanded data sources for a larger, richer token base
EXPANDED_DATA_SOURCES = [
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
    # Reliable raw text sources
    "https://en.wikipedia.org/w/index.php?title=William_Shakespeare&action=raw", # Wikipedia raw text
]

# Metadata tracking
METADATA_PATH = DATA_DIR / "pipeline_metadata.json"

# Regex patterns
GUTENBERG_HEADER_RE = re.compile(
    r"\*\*\*START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*(.+?)\*\*\*END OF",
    re.DOTALL | re.IGNORECASE,
)
WIKI_HEADER_RE = re.compile(
    r"<mediawiki.*?>.*?</mediawiki>",
    re.DOTALL | re.IGNORECASE,
)


def download_single_source(url: str, index: int, max_retries: int = 3) -> Tuple[int, str, bool]:
    """Download a single source with retry logic and incremental resumption."""
    filename = RAW_DIR / f"source_{index}.txt"
    
    # Incremental processing: skip if already downloaded
    if filename.exists() and filename.stat().st_size > 0:
        return index, str(filename), True
    
    for attempt in range(max_retries):
        try:
            print(f"[*] Downloading [{index}] {url} (attempt {attempt + 1}/{max_retries})")
            
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'ProjectBard/1.0 (Educational ML Project)'}
            )
            
            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read()
                
            # Try UTF-8 first, fall back to latin-1
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                text = content.decode('latin-1', errors='ignore')
            
            # Normalize line endings
            text = text.replace('\r\n', '\n').replace('\r', '\n')
            
            filename.write_text(text, encoding="utf-8")
            print(f"[+] Downloaded [{index}] {len(text):,} chars")
            return index, str(filename), True
            
        except Exception as e:
            print(f"[!] Failed [{index}] {url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
    
    return index, url, False


def download_all_sources_parallel() -> Dict:
    """Download all sources in parallel with progress tracking."""
    print("=" * 70)
    print("[PHASE 1.1] Parallel Data Download")
    print("=" * 70)
    
    stats = {
        "total_sources": len(EXPANDED_DATA_SOURCES),
        "successful": 0,
        "failed": 0,
        "total_chars": 0,
        "sources": []
    }
    
    # Use thread pool for parallel downloads
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(download_single_source, url, i): (i, url)
            for i, url in enumerate(EXPANDED_DATA_SOURCES)
        }
        
        for future in as_completed(futures):
            idx, filename, success = future.result()
            
            if success:
                stats["successful"] += 1
                filepath = Path(filename)
                if filepath.exists():
                    text = filepath.read_text(encoding="utf-8")
                    stats["total_chars"] += len(text)
                    stats["sources"].append({
                        "index": idx,
                        "url": EXPANDED_DATA_SOURCES[idx],
                        "file": filename,
                        "chars": len(text)
                    })
            else:
                stats["failed"] += 1
    
    # Combine all downloaded files
    all_text = []
    for source in stats["sources"]:
        filepath = Path(source["file"])
        if filepath.exists():
            all_text.append(filepath.read_text(encoding="utf-8"))
    
    combined = "\n\n".join(all_text)
    RAW_TEXT_PATH.write_text(combined, encoding="utf-8")
    
    print(f"\n[+] Download complete:")
    print(f"    Successful: {stats['successful']}/{stats['total_sources']}")
    print(f"    Failed: {stats['failed']}")
    print(f"    Total characters: {stats['total_chars']:,}")
    print(f"    Combined file: {RAW_TEXT_PATH} ({len(combined):,} chars)")
    
    return stats


# ============================================================================
# HEURISTIC FILTERING
# ============================================================================

def heuristic_filter(text: str) -> str:
    """
    Advanced heuristic filtering with multiple rules.
    CRITICAL FIX: Preserves paragraph structure (\n\n) instead of collapsing everything.
    """
    # Remove Gutenberg headers/footers
    text = GUTENBERG_HEADER_RE.sub("", text)
    text = re.sub(r"^\*{3,}.*\*{3,}$", "", text, flags=re.MULTILINE)
    
    # Remove Wikipedia XML tags and HTML
    text = WIKI_HEADER_RE.sub("", text)
    text = re.sub(r"<[^>]+>", "", text)
    
    paragraphs = []
    # CRITICAL FIX: Split by double newline to preserve paragraph structure
    for raw_para in text.split("\n\n"):
        lines = []
        for raw in raw_para.splitlines():
            line = raw.strip()
            
            if not line:
                continue
            
            # Skip very short lines
            if len(line) < 10:
                continue
            
            # Calculate character quality metrics
            alpha = sum(c.isalpha() for c in line)
            space = sum(c.isspace() for c in line)
            digit = sum(c.isdigit() for c in line)
            special = len(line) - alpha - space - digit
            
            # Skip lines with too many non-alphabetic characters
            if alpha / len(line) < 0.5:
                continue
            
            # Skip lines that are mostly digits (tables, code)
            if digit / len(line) > 0.3:
                continue
            
            # Skip lines with too many special characters
            if special / len(line) > 0.2:
                continue
            
            # Skip lines that look like URLs or file paths
            if re.match(r'^https?://', line) or re.match(r'^[/\\]', line):
                continue
            
            # Skip lines that look like code comments
            if line.startswith('//') or line.startswith('#') or line.startswith('/*'):
                continue
            
            lines.append(line)
        
        if lines:
            # Join valid lines within a paragraph, preserving internal structure
            paragraphs.append("\n".join(lines))
    
    return "\n\n".join(paragraphs)


# ============================================================================
# DEDUPLICATION
# ============================================================================

def exact_dedup_paragraphs(paragraphs: List[str]) -> List[str]:
    """Remove exact duplicate paragraphs using SHA-1 hashing."""
    seen = set()
    out = []
    for p in paragraphs:
        h = hashlib.sha1(p.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(p)
    return out


def exact_dedup_documents(texts: List[str]) -> List[str]:
    """Remove exact duplicate documents."""
    seen = set()
    out = []
    for text in texts:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(text)
    return out


def _shingles(text: str, k: int = 3) -> Iterator[str]:
    """Generate character n-grams for MinHash."""
    text = text.lower()
    for i in range(len(text) - k + 1):
        yield text[i : i + k]


def _minhash(text: str, num_perm: int = 128):
    """Create MinHash signature for text."""
    m = MinHash(num_perm=num_perm)
    for shingle in _shingles(text, 3):
        m.update(shingle.encode("utf-8"))
    return m


def fuzzy_dedup(paragraphs: List[str], threshold: float = 0.8) -> List[str]:
    """Remove near-duplicate paragraphs using MinHash LSH."""
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    kept = []
    
    for idx, p in enumerate(paragraphs):
        if len(p) < 50:  # Skip very short paragraphs
            kept.append(p)
            continue
            
        mh = _minhash(p)
        key = f"p{idx}"
        
        if not lsh.query(mh):
            lsh.insert(key, mh)
            kept.append(p)
    
    return kept


# ============================================================================
# PII SCRUBBING
# ============================================================================

def scrub_pii(text: str) -> str:
    """Comprehensive PII scrubbing with multiple patterns."""
    # Email addresses
    text = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", "[EMAIL]", text)
    
    # Phone numbers (various formats)
    text = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]", text)
    text = re.sub(r"\(\d{3}\)\s?\d{3}[-.\s]?\d{4}", "[PHONE]", text)
    
    # Credit card numbers (basic pattern)
    text = re.sub(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "[CREDIT_CARD]", text)
    
    # SSN (US Social Security Number)
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", text)
    
    # IP addresses
    text = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "[IP_ADDRESS]", text)
    
    # Dates (various formats)
    text = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "[DATE]", text)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "[DATE]", text)
    
    return text


# ============================================================================
# QUALITY FILTERING
# ============================================================================

def quality_filter(paragraphs: List[str]) -> List[str]:
    """Filter paragraphs based on quality metrics."""
    kept = []
    
    for p in paragraphs:
        # Skip very short paragraphs
        if len(p) < 100:
            continue
        
        # Skip very long paragraphs (likely malformed)
        if len(p) > 10000:
            continue
        
        # Calculate word count
        words = p.split()
        if len(words) < 20:
            continue
        
        # Check for excessive repetition
        unique_words = set(words)
        if len(unique_words) / len(words) < 0.3:
            continue  # Too repetitive
        
        # Check for proper sentence structure
        sentences = re.split(r'[.!?]+', p)
        if len(sentences) < 2:
            continue  # Likely not proper prose
        
        kept.append(p)
    
    return kept


# ============================================================================
# STATISTICS AND REPORTING
# ============================================================================

def calculate_statistics(text: str) -> Dict:
    """Calculate comprehensive statistics about the text."""
    words = text.split()
    chars = len(text)
    lines = text.split('\n')
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    
    # Word frequency
    word_freq = Counter(words)
    unique_words = len(word_freq)
    
    # Average metrics
    avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
    avg_words_per_paragraph = len(words) / len(paragraphs) if paragraphs else 0
    
    return {
        "total_characters": chars,
        "total_words": len(words),
        "total_lines": len(lines),
        "total_paragraphs": len(paragraphs),
        "unique_words": unique_words,
        "avg_word_length": avg_word_length,
        "avg_words_per_paragraph": avg_words_per_paragraph,
        "vocabulary_richness": unique_words / len(words) if words else 0,
    }


def save_metadata(stats: Dict):
    """Save pipeline metadata for tracking and resumption."""
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "statistics": stats,
    }
    
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[+] Metadata saved: {METADATA_PATH}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_data_pipeline() -> Path:
    """Execute the complete data pipeline with comprehensive tracking."""
    print("=" * 70)
    print("[PHASE 1] Production-Grade Data Pipeline")
    print("=" * 70)
    
    # Step 1: Download sources in parallel
    download_stats = download_all_sources_parallel()
    
    # Step 2: Read raw text
    raw = RAW_TEXT_PATH.read_text(encoding="utf-8")
    raw_stats = calculate_statistics(raw)
    
    print(f"\n[*] Raw text statistics:")
    print(f"    Characters: {raw_stats['total_characters']:,}")
    print(f"    Words: {raw_stats['total_words']:,}")
    print(f"    Paragraphs: {raw_stats['total_paragraphs']:,}")
    
    # Step 3: Heuristic filtering
    print("\n[*] Applying heuristic filters...")
    filtered = heuristic_filter(raw)
    filtered_stats = calculate_statistics(filtered)
    
    print(f"    After filtering: {filtered_stats['total_characters']:,} chars ({filtered_stats['total_characters']/raw_stats['total_characters']*100:.1f}% retained)")
    print(f"    Paragraphs retained: {filtered_stats['total_paragraphs']:,}")
    
    # Step 4: Split into paragraphs
    paragraphs = [p.strip() for p in filtered.split("\n\n") if p.strip()]
    
    # Step 5: Quality filtering
    print("\n[*] Applying quality filters...")
    paragraphs = quality_filter(paragraphs)
    print(f"    After quality filter: {len(paragraphs):,} paragraphs")
    
    # Step 6: Exact deduplication (paragraph level)
    print("\n[*] Removing exact duplicates...")
    paragraphs = exact_dedup_paragraphs(paragraphs)
    print(f"    After exact dedup: {len(paragraphs):,} paragraphs")
    
    # Step 7: Fuzzy deduplication
    print("\n[*] Removing near-duplicates (MinHash LSH)...")
    paragraphs = fuzzy_dedup(paragraphs, threshold=0.8)
    print(f"    After fuzzy dedup: {len(paragraphs):,} paragraphs")
    
    # Step 8: PII scrubbing
    print("\n[*] Scrubbing PII...")
    paragraphs = [scrub_pii(p) for p in paragraphs]
    
    # Step 9: Rejoin and save
    clean_text = "\n\n".join(paragraphs)
    CLEAN_TEXT_PATH.write_text(clean_text, encoding="utf-8")
    
    final_stats = calculate_statistics(clean_text)
    
    print(f"\n[+] Clean text saved: {CLEAN_TEXT_PATH}")
    print(f"    Final characters: {final_stats['total_characters']:,}")
    print(f"    Final words: {final_stats['total_words']:,}")
    print(f"    Final paragraphs: {final_stats['total_paragraphs']:,}")
    print(f"    Unique words: {final_stats['unique_words']:,}")
    print(f"    Vocabulary richness: {final_stats['vocabulary_richness']:.3f}")
    
    # Step 10: Save metadata
    all_stats = {
        "download": download_stats,
        "raw": raw_stats,
        "filtered": filtered_stats,
        "final": final_stats,
        "retention_rate": final_stats['total_characters'] / raw_stats['total_characters'] if raw_stats['total_characters'] > 0 else 0,
    }
    
    save_metadata(all_stats)
    
    print("\n" + "=" * 70)
    print("[+] Data pipeline complete!")
    print("=" * 70)
    
    return CLEAN_TEXT_PATH


if __name__ == "__main__":
    run_data_pipeline()