"""
tests/test_data.py
"""
import pytest
from data_pipeline import heuristic_filter, exact_dedup, fuzzy_dedup, scrub_pii


def test_heuristic_filter_removes_short():
    text = "Hi\nThis is a real line of text that is long enough."
    out = heuristic_filter(text)
    assert "Hi" not in out
    assert "This is a real line" in out


def test_heuristic_filter_removes_symbols():
    text = "===\n---\nReal text here."
    out = heuristic_filter(text)
    assert "===" not in out
    assert "Real text" in out


def test_exact_dedup():
    paras = ["hello", "world", "hello", "foo"]
    out = exact_dedup(paras)
    assert out == ["hello", "world", "foo"]


def test_fuzzy_dedup():
    paras = [
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox jumps over the lazy dog",  # exact dup
        "the quick brown fox leaps over the lazy dog",  # near dup
        "completely different sentence here",
    ]
    out = fuzzy_dedup(paras, threshold=0.7)
    assert len(out) <= 3  # Should remove at least one duplicate


def test_scrub_pii():
    text = "Email me at test@example.com or call 555-123-4567"
    out = scrub_pii(text)
    assert "test@example.com" not in out
    assert "555-123-4567" not in out
    assert "[EMAIL]" in out
    assert "[PHONE]" in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])