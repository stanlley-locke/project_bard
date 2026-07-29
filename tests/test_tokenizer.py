"""
tests/test_tokenizer.py
"""
import pytest
from tokenizer import train_bpe_tokenizer, load_tokenizer
from config import VOCAB_SIZE, SPECIAL_TOKENS


def test_tokenizer_train():
    tok = train_bpe_tokenizer()
    assert tok.get_vocab_size() == VOCAB_SIZE


def test_special_tokens():
    tok = load_tokenizer()
    for st in SPECIAL_TOKENS:
        assert tok.token_to_id(st) is not None


def test_encode_decode_roundtrip():
    tok = load_tokenizer()
    text = "To be, or not to be, that is the question."
    ids = tok.encode(text).ids
    decoded = tok.decode(ids)
    # Should preserve the text (modulo BOS/EOS handling)
    assert "To be" in decoded
    assert "question" in decoded


def test_empty_input():
    tok = load_tokenizer()
    ids = tok.encode("").ids
    assert isinstance(ids, list)


def test_unknown_chars():
    tok = load_tokenizer()
    # Random unicode should still encode (byte fallback)
    ids = tok.encode("🎭✨🚀").ids
    assert len(ids) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])