import pytest
from tokenizer import train_bpe_tokenizer, load_tokenizer
from config import VOCAB_SIZE, SPECIAL_TOKENS, BOS_TOKEN, EOS_TOKEN

def test_tokenizer_train():
    tok = train_bpe_tokenizer()
    assert tok.get_vocab_size() == VOCAB_SIZE

def test_special_tokens():
    tok = load_tokenizer()
    for st in SPECIAL_TOKENS:
        assert tok.token_to_id(st) is not None

def test_special_tokens_have_ids():
    tok = load_tokenizer()
    for st in SPECIAL_TOKENS:
        assert tok.token_to_id(st) is not None

def test_encode_decode_roundtrip():
    tok = load_tokenizer()
    text = "To be, or not to be, that is the question."
    ids = tok.encode(text).ids
    decoded = tok.decode(ids)
    assert "To be" in decoded
    assert "question" in decoded

def test_empty_input():
    tok = load_tokenizer()
    ids = tok.encode("").ids
    assert isinstance(ids, list)

def test_empty_input_handling():
    tok = load_tokenizer()
    ids = tok.encode("").ids
    assert isinstance(ids, list)

def test_unknown_chars():
    tok = load_tokenizer()
    ids = tok.encode("🎭✨🚀").ids
    assert len(ids) > 0

def test_unicode_byte_fallback():
    tok = load_tokenizer()
    text = "🎭✨🚀"
    ids = tok.encode(text).ids
    assert len(ids) > 0

def test_bos_eos_in_encoding():
    tok = load_tokenizer()
    text = "Hello world"
    encoding = tok.encode(text)
    ids = encoding.ids
    
    bos_id = tok.token_to_id(BOS_TOKEN)
    eos_id = tok.token_to_id(EOS_TOKEN)
    
    assert ids[0] == bos_id
    assert ids[-1] == eos_id

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
