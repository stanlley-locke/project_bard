"""
tests/test_model.py - Unit tests for the model
"""
import pytest
import torch
from model import ShakespeareGPT, ModelConfig, RMSNorm, apply_rope, count_parameters
from config import VOCAB_SIZE


def test_model_forward():
    cfg = ModelConfig(vocab_size=100, n_layer=2, n_head=2, n_embd=64, block_size=32)
    model = ShakespeareGPT(cfg)
    x = torch.randint(0, 100, (2, 16))
    y = torch.randint(0, 100, (2, 16))
    logits, loss, _ = model(x, y)
    assert logits.shape == (2, 16, 100)
    assert loss.item() > 0


def test_model_deterministic():
    torch.manual_seed(42)
    cfg = ModelConfig(vocab_size=100, n_layer=2, n_head=2, n_embd=64, block_size=32)
    model = ShakespeareGPT(cfg).eval()
    x = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        out1, _, _ = model(x)
        out2, _, _ = model(x)
    assert torch.allclose(out1, out2)


def test_kv_cache_equivalence():
    """KV cache generation should match non-cached generation"""
    torch.manual_seed(42)
    cfg = ModelConfig(vocab_size=100, n_layer=2, n_head=2, n_embd=64, block_size=32)
    model = ShakespeareGPT(cfg).eval()
    
    x = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        # Without KV cache
        full_logits, _, _ = model(x)
        
        # With KV cache (prefill + incremental)
        logits_prefill, _, past_kvs = model(x, use_kv_cache=True)
        assert torch.allclose(full_logits, logits_prefill, atol=1e-4)


def test_gradient_flow():
    cfg = ModelConfig(vocab_size=100, n_layer=2, n_head=2, n_embd=64, block_size=32)
    model = ShakespeareGPT(cfg)
    x = torch.randint(0, 100, (2, 8))
    y = torch.randint(0, 100, (2, 8))
    _, loss, _ = model(x, y)
    loss.backward()
    
    # Check all parameters got gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


def test_parameter_count():
    cfg = ModelConfig(vocab_size=1000, n_layer=4, n_head=4, n_embd=128, block_size=64)
    model = ShakespeareGPT(cfg)
    n_params = count_parameters(model)
    assert n_params > 1_000_000  # Should be in the millions
    print(f"Parameters: {n_params:,}")


def test_rmsnorm():
    norm = RMSNorm(64)
    x = torch.randn(2, 10, 64)
    y = norm(x)
    assert y.shape == x.shape
    # RMS should be close to 1 after normalization (times weight)
    rms = torch.sqrt(y.pow(2).mean(dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)


def test_generation():
    cfg = ModelConfig(vocab_size=100, n_layer=2, n_head=2, n_embd=64, block_size=32)
    model = ShakespeareGPT(cfg).eval()
    idx = torch.randint(0, 100, (1, 4))
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=10, temperature=1.0)
    assert out.shape == (1, 14)


def test_sequence_too_long():
    cfg = ModelConfig(vocab_size=100, n_layer=2, n_head=2, n_embd=64, block_size=16)
    model = ShakespeareGPT(cfg)
    x = torch.randint(0, 100, (1, 32))  # Exceeds block_size
    with pytest.raises(AssertionError):
        model(x)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])