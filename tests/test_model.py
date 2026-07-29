import pytest
import torch
from model import ShakespeareGPT, ModelConfig, count_parameters, RMSNorm

@pytest.fixture
def config():
    cfg = ModelConfig()
    cfg.n_layer = 2
    cfg.n_head = 2
    cfg.n_embd = 32
    cfg.mlp_hidden = 64
    cfg.vocab_size = 100
    cfg.block_size = 32
    return cfg

@pytest.fixture
def model(config):
    return ShakespeareGPT(config)

def test_model_forward(model):
    x = torch.randint(0, model.cfg.vocab_size, (2, 10))
    y = torch.randint(0, model.cfg.vocab_size, (2, 10))
    out = model(x, y)
    
    logits = out['logits']
    loss = out['loss']
    
    assert logits.shape == (2, 10, model.cfg.vocab_size)
    assert loss is not None
    assert loss > 0

def test_model_deterministic(model):
    x = torch.randint(0, model.cfg.vocab_size, (2, 10))
    model.eval()
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1['logits'], out2['logits'])

def test_kv_cache_equivalence(model):
    x = torch.randint(0, model.cfg.vocab_size, (1, 10))
    model.eval()
    with torch.no_grad():
        out_full = model(x)
        logits_full = out_full['logits']
        
        # Step by step with KV cache
        out_part1 = model(x[:, :-1], use_kv_cache=True)
        past_kvs = out_part1['new_kvs']
        
        out_part2 = model(x[:, -1:], past_kvs=past_kvs, use_kv_cache=True)
        logits_cached = out_part2['logits']
        
    # The last token logits should match
    assert torch.allclose(logits_full[:, -1, :], logits_cached[:, -1, :], atol=1e-4)

def test_gradient_flow(model):
    x = torch.randint(0, model.cfg.vocab_size, (2, 10))
    y = torch.randint(0, model.cfg.vocab_size, (2, 10))
    out = model(x, y)
    loss = out['loss']
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"

def test_parameter_count(model):
    params = count_parameters(model)
    assert params > 0
    assert params < 10_000_000  # for this small config

def test_rmsnorm():
    norm = RMSNorm(32)
    x = torch.randn(2, 10, 32)
    y = norm(x)
    assert y.shape == x.shape

def test_generation(model):
    x = torch.randint(0, model.cfg.vocab_size, (1, 5))
    model.eval()
    max_new_tokens = 10
    out = model.generate(x, max_new_tokens=max_new_tokens)
    assert out.shape == (1, 5 + max_new_tokens)

def test_sequence_too_long(model):
    x = torch.randint(0, model.cfg.vocab_size, (1, model.cfg.block_size + 1))
    with pytest.raises(AssertionError):
        model(x)

def test_hidden_states_output(model):
    x = torch.randint(0, model.cfg.vocab_size, (2, 10))
    out = model(x, output_hidden_states=True)
    hidden_states = out['hidden_states']
    assert hidden_states is not None
    assert len(hidden_states) == model.cfg.n_layer + 1  # input + layers

def test_attention_weights_output(model):
    x = torch.randint(0, model.cfg.vocab_size, (2, 10))
    model.cfg.use_flash_attn = False
    for block in model.blocks:
        block.attn.use_flash = False
        
    out = model(x, output_attentions=True)
    attentions = out['attentions']
    assert attentions is not None
    assert len(attentions) == model.cfg.n_layer
    for att in attentions:
        assert att.shape == (2, model.cfg.n_head, 10, 10)

def test_weight_tying(model):
    assert torch.equal(model.token_emb.weight, model.lm_head.weight)

def test_generate_stream(model):
    x = torch.randint(0, model.cfg.vocab_size, (1, 5))
    model.eval()
    gen = model.generate_stream(x, max_new_tokens=5)
    tokens = list(gen)
    assert len(tokens) == 5
    for t in tokens:
        assert isinstance(t, int)
        assert 0 <= t < model.cfg.vocab_size

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
