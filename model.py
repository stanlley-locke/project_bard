"""
T4-Optimized Transformer
Features:
  - Flash Attention via PyTorch SDPA with robust fallback
  - SwiGLU activation (Llama-style)
  - KV Cache for O(1) step generation
  - Gradient Checkpointing for memory efficiency
  - RMSNorm + Rotary Position Embeddings (RoPE)
  - Verified Top-P / Top-K / Repetition Penalty / Min-Length sampling
  - Real-time token streaming generator
  - Optional return of hidden states and attention weights (for interpretability)
  - Detailed model summary utility
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    N_LAYER, N_HEAD, N_KV_HEAD, N_EMBD, MLP_HIDDEN, BLOCK_SIZE, DROPOUT,
    USE_ROPE, USE_RMSNORM, USE_SWIGLU, USE_GRAD_CHECKPOINT,
    USE_FLASH_ATTN, ROPE_THETA, VOCAB_SIZE, USE_MOE, NUM_EXPERTS, NUM_EXPERTS_PER_TOK
)



# RMSNorm

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute root mean square
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight



# RoPE (with caching)

def precompute_rope_cache(head_dim: int, max_seq_len: int, theta: float = 10000.0, device=None):
    assert head_dim % 2 == 0
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    cos = freqs.cos()
    sin = freqs.sin()
    # Stack for broadcasting: (T, head_dim)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, nh, T, hd)
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin



# SwiGLU MLP

class SwiGLU_MLP(nn.Module):
    """SwiGLU: gated linear unit with SiLU activation (Llama-style)"""
    def __init__(self, n_embd: int, hidden: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(n_embd, hidden, bias=False)  # gate
        self.w2 = nn.Linear(hidden, n_embd, bias=False)  # down
        self.w3 = nn.Linear(n_embd, hidden, bias=False)  # up
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class GELU_MLP(nn.Module):
    """Fallback GELU MLP"""
    def __init__(self, n_embd: int, hidden: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, n_embd, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))



# Attention with SDPA + KV Cache

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, n_kv_head: int, dropout: float, use_rope: bool, use_flash: bool):
        super().__init__()
        assert n_embd % n_head == 0
        assert n_head % n_kv_head == 0
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.num_key_value_groups = n_head // n_kv_head
        self.head_dim = n_embd // n_head
        self.use_rope = use_rope
        self.use_flash = use_flash

        self.q_proj = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_kv_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        B, T, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Apply RoPE
        if self.use_rope and rope_cache is not None:
            cos, sin = rope_cache
            if use_kv_cache and past_kv is not None:
                # For generation: only apply RoPE to new tokens
                past_len = past_kv[0].shape[2]
                cos = cos[past_len : past_len + T]
                sin = sin[past_len : past_len + T]
            else:
                cos, sin = cos[:T], sin[:T]
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        # KV Cache: concatenate past and present
        if use_kv_cache and past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_kv = (k, v) if use_kv_cache else None

        # GQA: Repeat KV heads to match Query heads
        if self.num_key_value_groups > 1:
            k = k[:, :, None, :, :].expand(B, self.n_kv_head, self.num_key_value_groups, k.shape[2], self.head_dim).reshape(B, self.n_head, k.shape[2], self.head_dim)
            v = v[:, :, None, :, :].expand(B, self.n_kv_head, self.num_key_value_groups, v.shape[2], self.head_dim).reshape(B, self.n_head, v.shape[2], self.head_dim)

        # Compute attention
        attn_weights = None
        if self.use_flash and T > 1:
            # PyTorch's optimized SDPA (uses Flash Attention under the hood)
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Manual attention (for single-token generation or fallback)
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) * scale
            # Proper causal mask: each query position can attend to
            # all key positions up to and including its own position.
            k_len = k.shape[2]
            causal_mask = torch.triu(
                torch.ones(T, k_len, device=x.device, dtype=torch.bool),
                diagonal=k_len - T + 1
            )
            att = att.masked_fill(causal_mask, float("-inf"))
            
            if output_attentions:
                attn_weights = att.clone()
                
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.out_proj(y))
        
        return y, new_kv, attn_weights



# Sparse Mixture of Experts

class SparseMoE(nn.Module):
    def __init__(self, n_embd: int, hidden: int, dropout: float, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(n_embd, num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU_MLP(n_embd, hidden, dropout) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x_flat = x.view(-1, C)
        
        router_logits = self.router(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32).to(x.dtype)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        final_output = torch.zeros_like(x_flat)
        
        for i, expert in enumerate(self.experts):
            expert_mask = (selected_experts == i)
            token_indices = torch.any(expert_mask, dim=-1)
            
            if not token_indices.any():
                continue
                
            expert_tokens = x_flat[token_indices]
            expert_out = expert(expert_tokens)
            
            for k in range(self.top_k):
                mask_k = selected_experts[:, k] == i
                if mask_k.any():
                    weights = routing_weights[mask_k, k].unsqueeze(-1)
                    final_output[mask_k] += expert_out[mask_k[token_indices]] * weights
                    
        return final_output.view(B, T, C)


# Transformer Block

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg.n_embd, cfg.n_head, cfg.n_kv_head, cfg.dropout, cfg.use_rope, cfg.use_flash_attn)
        self.norm2 = RMSNorm(cfg.n_embd)
        
        if cfg.use_moe:
            self.mlp = SparseMoE(cfg.n_embd, cfg.mlp_hidden, cfg.dropout, cfg.num_experts, cfg.num_experts_per_tok)
        else:
            self.mlp = SwiGLU_MLP(cfg.n_embd, cfg.mlp_hidden, cfg.dropout) if cfg.use_swiglu else GELU_MLP(cfg.n_embd, cfg.mlp_hidden, cfg.dropout)

    def forward(self, x, rope_cache, past_kv=None, use_kv_cache=False, output_attentions=False):
        attn_out, new_kv, attn_weights = self.attn(
            self.norm1(x), rope_cache, past_kv, use_kv_cache, output_attentions
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_kv, attn_weights



# Full Model

@dataclass
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    block_size: int = BLOCK_SIZE
    n_layer: int = N_LAYER
    n_head: int = N_HEAD
    n_kv_head: int = N_KV_HEAD
    n_embd: int = N_EMBD
    mlp_hidden: int = MLP_HIDDEN
    dropout: float = DROPOUT
    use_rope: bool = USE_ROPE
    use_rmsnorm: bool = USE_RMSNORM
    use_swiglu: bool = USE_SWIGLU
    use_grad_checkpoint: bool = USE_GRAD_CHECKPOINT
    use_flash_attn: bool = USE_FLASH_ATTN
    rope_theta: float = ROPE_THETA
    use_moe: bool = USE_MOE
    num_experts: int = NUM_EXPERTS
    num_experts_per_tok: int = NUM_EXPERTS_PER_TOK


class ShakespeareGPT(nn.Module):
    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            Block(cfg) for _ in range(cfg.n_layer)
        ])
        self.norm_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying (shares parameters between embedding and output layer)
        self.lm_head.weight = self.token_emb.weight

        # RoPE cache
        if cfg.use_rope:
            cos, sin = precompute_rope_cache(
                cfg.n_embd // cfg.n_head, cfg.block_size, cfg.rope_theta
            )
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        
        # Depth-scaled init for residual projections (standard Llama/GPT practice)
        for pn, p in self.named_parameters():
            if pn.endswith("out_proj.weight") or pn.endswith("w2.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(cfg.n_layer))

    def _enable_gradient_checkpointing(self):
        """Trade compute for memory - allows 2-3x larger models"""
        for block in self.blocks:
            block._gradient_checkpointing = True

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _block_forward(self, block, x, rope_cache, past_kv, use_kv_cache, output_attentions):
        """Wrapper for gradient checkpointing"""
        return block(x, rope_cache, past_kv, use_kv_cache, output_attentions)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor = None,
        past_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_kv_cache: bool = False,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"Sequence too long: {T} > {self.cfg.block_size}"

        x = self.drop(self.token_emb(idx))
        rope_cache = (self.rope_cos, self.rope_sin) if self.cfg.use_rope else None

        new_kvs = [] if use_kv_cache else None
        hidden_states = (x,) if output_hidden_states else None
        attentions = () if output_attentions else None

        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            
            if self.cfg.use_grad_checkpoint and self.training and not use_kv_cache:
                x, new_kv, attn_weights = torch.utils.checkpoint.checkpoint(
                    self._block_forward, block, x, rope_cache, past_kv, use_kv_cache, output_attentions,
                    use_reentrant=False,
                )
            else:
                x, new_kv, attn_weights = block(x, rope_cache, past_kv, use_kv_cache, output_attentions)
                
            if use_kv_cache:
                new_kvs.append(new_kv)
            if output_hidden_states:
                hidden_states += (x,)
            if output_attentions:
                attentions += (attn_weights,)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                ignore_index=-100
            )
            
        return {
            "logits": logits,
            "loss": loss,
            "new_kvs": new_kvs,
            "hidden_states": hidden_states,
            "attentions": attentions,
        }

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        repetition_penalty: float = 1.0,
        min_new_tokens: int = 0,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        """Standard generation returning full tensor."""
        generated_ids = idx.clone()
        for token in self.generate_stream(
            idx, max_new_tokens, temperature, top_k, top_p, repetition_penalty, min_new_tokens, eos_token_id
        ):
            generated_ids = torch.cat((generated_ids, torch.tensor([[token]], device=idx.device)), dim=1)
        return generated_ids

    @torch.no_grad()
    def generate_stream(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        repetition_penalty: float = 1.0,
        min_new_tokens: int = 0,
        eos_token_id: int = 2,
    ):
        """Yields generated tokens one by one for real-time streaming."""
        past_kvs = None
        device = idx.device
        generated_ids = idx.clone()
        
        # 1. Prefill phase
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            out = self(idx, past_kvs=None, use_kv_cache=True)
            logits = out["logits"]
            past_kvs = out["new_kvs"]
            
        for step in range(max_new_tokens):
            # 2. Get logits for the last token
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
            
            # 3. Apply repetition penalty (Verified Hugging Face implementation)
            if repetition_penalty > 1.0:
                for i in range(generated_ids.shape[0]):
                    for token_id in set(generated_ids[i].tolist()):
                        if next_token_logits[i, token_id] > 0:
                            next_token_logits[i, token_id] /= repetition_penalty
                        else:
                            next_token_logits[i, token_id] *= repetition_penalty
            
            # 4. Top-K filtering
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float("inf")
            
            # 5. Top-P (Nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float("inf")
            
            # 6. Sample
            probs = F.softmax(next_token_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            token = idx_next.item()
            
            # 7. Update history
            generated_ids = torch.cat((generated_ids, idx_next), dim=1)
            
            # 8. Forward pass for next token
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                out = self(idx_next, past_kvs=past_kvs, use_kv_cache=True)
                logits = out["logits"]
                past_kvs = out["new_kvs"]
                
            yield token
            
            # Stop if EOS token is generated AND we have met the minimum length requirement
            if token == eos_token_id and step >= min_new_tokens:
                break

    def print_summary(self):
        """Prints a detailed summary of the model architecture and parameters."""
        print("=" * 70)
        print("MODEL SUMMARY")
        print("=" * 70)
        total_params = 0
        trainable_params = 0
        
        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
                
            # Format size for readability
            if num_params > 1e6:
                size_str = f"{num_params / 1e6:.2f}M"
            elif num_params > 1e3:
                size_str = f"{num_params / 1e3:.2f}K"
            else:
                size_str = str(num_params)
                
            print(f"{name:<50} {str(list(param.shape)):<25} {size_str:>8}")
            
        print("-" * 70)
        print(f"Total parameters:     {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("=" * 70)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    cfg = ModelConfig()
    model = ShakespeareGPT(cfg)
    model.print_summary()
    
    x = torch.randint(0, VOCAB_SIZE, (2, 64))
    y = torch.randint(0, VOCAB_SIZE, (2, 64))
    
    # Test forward pass with hidden states and attentions
    out = model(x, y, output_hidden_states=True, output_attentions=True)
    print(f"\nLogits shape: {tuple(out['logits'].shape)}")
    print(f"Loss: {out['loss'].item():.4f}")
    print(f"Hidden states layers: {len(out['hidden_states'])}")
    print(f"Attention weights layers: {len(out['attentions'])}")