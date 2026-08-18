"""
A decoder-only transformer, written to be read top to bottom.

Architecture follows the BitNet b1.58 2B4T recipe (arXiv:2504.12285):

    * BitLinear in place of nn.Linear for all six projection matrices
    * SubLN -- an extra normalization inside each sublayer
    * squared ReLU in the FFN instead of SwiGLU (chosen for sparsity)
    * RoPE for positional information
    * no bias terms anywhere, in linear or normalization layers

Deliberately kept in full precision, because quantizing any of these
collapses the model into noise:

    * the token embedding table
    * the LM head (tied to the embedding here, as BitNet does)
    * every normalization layer
    * the residual stream itself

Nothing from torch.nn.Transformer* is used. Attention, RoPE, the FFN and the
norms are all spelled out.
"""

from dataclasses import dataclass, field
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant import act_quant, quantize_weight, ste


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

@dataclass
class Config:
    vocab_size: int = 4096
    n_layer: int = 8
    n_head: int = 6
    n_embd: int = 384
    ffn_mult: int = 4              # FFN inner dim = ffn_mult * n_embd
    block_size: int = 512          # max context length
    rope_base: float = 10000.0

    # --- quantization ---------------------------------------------------
    weight_mode: str = "ternary"   # see quant.WEIGHT_MODES
    act_bits: int = 8              # 0 disables activation quantization
    # Keeping the first and last block in full precision is NOT part of
    # BitNet's recipe. It is scaffolding for the pilot run: if fp32 works and
    # fully-quantized does not, this flag tells you whether the problem is
    # your STE or the edge blocks. Turn it off for the real run.
    keep_edge_blocks_fp: bool = False

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must divide by n_head"

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def ffn_dim(self) -> int:
        return self.ffn_mult * self.n_embd


# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """RMSNorm, no bias -- BitNet strips bias from normalization too."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # compute in fp32 regardless of autocast: norms are cheap and this
        # avoids a class of fp16 overflow bugs that look like model bugs
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class BitLinear(nn.Module):
    """nn.Linear, except the weights are quantized in the forward pass.

    This class is the entire difference between "a transformer" and "a 1-bit
    LLM". Everything else in this file is an ordinary transformer.

    The master weight stays full precision -- it is what the optimizer
    updates. Each forward pass quantizes a *copy* of it and uses that for the
    matmul, with the STE carrying gradients back to the master copy. That is
    what "trained quantized from scratch" means, as opposed to quantizing a
    finished model afterwards.
    """

    def __init__(self, in_features: int, out_features: int, cfg: Config,
                 weight_mode: str | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mode = cfg.weight_mode if weight_mode is None else weight_mode
        self.act_bits = cfg.act_bits

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # activations: per-token absmax int8, through the STE
        if self.act_bits > 0:
            x = ste(x, act_quant(x, self.act_bits))

        # weights: ternary via per-tensor absmean, through the STE.
        # The master weight is kept in its own dtype (fp32) so the quantizer
        # arithmetic never happens in fp16 -- autocast will downcast for the
        # matmul itself, which is where the speed actually comes from.
        w = self.weight
        if self.weight_mode != "none":
            w = ste(w, quantize_weight(w, self.weight_mode))

        return F.linear(x, w)

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"w={self.weight_mode}, a={self.act_bits}bit")


# --------------------------------------------------------------------------
# rotary position embedding
# --------------------------------------------------------------------------

def build_rope_cache(seq_len: int, head_dim: int, base: float,
                     device, dtype=torch.float32):
    """Precompute cos/sin tables of shape (seq_len, head_dim // 2)."""
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)                 # (T, half)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x, cos, sin):
    """Rotate pairs of channels by a position-dependent angle.

    x is (B, H, T, D). The first half of D is paired with the second half --
    the GPT-NeoX / LLaMA convention. Position information enters as a
    rotation rather than an added vector, which is why it extrapolates
    better than learned absolute embeddings.
    """
    T, D = x.shape[-2], x.shape[-1]
    x1, x2 = x[..., : D // 2], x[..., D // 2:]
    cos = cos[:T].view(1, 1, T, D // 2).to(x.dtype)
    sin = sin[:T].view(1, 1, T, D // 2).to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# --------------------------------------------------------------------------
# attention and FFN
# --------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, cfg: Config, weight_mode: str | None = None):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim

        self.q_proj = BitLinear(cfg.n_embd, cfg.n_embd, cfg, weight_mode)
        self.k_proj = BitLinear(cfg.n_embd, cfg.n_embd, cfg, weight_mode)
        self.v_proj = BitLinear(cfg.n_embd, cfg.n_embd, cfg, weight_mode)
        self.o_proj = BitLinear(cfg.n_embd, cfg.n_embd, cfg, weight_mode)

        # SubLN: an extra norm before the output projection. This is the
        # Magneto / Foundation-Transformer trick that BitNet adopts, and it
        # is load-bearing for low-bit stability -- it keeps the input to the
        # last quantized matmul in a predictable range.
        self.subln = RMSNorm(cfg.n_embd)

    def forward(self, x, cos, sin):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # causal masking is what makes this a *language* model: position t
        # may only attend to positions <= t
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(self.subln(y))


class FeedForward(nn.Module):
    """Two projections with squared ReLU between them.

    BitNet picks ReLU^2 over SwiGLU specifically for sparsity: squaring
    pushes small activations toward zero, and a sparser activation vector
    interacts better with low-bit weights. It also means the FFN has two
    matrices rather than SwiGLU's three.
    """

    def __init__(self, cfg: Config, weight_mode: str | None = None):
        super().__init__()
        self.up = BitLinear(cfg.n_embd, cfg.ffn_dim, cfg, weight_mode)
        self.down = BitLinear(cfg.ffn_dim, cfg.n_embd, cfg, weight_mode)
        self.subln = RMSNorm(cfg.ffn_dim)   # SubLN again, same reasoning

    def forward(self, x):
        h = self.up(x)
        h = F.relu(h).pow(2)
        return self.down(self.subln(h))


class Block(nn.Module):
    """Pre-norm transformer block. Residual stream stays full precision."""

    def __init__(self, cfg: Config, weight_mode: str | None = None):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg, weight_mode)
        self.ffn_norm = RMSNorm(cfg.n_embd)
        self.ffn = FeedForward(cfg, weight_mode)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------

class BitLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # full precision, deliberately
        self.embed = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

        modes = [cfg.weight_mode] * cfg.n_layer
        if cfg.keep_edge_blocks_fp and cfg.n_layer >= 3:
            modes[0] = modes[-1] = "none"
        self.blocks = nn.ModuleList(Block(cfg, m) for m in modes)

        self.final_norm = RMSNorm(cfg.n_embd)

        # LM head tied to the embedding, as in BitNet's config
        # (tie_word_embeddings: true). Saves vocab*n_embd parameters and
        # keeps the un-quantizable share of the model as small as possible.
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        cos, sin = build_rope_cache(cfg.block_size, cfg.head_dim,
                                    cfg.rope_base, device="cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, \
            f"sequence length {T} exceeds block_size {self.cfg.block_size}"

        x = self.embed(idx)
        cos = self.rope_cos.to(x.device)
        sin = self.rope_sin.to(x.device)

        for block in self.blocks:
            x = block(x, cos, sin)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss

    # ---------------------------------------------------------------- utils

    def param_report(self) -> dict:
        """Split the parameter count into quantizable and not.

        This is the number that decides whether the memory story is real. If
        the full-precision share dominates, ternarizing the rest buys you
        very little -- which is exactly the trap in BitNet's own headline
        figure, where the 0.4GB is labelled "non-embedding".
        """
        quantizable = 0
        full_precision = 0
        for module in self.modules():
            if isinstance(module, BitLinear) and module.weight_mode != "none":
                quantizable += module.weight.numel()
        seen = set()
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            full_precision += p.numel()
        full_precision -= quantizable

        bpw = __import__("math").log2(3) if self.cfg.weight_mode == "ternary" \
            else {"binary": 1.0, "none": 32.0}.get(self.cfg.weight_mode, 8.0)

        return {
            "total": quantizable + full_precision,
            "quantizable": quantizable,
            "full_precision": full_precision,
            "packed_bytes_quantizable": int(quantizable * bpw / 8),
            "packed_bytes_fp16_rest": full_precision * 2,
        }

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8,
                 top_k: int | None = 200):
        """Plain autoregressive sampling. No KV cache -- that is an inference
        concern and belongs with the packed-weight work."""
        self.eval()
        for _ in range(max_new_tokens):
            window = idx[:, -self.cfg.block_size:]
            logits, _ = self(window)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                thresh = torch.topk(logits, k)[0][..., -1, None]
                logits = logits.masked_fill(logits < thresh, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


def apply_residual_scaling(model: BitLM):
    """Scale the residual-path output projections after construction.

    Kept as a function rather than folded into __init__ so the reason stays
    visible: o_proj and ffn.down write directly into the residual stream, so
    their initial scale controls how fast variance accumulates with depth.
    """
    n_layer = model.cfg.n_layer
    scale = 1.0 / math.sqrt(2 * n_layer)
    with torch.no_grad():
        for block in model.blocks:
            block.attn.o_proj.weight.mul_(scale)
            block.ffn.down.weight.mul_(scale)
    return model


def build_model(cfg: Config) -> BitLM:
    return apply_residual_scaling(BitLM(cfg))
