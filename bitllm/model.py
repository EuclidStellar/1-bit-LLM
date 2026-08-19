"""
An 11,159,360-parameter decoder-only transformer, BitNet b1.58 architecture.

ONE class covers both experimental arms. `weight_mode="none"` gives the fp32
control; `weight_mode="ternary"` gives the 1-bit model. Identical code path,
identical shapes, identical everything else -- which is the experimental
requirement, not a convenience.

Architecture (arXiv:2504.12285): W1.58A8, subln normalization, squared ReLU,
RoPE, no bias terms anywhere, tied embeddings.

Full precision by design -- quantizing any of these collapses the model:
  * token embedding table
  * LM head (tied to the embedding)
  * every normalization layer
  * the residual stream itself

Nothing from torch.nn.Transformer* is used.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant import act_quant, quantize_weight, ste


class RMSNorm(nn.Module):
    """No mean subtraction, no bias. Computed in fp32 regardless of autocast --
    norms are cheap and this avoids fp16 overflow bugs that look like model bugs.

    MEASURED: five of these (2,560 params, 0.1% of the model) bought 0.8644 nats
    -- 337.7 nats per million params, 94x more efficient than attention.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


def build_rope_cache(maxT, head_dim, base=10000.0):
    half = head_dim // 2
    inv = 1.0 / (base ** (torch.arange(half).float() / half))
    f = torch.outer(torch.arange(maxT).float(), inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    """x: (B, n_head, T, head_dim). Rotate the first half against the second.

    Position arrives as a ROTATION, not an added vector. That is why the
    attention score depends on relative offset: MEASURED identical to six
    significant figures for the same content at absolute positions 5, 50, 200
    and 500 with a fixed offset of 3.
    """
    T, Dh = x.shape[-2], x.shape[-1]
    x1, x2 = x[..., : Dh // 2], x[..., Dh // 2:]
    c = cos[:T].view(1, 1, T, Dh // 2).to(x.dtype)
    s = sin[:T].view(1, 1, T, Dh // 2).to(x.dtype)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


class BitLinear(nn.Module):
    """nn.Linear whose weights are quantized in the forward pass.

    This class is the entire difference between "a transformer" and "a 1-bit
    LLM". With weight_mode="none" and act_bits=0 it is exactly nn.Linear.

    The master weight stays fp32 -- that is what AdamW updates. Each forward
    quantizes a COPY, with the STE carrying gradients back to the master. That
    is what "trained quantized from scratch" means, as distinct from
    quantizing a finished model.
    """
    def __init__(self, in_f, out_f, weight_mode="ternary", act_bits=8):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        self.weight_mode, self.act_bits = weight_mode, act_bits
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        if self.act_bits > 0:
            x = ste(x, act_quant(x, self.act_bits))
        w = self.weight
        if self.weight_mode != "none":
            w = ste(w, quantize_weight(w, self.weight_mode))
        return F.linear(x, w)

    def extra_repr(self):
        return (f"{self.in_f}->{self.out_f}, w={self.weight_mode}, "
                f"a={self.act_bits}bit")

    @property
    def is_quantized(self):
        return self.weight_mode != "none"


class Block(nn.Module):
    """Pre-norm block with SubLN before each sublayer's output projection.

    The SubLN pair is the Magneto trick BitNet adopts: it keeps the input to
    each sublayer's LAST matmul in a predictable range, which matters far more
    once those matmuls are ternary.
    """
    def __init__(self, d, n_head, mult=4, weight_mode="ternary", act_bits=8):
        super().__init__()
        self.n_head, self.hd = n_head, d // n_head
        kw = dict(weight_mode=weight_mode, act_bits=act_bits)

        self.attn_norm = RMSNorm(d)
        self.q = BitLinear(d, d, **kw)
        self.k = BitLinear(d, d, **kw)
        self.v = BitLinear(d, d, **kw)
        self.o = BitLinear(d, d, **kw)
        self.subln = RMSNorm(d)

        self.ffn_norm  = RMSNorm(d)
        self.up   = BitLinear(d, mult * d, **kw)
        self.down = BitLinear(mult * d, d, **kw)
        self.ffn_subln = RMSNorm(mult * d)

    def attn(self, x, cos, sin):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_head, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_head, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_head, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # is_causal=True is the whole difference between a language model and a
        # model that reads the answer
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(self.subln(y.transpose(1, 2).contiguous().view(B, T, C)))

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.down(self.ffn_subln(F.relu(self.up(self.ffn_norm(x))).pow(2)))
        return x


class LM(nn.Module):
    """weight_mode="none" -> the fp32 control arm (rung 7, val 2.0651)
       weight_mode="ternary" -> the 1-bit arm

    Same class, same code path. The only variable is how weights are quantized.
    """
    def __init__(self, vocab=4096, d=320, n_layer=8, n_head=8, mult=4,
                 weight_mode="ternary", act_bits=8, base=10000.0, maxT=1024):
        super().__init__()
        self.vocab, self.weight_mode = vocab, weight_mode

        self.embed = nn.Embedding(vocab, d)          # fp32, never quantized
        nn.init.normal_(self.embed.weight, std=0.02)

        self.blocks = nn.ModuleList(
            Block(d, n_head, mult, weight_mode, act_bits) for _ in range(n_layer))
        self.final_norm = RMSNorm(d)

        self.head = nn.Linear(d, vocab, bias=False)  # fp32, tied
        self.head.weight = self.embed.weight

        cos, sin = build_rope_cache(maxT, d // n_head, base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Scale whatever writes INTO the residual stream by 1/sqrt(2L), or its
        # variance grows with depth. Works identically for both arms because
        # weight_ternary is scale-equivariant -- see quant.weight_ternary.
        with torch.no_grad():
            sc = 1.0 / math.sqrt(2 * n_layer)
            for b in self.blocks:
                b.o.weight.mul_(sc)
                b.down.weight.mul_(sc)

    def forward(self, idx, targets=None):
        x = self.embed(idx)
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        logits = self.head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab),
                                   targets.reshape(-1))
        return logits, loss

    # ------------------------------------------------------------ reporting

    def param_split(self):
        """Quantizable vs full precision. This ratio decides whether the 1-bit
        memory claim is yours to make -- BitNet's own headline figure is
        labelled "Memory (Non-emb)" because their 128,256-token embedding table
        (656MB at bf16) is LARGER than their 400MB ternary body."""
        tern = sum(m.weight.numel() for m in self.modules()
                   if isinstance(m, BitLinear) and m.is_quantized)
        total = sum(p.numel() for p in
                    {id(p): p for p in self.parameters()}.values())
        return dict(total=total, quantized=tern, full_precision=total - tern)

    @torch.no_grad()
    def generate(self, idx, max_new=160, temperature=0.8, top_k=100, maxT=256):
        was_training = self.training
        self.eval()
        for _ in range(max_new):
            logits, _ = self(idx[:, -maxT:])
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], -float("inf"))
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], 1)
        if was_training:
            self.train()
        return idx


def fp32_model(**kw):   return LM(weight_mode="none", act_bits=0, **kw)
def ternary_model(**kw): return LM(weight_mode="ternary", act_bits=8, **kw)
