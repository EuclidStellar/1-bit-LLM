# ---------------------------------------------------------------------------
# Phase 2: the transformer, built seven times.
#
# Reading a finished transformer teaches you very little -- correct code looks
# inevitable. So this builds seven working language models, each adding exactly
# one component, each retrained on the identical budget. Every addition then has
# to justify itself with a measured loss.
#
# Run order: this comes BEFORE 02_train_kaggle.py. It is where the architecture
# was decided, and it is what makes the final config defensible rather than
# copied.
#
# Cost: about 25 minutes of T4 time for all seven rungs.
#
# MEASURED RESULTS (val loss, 20M tokens, identical harness):
#
#   rung  model                      val     ppl   params added   nats/M params
#   ----  -------------------------  ------  ----  -------------  ------------
#   0     uniform guess              8.3178  4096  0              --
#   1     unigram frequencies        6.0380   419  0              --
#   2     embedding + tied head      5.3828   218  1,310,720      0.50
#   3     + one attention layer      3.9106  49.9    409,600      3.59
#   4     + FFN (squared ReLU)       3.7150  41.1    819,200      0.24
#   5     + RoPE                     3.2948  27.0          0      infinite
#   6     + norms (RMSNorm/SubLN)    2.4304  11.4      2,560      337.7
#   7     8 layers                   2.0651   7.9  8,617,280      0.042
#
#   attention + norms + RoPE  =    412,160 params ( 3.7%) -> 69.5% of the gain
#   FFN + depth               =  9,436,480 params (84.6%) -> 14.1% of the gain
#
# Caveat that keeps this honest: 20M tokens against 11.16M params is 1.79
# tokens/param, about 11x under-trained, with a train/val gap of 0.014 -- so the
# model never even fit its data. Depth and FFN capacity are exactly the things
# that need data to pay off, so this ranks components AT THIS BUDGET, not in
# general.
# ---------------------------------------------------------------------------

# %% ------------------------------------------------------------------ cell 1
# Setup and the shared harness. Identical to 02_train_kaggle.py cells 1-2.

import sys, math, time, subprocess, importlib
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

V, D = 4096, 320
DS = "euclidstellar/tinystories-bpe4096"
paths = {f: hf_hub_download(DS, f, repo_type="dataset")
         for f in ["tokenizer.json", "train-20m.bin", "val.bin"]}
train = np.memmap(paths["train-20m.bin"], dtype=np.uint16, mode="r")
val   = np.memmap(paths["val.bin"],       dtype=np.uint16, mode="r")
tk    = Tokenizer.from_file(paths["tokenizer.json"])
dev   = "cuda" if torch.cuda.is_available() else "cpu"

TOKENS, BS, T = 20_000_000, 32, 256
STEPS, BASE_LR, WARMUP = TOKENS // (BS * T), 1e-3, 100


def get_batch(arr, bs, T):
    ix = np.random.randint(0, len(arr) - T - 1, size=bs)
    x = np.stack([arr[i:i + T]         for i in ix]).astype(np.int64)
    y = np.stack([arr[i + 1:i + 1 + T] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)


@torch.no_grad()
def evaluate(m, iters=60):
    m.eval()
    ls = [m(*get_batch(val, BS, T))[1].item() for _ in range(iters)]
    m.train()
    return float(np.mean(ls))


def run(Model, label, lr=BASE_LR):
    """Unchanged across every rung. lr is FIXED deliberately: tuning each rung
    separately would make the ladder a comparison of tuning, not architecture."""
    torch.manual_seed(1337); np.random.seed(1337)
    m = Model().to(dev)
    n = sum(p.numel() for p in {id(p): p for p in m.parameters()}.values())
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    t0 = time.time()
    for step in range(1, STEPS + 1):
        cur = (lr * step / WARMUP if step <= WARMUP else
               lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(
                   math.pi * (step - WARMUP) / (STEPS - WARMUP)))))
        for g in opt.param_groups:
            g["lr"] = cur
        x, y = get_batch(train, BS, T)
        opt.zero_grad(set_to_none=True)
        _, l = m(x, y)
        l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step % 500 == 0 or step == STEPS:
            print(f"  {step:>5}/{STEPS}  train {l.item():.4f}  lr {cur:.2e}")
    vl = evaluate(m)
    print(f"\n[{label}]  params {n:,}   VAL {vl:.4f}   ppl {math.exp(vl):.1f}"
          f"   {time.time() - t0:.0f}s")
    return m, vl


# %% ------------------------------------------------------------------ cell 2
# Rungs 0 and 1 need no model at all, and they are the anchors everything else
# is judged against.

c = np.bincount(np.asarray(train[:20_000_000]), minlength=V).astype(np.float64)
p = c / c.sum(); p = p[p > 0]
print(f"rung 0  uniform : {np.log(V):.4f}   ({V} effective choices)")
print(f"rung 1  unigram : {-(p * np.log(p)).sum():.4f}   "
      f"({np.exp(-(p * np.log(p)).sum()):.1f} effective choices)")
# Knowing ONLY how often each token appears -- no context, no network -- cuts
# effective choices from 4096 to 419. ~90% of the guessing, from counting.

# Optional third anchor: the full bigram table, 4096x4096 = 16.8M params.
# Conditional entropy on the same tokens comes to 3.6140 (37.1 choices). It is
# optimistic (only 3.0% of possible bigrams were ever observed) but it bounds
# what rung 2 could achieve, since rung 2 IS a bigram model at rank 320.


# %% ------------------------------------------------------------------ cell 3
# Rung 2: embedding + tied head. NO attention, no position, no depth.
#
# Each token predicts its successor from its own identity alone, so this is a
# bigram model squeezed through a 320-dimensional bottleneck:
#     true bigram table  4096 x 4096 = 16,777,216 params
#     this               4096 x  320 =  1,310,720 params   (12.8x smaller)
#
# It exists to prove the data loader, the loss and the shift-by-one are correct
# BEFORE attention exists to hide bugs in.

class Rung2(nn.Module):
    def __init__(self, vocab=V, d=D):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, d)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.weight        # tied: the SAME tensor

    def forward(self, idx, targets=None):
        logits = self.head(self.embed(idx))
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, self.vocab), targets.reshape(-1))
        return logits, loss


# An UNTRAINED model must score exactly ln(vocab): random weights mean uniform
# predictions. This one number validates initialization, the loss and the shapes.
m = Rung2().to(dev)
x, y = get_batch(train, 4, 16)
print(f"untrained: {m(x, y)[1].item():.4f}   expect {math.log(V):.4f}")
print(f"params: {sum(p.numel() for p in {id(p): p for p in m.parameters()}.values()):,}"
      f"   expect 1,310,720 (tied, counted once)")

m2, v2 = run(Rung2, "rung 2")                       # 5.3828
# Captured (6.0380-5.3828)/(6.0380-3.6140) = 27% of the available bigram
# information. Train 5.3708 vs val 5.3828: no overfitting at all, so it is
# architecture-limited, not data-limited. Exactly the state you want before
# adding attention.


# %% ------------------------------------------------------------------ cell 4
# Rung 3: one causal attention layer. No norms, no FFN, NO POSITION -- so it is
# a literal bag of words over the causal prefix.

class Rung3(nn.Module):
    def __init__(self, vocab=V, d=D, n_head=8):
        super().__init__()
        self.vocab, self.n_head, self.hd = vocab, n_head, d // n_head
        self.embed = nn.Embedding(vocab, d)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        for lin in (self.q, self.k, self.v, self.o):
            nn.init.normal_(lin.weight, std=0.02)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.weight

    def attn(self, x):
        B, T_, C = x.shape
        q = self.q(x).view(B, T_, self.n_head, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T_, self.n_head, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T_, self.n_head, self.hd).transpose(1, 2)
        # is_causal=True is the entire difference between a language model and a
        # model that reads the answer
        y_ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y_.transpose(1, 2).contiguous().view(B, T_, C))

    def forward(self, idx, targets=None):
        x = self.embed(idx)
        x = x + self.attn(x)                        # residual keeps identity
        logits = self.head(x)
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, self.vocab), targets.reshape(-1))
        return logits, loss


m3, v3 = run(Rung3, "rung 3")                       # 3.9106
# -1.4722 nats for 409,600 params = 3.59 nats/M. The biggest single jump on the
# ladder, and ~7x more parameter-efficient than the embedding.


# %% ------------------------------------------------------------------ cell 5
# Rung 4: + FFN with squared ReLU. BitNet picks ReLU^2 over SwiGLU for sparsity:
# squaring crushes small activations toward zero, and sparse activations
# tolerate low-bit weights better. Two matrices instead of SwiGLU's three.

class Rung4(Rung3):
    def __init__(self, vocab=V, d=D, n_head=8, mult=4):
        super().__init__(vocab, d, n_head)
        self.up = nn.Linear(d, mult * d, bias=False)
        self.down = nn.Linear(mult * d, d, bias=False)
        for lin in (self.up, self.down):
            nn.init.normal_(lin.weight, std=0.02)

    def ffn(self, x):
        return self.down(F.relu(self.up(x)).pow(2))

    def forward(self, idx, targets=None):
        x = self.embed(idx)
        x = x + self.attn(x)
        x = x + self.ffn(x)                         # the only new line
        logits = self.head(x)
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, self.vocab), targets.reshape(-1))
        return logits, loss


m4, v4 = run(Rung4, "rung 4")                       # 3.7150
# Only -0.1956 nats for 819,200 params = 0.24 nats/M. TWICE attention's
# parameter cost for an EIGHTH of the gain -- 15x less efficient. Handicapped
# here by having one layer (nothing composed to process) and no normalization.


# %% ------------------------------------------------------------------ cell 6
# Rung 5: + RoPE. Position arrives as a ROTATION of paired channels, not an
# added vector, which is why the attention score depends only on relative
# distance. ZERO new parameters.

class Rung5(Rung4):
    def __init__(self, vocab=V, d=D, n_head=8, mult=4, base=10000.0, maxT=1024):
        super().__init__(vocab, d, n_head, mult)
        half = self.hd // 2
        inv = 1.0 / (base ** (torch.arange(half).float() / half))
        f = torch.outer(torch.arange(maxT).float(), inv)
        self.register_buffer("cos", f.cos(), persistent=False)
        self.register_buffer("sin", f.sin(), persistent=False)

    @staticmethod
    def rope(x, cos, sin):
        T_, Dh = x.shape[-2], x.shape[-1]
        x1, x2 = x[..., : Dh // 2], x[..., Dh // 2:]
        c = cos[:T_].view(1, 1, T_, Dh // 2).to(x.dtype)
        s = sin[:T_].view(1, 1, T_, Dh // 2).to(x.dtype)
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)

    def attn(self, x):
        B, T_, C = x.shape
        q = self.q(x).view(B, T_, self.n_head, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T_, self.n_head, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T_, self.n_head, self.hd).transpose(1, 2)
        q, k = self.rope(q, self.cos, self.sin), self.rope(k, self.cos, self.sin)
        y_ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y_.transpose(1, 2).contiguous().view(B, T_, C))


m5, v5 = run(Rung5, "rung 5")                       # 3.2948
# -0.4202 nats for ZERO parameters -- more than TWICE the FFN's gain, free, and
# only 4% more wall clock. First rung below the bigram bound of 3.6140.

# Verify the mechanism: same content, same offset, wildly different absolute
# positions -> identical score. MEASURED identical to six significant figures
# at positions 5, 50, 200 and 500.
mm = Rung5().to(dev)
qc, kc = torch.randn(mm.hd, device=dev), torch.randn(mm.hd, device=dev)
def score(pq, pk):
    q = torch.zeros(1, 1, pq + 1, mm.hd, device=dev); q[0, 0, pq] = qc
    k = torch.zeros(1, 1, pk + 1, mm.hd, device=dev); k[0, 0, pk] = kc
    qr, kr = mm.rope(q, mm.cos, mm.sin), mm.rope(k, mm.cos, mm.sin)
    return (qr[0, 0, pq] * kr[0, 0, pk]).sum().item()
for pp in (5, 50, 200, 500):
    print(f"  q at {pp:>3}, k at {pp-3:>3}: {score(pp, pp-3):.6f}")


# %% ------------------------------------------------------------------ cell 7
# Rung 6: + normalization. STILL ONE LAYER, so this isolates norms from depth.
# SubLN -- the extra norm before each sublayer's output projection -- is the
# Magneto trick BitNet adopts, and it is what keeps the input to each sublayer's
# LAST matmul in a predictable range. Matters far more once those are ternary.

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dt = x.dtype
        x = x.float()                               # fp32 regardless of autocast
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


class Rung6(Rung5):
    def __init__(self, vocab=V, d=D, n_head=8, mult=4):
        super().__init__(vocab, d, n_head, mult)
        self.attn_norm  = RMSNorm(d)
        self.subln      = RMSNorm(d)
        self.ffn_norm   = RMSNorm(d)
        self.ffn_subln  = RMSNorm(mult * d)
        self.final_norm = RMSNorm(d)

    def attn(self, x):
        B, T_, C = x.shape
        q = self.q(x).view(B, T_, self.n_head, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T_, self.n_head, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T_, self.n_head, self.hd).transpose(1, 2)
        q, k = self.rope(q, self.cos, self.sin), self.rope(k, self.cos, self.sin)
        y_ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y_ = y_.transpose(1, 2).contiguous().view(B, T_, C)
        return self.o(self.subln(y_))               # SubLN

    def ffn(self, x):
        return self.down(self.ffn_subln(F.relu(self.up(x)).pow(2)))

    def forward(self, idx, targets=None):
        x = self.embed(idx)
        x = x + self.attn(self.attn_norm(x))        # pre-norm
        x = x + self.ffn(self.ffn_norm(x))
        logits = self.head(self.final_norm(x))
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, self.vocab), targets.reshape(-1))
        return logits, loss


m6, v6 = run(Rung6, "rung 6")                       # 2.4304
# -0.8644 nats for 2,560 params = 337.7 nats/M. The efficiency champion:
# 94x attention, 1400x the FFN. 0.1% of the model, second-largest gain.

# The hypothesis going in was that norms would UNLOCK the FFN, since ReLU^2 is
# scale-sensitive. Wrong -- they partly REPLACED it:
class Rung6NoFFN(Rung6):
    def forward(self, idx, targets=None):
        x = self.embed(idx)
        x = x + self.attn(self.attn_norm(x))
        logits = self.head(self.final_norm(x))
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, self.vocab), targets.reshape(-1))
        return logits, loss

_, v6_noffn = run(Rung6NoFFN, "rung 6 minus FFN")   # 2.5479
print(f"FFN without norms (r3->r4): {v3 - v4:+.4f}")
print(f"FFN with    norms (->r6)  : {v6_noffn - v6:+.4f}")
# +0.1956 -> +0.1175, 40% SMALLER. Part of what an unnormalized FFN buys is
# scale correction, not computation -- and 2,560 dedicated parameters do that
# job better and far cheaper.


# %% ------------------------------------------------------------------ cell 8
# Rung 7: eight layers. This is the final architecture, and the parameter count
# should come out at exactly 11,159,360.

class Block(nn.Module):
    def __init__(self, d, n_head, mult=4):
        super().__init__()
        self.n_head, self.hd = n_head, d // n_head
        self.attn_norm = RMSNorm(d)
        self.q = nn.Linear(d, d, bias=False); self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False); self.o = nn.Linear(d, d, bias=False)
        self.subln = RMSNorm(d)
        self.ffn_norm = RMSNorm(d)
        self.up = nn.Linear(d, mult * d, bias=False)
        self.down = nn.Linear(mult * d, d, bias=False)
        self.ffn_subln = RMSNorm(mult * d)
        for lin in (self.q, self.k, self.v, self.o, self.up, self.down):
            nn.init.normal_(lin.weight, std=0.02)

    def attn(self, x, cos, sin):
        B, T_, C = x.shape
        q = self.q(x).view(B, T_, self.n_head, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T_, self.n_head, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T_, self.n_head, self.hd).transpose(1, 2)
        q, k = Rung5.rope(q, cos, sin), Rung5.rope(k, cos, sin)
        y_ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(self.subln(y_.transpose(1, 2).contiguous().view(B, T_, C)))

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.down(self.ffn_subln(F.relu(self.up(self.ffn_norm(x))).pow(2)))
        return x


class Rung7(nn.Module):
    def __init__(self, vocab=V, d=D, n_layer=8, n_head=8, mult=4,
                 base=10000.0, maxT=1024):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, d)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.blocks = nn.ModuleList(Block(d, n_head, mult) for _ in range(n_layer))
        self.final_norm = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.weight
        half = (d // n_head) // 2
        inv = 1.0 / (base ** (torch.arange(half).float() / half))
        f = torch.outer(torch.arange(maxT).float(), inv)
        self.register_buffer("cos", f.cos(), persistent=False)
        self.register_buffer("sin", f.sin(), persistent=False)
        # Scale whatever writes INTO the residual stream, or its variance grows
        # with depth and early training is unstable.
        with torch.no_grad():
            sc = 1.0 / math.sqrt(2 * n_layer)
            for b in self.blocks:
                b.o.weight.mul_(sc); b.down.weight.mul_(sc)

    def forward(self, idx, targets=None):
        x = self.embed(idx)
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        logits = self.head(self.final_norm(x))
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, self.vocab), targets.reshape(-1))
        return logits, loss


m7, v7 = run(Rung7, "rung 7")                       # 2.0651, 11,159,360 params
# -0.3653 nats for 8,617,280 params = 0.042 nats/M -- the WORST value on the
# ladder. But see the caveat at the top: at 1.79 tokens/param the model is
# starved, and depth is exactly what needs data. This ranks components at THIS
# budget.


# %% ------------------------------------------------------------------ cell 9
# The two correctness tests that catch bugs which raise no exception.

@torch.no_grad()
def causality_test(m, label=""):
    m.eval()
    x = torch.randint(0, V, (1, 16)).to(dev)
    a = m(x)[0][0, 5].clone()
    x[0, 6] = (x[0, 6] + 1) % V              # perturb a LATER token
    d = (a - m(x)[0][0, 5]).abs().max().item()
    print(f"causality {label}: {'PASS' if d < 1e-4 else 'FAIL, MASK LEAKING'}"
          f"  (delta {d:.2e})")
    m.train()

causality_test(m7, "rung7")                  # MEASURED delta exactly 0.00e+00


# Position, isolated. target[t] = input[t-2] over an alphabet of 8, iid -- so the
# current token carries ZERO information about the target and a context-free
# model is pinned at ln(8) = 2.0794. Memorization is impossible.
def copy_task(back=2, bs=16, T_=65, alphabet=8):
    s = torch.randint(0, alphabet, (bs, T_ + back))
    return s[:, back:].to(dev), s[:, :-back].to(dev)

def fit(Model, inp, tgt, steps=800, lr=3e-3):
    torch.manual_seed(0)
    mm = Model().to(dev)
    opt = torch.optim.AdamW(mm.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        _, l = mm(inp, tgt)
        l.backward()
        torch.nn.utils.clip_grad_norm_(mm.parameters(), 1.0)   # NOT optional
        opt.step()
    return l.item()

inp, tgt = copy_task()
print(f"\ncopy-from-2-back, floor ln(8) = {math.log(8):.4f}")
for name, M in [("rung 2  no context", Rung2), ("rung 3  unordered", Rung3),
                ("rung 5  ordered  ", Rung5)]:
    print(f"  {name}: {fit(M, inp, tgt):.4f}")
# MEASURED  rung 2: 2.0665 (pinned at the floor, flat trajectory -- no context)
#           rung 3: 1.5458 (beats the floor: the prefix MULTISET is informative
#                           even unordered, which contradicted my prediction)
#           rung 5: 0.0144 (solved -- it can point at exactly two back)
#
# Without clip_grad_norm_ this same rung-5 run gave 8.0065, i.e. worse than
# random over the full 4096 vocabulary. One line, total collapse vs solved.


# %% ----------------------------------------------------------------- cell 10
print(f"\n{'rung':<28}{'val':>9}{'ppl':>9}")
print("-" * 46)
for lbl, vv in [("0 uniform", math.log(V)), ("1 unigram", 6.0380),
                ("2 embedding + tied head", v2), ("3 + attention", v3),
                ("4 + FFN", v4), ("5 + RoPE", v5), ("6 + norms", v6),
                ("7 eight layers", v7)]:
    print(f"{lbl:<28}{vv:>9.4f}{math.exp(vv):>9.1f}")
