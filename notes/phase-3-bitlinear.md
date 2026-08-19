# Phase 3 — Making it 1-bit

Same architecture, same data, same harness. One class, `weight_mode` selects the
arm, so the two arms cannot differ in anything except quantization.

## The headline result

| arm | val loss | ppl | when quantized |
|---|---|---|---|
| fp32 control | **2.0553** | 7.8 | never |
| **QAT-ternary** | **2.1760** | 8.8 | inside the training loop |
| PTQ-ternary | **5.0229** | 151.9 | after training |

```
QAT gap to fp32          : +0.1207 nats   (5.9% relative)
QAT recovered            : 95.9% of what PTQ destroyed
PTQ-vs-QAT difference    : 2.85 nats, 17x perplexity
```

**Nothing differs between QAT and PTQ except *when* the quantizer was applied.**
Identical architecture, identical ternary precision, identical weights format.
That single scheduling choice is worth 2.85 nats.

Wall clock: 916s vs 820s fp32 = **1.12x**. (Measured 1.03x on an M4 earlier; the
overhead is elementwise quantization ops, negligible against the matmuls.)

Causality max delta 0.00e+00 on all three arms.

## Why PTQ collapses — measured, not asserted

Ternarization applied to the trained fp32 weights:

```
whole-model relative error : 0.531        <- 53% change to every weight matrix
cosine similarity          : ~0.88 per matrix, uniform across all 48
```

| | |
|---|---|
| each matrix keeps | **88% of its direction** |
| the model keeps | **25% of its learned function** |

That gap is **composition**. A 12% angular deviation per matrix, compounded
through 48 matrices across 8 layers, destroys three quarters of what training
built. This is precisely why QAT works and PTQ cannot: QAT lets the network
*adapt around* the quantization, rather than having it imposed on a solution
that assumed full precision.

Where PTQ lands on the Phase 2 ladder is the clearest way to say it:

| | val loss |
|---|---|
| rung 2 — embedding only, no attention | 5.3828 |
| **PTQ-ternary — the full 8-layer model** | **5.0229** |
| rung 3 — one attention layer | 3.9106 |

**A post-quantized 8-layer transformer performs worse than a single attention
layer in full precision.**

## The weight distribution stays Gaussian through training

```
trained model zero fraction : 31.2% - 32.3%, uniform across all 48 matrices
closed-form Gaussian value  : 2*Phi(0.5*sqrt(2/pi)) - 1 = 31.01%
```

After 20M tokens, training changed *which* weights are large but not the *shape*
of the distribution — so the analytic prediction derived for an untrained model
still holds on a trained one. Consequence: **the zero fraction is a poor progress
signal at this scale.** Do not expect to watch the quantizer "learn" in it.

## Gradient flow is inverted from the usual worry

```
block 0 q.weight grad norm: 3.702e-02      furthest from the loss, LARGEST
block 3                   : 2.092e-03
block 7                   : 1.503e-03      closest to the loss, smallest
embed                     : 1.855e+00
```

Not vanishing at the bottom — **growing** toward it, 24.6x from block 7 to
block 0. Mechanism: RMSNorm's backward pass scales inversely with input
magnitude, and the residual stream accumulates magnitude with depth, so deeper
layers normalize by a larger number and their gradients shrink.

Not a blocker, because **Adam divides by per-parameter gradient RMS** — a
consistently smaller gradient still receives a full-sized step. `embed` is large
because it is tied and collects gradient from both the input lookup and the
output projection.

## The memory story, honestly

```
ternary body     9,830,400 x 1.585 bits =  1.95 MB
full precision   1,328,960 x 16 bits    =  2.66 MB
packed total                               4.61 MB
same model at fp16                        22.32 MB     -> 4.85x compression
same model at fp32                        44.64 MB     -> 9.69x compression

>>> full-precision share of the packed file: 57.7%
```

**More than half of our "1-bit model" is full precision**, at vocab 4,096 — the
small vocabulary chosen specifically to avoid this. The embedding-table asterisk
from Phase 1 reappears in our own model. Depth is the only fix:

| layers | packed MB | fp share |
|---|---|---|
| **8 (ours)** | **4.61** | **57.7%** |
| 16 | 6.59 | 40.9% |
| 32 | 10.56 | 26.2% |

Same conclusion the Phase 2 ladder reached independently: **add layers, never
vocabulary.**

## Two comparisons, and only one of them is the interesting one

At **equal parameter count**, ternary loses by 0.1207 nats. Unavoidable — 1.58
bits carries less information than 32.

At **equal memory budget**, the question is open: 4.61 MB of ternary buys
11,159,360 params, while 4.21 MB of fp16 buys only 2,104,448. That comparison is
the one BitNet actually claims, and it is the one worth running.

```
                                        val     ppl      MB
fp16    d=320   11,159,360 params    2.0553     7.8   22.32
ternary d=320   11,159,360 params    2.1760     8.8    4.61
fp16    d=128    2,104,448 params    2.3607    10.6    4.21
```

The fp16 comparison model received 9.5% fewer bytes, so correcting by
log-linear interpolation between the two fp16 points:

```
fp16 params that fit in exactly 4.61 MB : 2,302,761   (d ~ 135)
log-linear estimate of its loss         : 2.3442
ternary measured                        : 2.1760
  -> ternary wins by +0.1682 nats after correction
```

### Both statements are true, and both are measured

| comparison | result |
|---|---|
| **equal parameter count** | ternary **loses** by 0.1207 nats (2.1760 vs 2.0553) |
| **equal memory budget** | ternary **wins** by ~0.168 nats (2.1760 vs ~2.344) |

4.61 MB buys **4.85x more parameters** as ternary than as fp16. Reporting only
the first comparison understates ternary; reporting only the second overstates
it. The pair is the finding.

To measure rather than interpolate the fair point: `d=136, n_head=4` gives
2,340,424 params at 4.68 MB, within 1.5% of the ternary budget.

## Verified along the way

- 48 BitLinear modules, 6 per block. Embedding, tied LM head, and all 41 norms
  untouched. 9,830,400 ternary (88.1%) / 1,328,960 fp (11.9%), total 11,159,360.
- `weight_ternary` produces exactly 3 distinct values per tensor.
- Untrained ternary loss ~= ln(4096), so quantization does not break init.
- STE gradient = 1.0 with `.detach()`, 7.12e-06 without. The residual is not the
  rounding (which has exactly zero gradient) but the differentiable absmean
  *scale*, which appears twice in `weight_ternary`. Detach only the scale and the
  gradient is exactly 0.
- `weight_ternary` is scale-equivariant: `weight_ternary(c*w) == c*weight_ternary(w)`,
  so the `1/sqrt(2L)` residual init scaling applies identically to both arms.

## Operational note

Cloning a repo inside a notebook that already had a *failed* clone attempt leaves
the directory on `sys.path` while empty. Python's `FileFinder` caches that empty
listing in `sys.path_importer_cache`, keyed by path string, and it survives
removing and re-adding the entry to `sys.path`. The files then exist and the
import still fails. Fix: `importlib.invalidate_caches()`.


## How the paper handles the full-precision share: it does not

Computed from BitNet b1.58 2B4T's own `config.json` (vocab 128,256, d 2,560,
30 layers, ffn_inner 6,912, tied embeddings):

```
              ternary body        full precision      packed    fp share
OURS  (11M)        1.9 MB              2.7 MB          4.6 MB     57.7%
BitNet (2B)      366.1 MB            657.6 MB       1023.7 MB     64.2%
```

**BitNet's own model is 64.2% full precision by packed bytes -- worse than ours.**
Independent validation that this arithmetic is right: the computed ternary body
is 366 MB, and the paper's published figure is "Memory (Non-emb) 0.4GB". The
658 MB embedding table is simply not in the headline number.

### Which lever moves the ratio

fp share is governed by `d*L/V`, and only `d` sits in the quadratic term -- the
body grows as `d^2*L` while the embedding grows only as `d`. Holding
V = 128,256:

| | d=1024 | d=2560 | d=4096 | d=8192 |
|---|---|---|---|---|
| L=8 | 94.4% | 87.1% | 80.8% | 67.8% |
| L=30 *(BitNet)* | 81.8% | **64.2%** | 52.9% | 35.9% |
| L=64 | 67.8% | 45.7% | 34.5% | 20.9% |

Depth helps linearly, **width helps quadratically**. So the 1-bit memory claim
strengthens automatically with scale, and an 11M-parameter model is inherently a
poor showcase for it.

### The distinction that makes "non-emb" defensible for one claim and not the other

**For the compute claim it is the right number.** The embedding is a gather, not
a matmul -- no multiplication happens. The body holds essentially all the
multiply-accumulates, and ternary genuinely removes the multiplier from them.

**For the memory claim it is the wrong number.** A 1 GB file is a 1 GB file.

BitNet's paper argues compute and energy, where the exclusion is legitimate. The
figure gets quoted as file size, where it is not.

One correction to that framing in our favour of accuracy: the tied LM head *is* a
real matmul (d x V per token) and stays full precision -- 12% of MACs for us,
15% for BitNet. So "no multipliers" is ~88%, not 100%.


## Quantizing the embedding table too (not a step the paper takes)

All variants evaluated on **identical fixed batches** (seed 1337, 60 x 32 x 256),
so differences are precision and not eval noise. Applied post-hoc to the trained
QAT-ternary model. `embed` and `head` are tied, so one change quantizes both the
input lookup and the output projection.

| embedding precision | val | delta | ppl | packed MB | fp share | compression |
|---|---|---|---|---|---|---|
| fp16 (baseline) | 2.1789 | +0.0000 | 8.8 | 4.61 | 57.7% | 4.85x |
| **int8** | **2.1795** | **+0.0006** | **8.8** | **3.28** | **40.6%** | **6.81x** |
| int4 | 2.3650 | +0.1861 | 10.6 | 2.61 | 25.4% | 8.54x |
| ternary | 2.7385 | +0.5596 | 15.5 | 2.21 | 11.9% | 10.10x |

### int8 embeddings are free

+0.0006 nats is 0.03% -- noise. Packed size drops 4.61 -> 3.28 MB, the
full-precision share falls 57.7% -> 40.6%, and compression against fp16 goes
4.85x -> 6.81x. No training, no measurable quality cost.

### The embedding is more precision-sensitive per parameter than the body

```
ternarizing 9,830,400 body weights  : +0.1207 nats
int4 on 1,328,960 embedding weights : +0.1861 nats
```

Seven times fewer weights, a much milder quantization, and a *larger* penalty.
This is the quantitative reason the embedding gets excluded -- not a convention,
a measured asymmetry.

### Correction: ternary embeddings degrade, they do not collapse

The claim "ternarize the embedding and the model collapses into noise" is
commonly stated (including repeatedly in this project's own earlier notes) and is
**too strong**. Measured: +0.5596 nats, ppl 8.8 -> 15.5. That is 76% worse
perplexity and 4.6x the cost of int8 -- bad, but for scale, PTQ on the body gave
ppl 151.9 and the unigram floor is 419. A ternary-embedding model still works.

The rule is real. Its usual statement is not.

### Open follow-up

The above is **PTQ on the embedding** -- quantizing a table that was trained in
fp32. On the body, PTQ vs QAT was worth 2.85 nats. So a model *trained* with
ternary embeddings could plausibly recover most of that 0.5596. If it does:
fp share 11.9%, packed 2.21 MB, 10.1x compression against fp16. Untested.


## Quantization-aware training on the embedding table (beyond the paper)

BitNet holds the embedding at bf16 throughout. Training it as ternary through the
same STE:

```
ternary body, ternary embed (PTQ)  2.7385   ppl 15.5
ternary body, ternary embed (QAT)  2.3107   ppl 10.1     913s
  -> QAT recovered 76.0% of the 0.5596 nats PTQ cost
```

Cost of a ternary embedding, once trained: **+0.1347 nats** over the fp16-embed
baseline of 2.1760. For that, the packed model goes 4.61 -> 2.21 MB and the
full-precision share collapses from 57.7% to **11.9%**.

### QAT recovers most of what PTQ destroys, wherever it is applied

```
body       PTQ 5.0229 -> QAT 2.1760   (vs fp32 2.0553)    95.9% recovered
embedding  PTQ 2.7385 -> QAT 2.3107   (vs      2.1760)    76.0% recovered
```

This is the paper's central claim generalized to a component the paper never
quantizes. It is also what makes the model genuinely 1-bit rather than
mostly-fp16: 88.1% of the 2.21 MB file is ternary.

## The Pareto frontier

| configuration | val | ppl | packed MB | fp share | vs fp16 |
|---|---|---|---|---|---|
| fp32 body, fp16 embed | 2.0553 | 7.8 | 22.32 | — | 1.00x |
| ternary body, fp16 embed | 2.1760 | 8.8 | 4.61 | 57.7% | 4.85x |
| **ternary body, int8 embed (PTQ)** | **2.1795** | **8.8** | **3.28** | **40.6%** | **6.81x** |
| ternary body, ternary embed (QAT) | 2.3107 | 10.1 | 2.21 | 11.9% | 10.09x |

**10.1x smaller than fp16 for a total cost of 0.2554 nats.**

The int8-embedding row is the practical choice: 6.81x compression for +0.0006
nats, which is noise. The ternary-embedding row trades 0.1347 nats for another
32% off the file and a genuinely 1-bit weight format.

## Ternary wins at every memory budget tested

fp16 alternatives sized to the same bytes, loss estimated by log-linear
interpolation between the two measured fp16 points (d=128 at 2.3607 and d=320 at
2.0553):

| budget | fp16 params it buys | fp16 estimated | ternary measured | margin |
|---|---|---|---|---|
| 4.61 MB | 2,305,000 | 2.3440 | **2.1760** | **+0.168** |
| 3.28 MB | 1,640,000 | 2.4063 | **2.1795** | **+0.227** |
| 2.21 MB | 1,105,000 | 2.4786 | **2.3107** | **+0.168** |

A consistent 0.17-0.23 nat advantage across a 2x range of budgets, not a single
fortunate operating point.

## A floating-point detail that matters for Phase 4

`ste(w, q)` returns `w + (q - w).detach()`, which equals `q` in exact arithmetic
but **not** in fp32: `q - w` rounds, then adding `w` rounds again. Counting
unique values on the STE output of a ternary tensor gives **7**, not 3 -- roughly
three adjacent floats near `+g`, three near `-g`, and exactly zero (IEEE-754
guarantees `w + (-w) == 0`).

Relative deviation is ~1e-9, irrelevant to training. But:

- do not test "is it quantized" by counting unique values on the STE output
- **when packing weights for inference, pack from `quantize_weight(w)` directly,
  never from the STE output**


## CORRECTION to the precision accounting above

The Pareto table bucketed 1,328,960 params as "full precision" -- that is the
embedding (1,310,720) **plus** the norms (18,240). For the ternary-embed row the
whole bucket was then quantized while the column kept its label, so the reported
"11.9% fp share" is actually the **embedding's share of the packed file**, not the
full-precision share. Three buckets, correctly separated:

| configuration | MB | at 1.58 bits | at 8 bits | at fp16 | vs fp16 |
|---|---|---|---|---|---|
| all fp16 | 22.32 | 0% | 0% | 100% | 1.00x |
| ternary body, fp16 embed | 4.61 | 42.3% | 0% | 57.7% | 4.85x |
| ternary body, int8 embed | 3.29 | 59.1% | 39.8% | **1.11%** | 6.77x |
| **ternary body, ternary embed** | **2.24** | **98.4%** | 0% | **1.63%** | **9.95x** |

```
ternary-embed model by parameter count:
  ternary       11,141,120   99.84%
  fp16 (norms)      18,240    0.16%    <- 41 RMSNorm vectors, 36 KB
```

**98.4% of the packed file is 1.58-bit weights.** The only full-precision
parameters remaining are the normalization vectors.

For comparison, **BitNet b1.58 2B4T is 64.2% fp16 by packed bytes**, because it
holds a 128,256 x 2,560 embedding table at bf16. Quantizing the embedding is what
moves a model from "mostly fp16 with a ternary body" to "actually 1-bit".

## Honest assessment against the BitNet spec

Architecture fidelity is complete: ternary per-tensor absmean weights, per-token
absmax int8 activations, SubLN, squared ReLU, RoPE, no biases, tied embeddings,
trained quantized from scratch. The embedding quantization goes beyond the paper.

**But this is a training-time demonstration, not an inference artifact.** Six gaps:

1. **Weights are not packed.** 2.24 MB is arithmetic. The file on disk is 44.7 MB
   of fp32 master weights. Nothing has been written at 1.58 bits.
2. **No integer inference path.** The forward pass runs fp32 matmuls on tensors
   that merely *contain* ternary values. No add/subtract/skip exists. BitNet's
   central claim -- eliminating the multiplier -- is unrealized.
3. **Under-trained**: 1.8 tokens/param vs Chinchilla ~20.
4. **Hard scale regime**: 11M vs 2B. The memory advantage grows as d^2 while the
   embedding grows as d, so small models are the worst case for this claim. The
   equal-memory win held anyway.
5. **No standard benchmarks.** Val loss on TinyStories only.
6. **act_bits=8 never ablated.** int8 activations assumed to matter, not tested.

Gaps 1 and 2 are the same gap, and they are Phase 4.
