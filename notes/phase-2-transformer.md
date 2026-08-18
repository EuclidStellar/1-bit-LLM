# Phase 2 — The transformer, built as seven models

Method: do not write the finished architecture and then test it. Build seven
working language models, each adding exactly one component, each retrained. Every
addition then has to justify itself against the previous rung's loss.

Measured on the 20M-token local slice (`data/train-20m.bin`), vocab 4,096.

## Baselines before any model exists

| baseline | cross-entropy | effective choices | params |
|---|---|---|---|
| **rung 0** — uniform over 4,096 | **8.3178** | 4,096.0 | 0 |
| **rung 1** — unigram frequencies | **6.0380** | 419.0 | 0 |
| full bigram table (reference only) | 3.6140 | 37.1 | 16,777,216 |

`ln(4096) = 8.3178` exactly. Knowing only how often each token appears — no
context, no model — cuts effective choices from 4,096 to 419. **About 90% of the
guessing is eliminated by counting alone.** Any model that does not beat 6.0380 has
learned nothing a `bincount` could not.

The bigram figure is an **optimistic** bound: it is the conditional entropy of the
empirical distribution measured on the same tokens it was counted from, and only
504,896 of 16,777,216 possible bigrams (3.0%) were ever observed. A real bigram
model would score worse on held-out text.

## Why the bigram number is the right target for rung 2

Rung 2 is embedding + tied LM head with no blocks at all. Its prediction for
"what follows token *i*" depends only on token *i*'s own embedding vector — so it
is a **bigram model expressed through a rank-320 bottleneck**:

```
true bigram table :  4096 x 4096  =  16,777,216 params   exact P(next|current)
rung 2            :  4096 x  320  =   1,310,720 params   12.8x smaller
```

Rung 2 must land between **6.0380** and **3.6140**. Where it lands measures how much
bigram structure survives a 320-dimensional factorization.

## Rung results

| # | model | new component | cross-entropy | vs previous |
|---|---|---|---|---|
| 0 | uniform | — | 8.3178 | — |
| 1 | unigram | token frequencies | 6.0380 | −2.2798 |
| 2 | embedding + tied head | lookup only | **5.3828** | −0.6552 |
| 3 | + 1 attention layer | q,k,v,o + causal mask | **3.9106** | −1.4722 |
| 4 | + FFN | up, ReLU², down | **3.7150** | −0.1956 |
| 5 | + RoPE | rotary positions | **3.2948** | −0.4202 |
| 6 | + norms | RMSNorm pre-norm + SubLN | _todo_ | |
| 7 | 8 layers | depth | _todo_ | |

## Shared harness

Identical for every rung — the model is the only variable.

```
20M tokens (2,441 steps x batch 32 x ctx 256)
AdamW lr 1e-3, betas (0.9, 0.95), wd 0.1, grad clip 1.0
100-step linear warmup then cosine decay to 10%
seed 1337, val loss averaged over 60 batches of held-out val.bin
```

`lr` is fixed across rungs deliberately. Each rung's number is therefore
"under shared hyperparameters", not "at its own optimum" — tuning per rung
would make the ladder a comparison of tuning rather than of architecture.

## Findings

### Rung 2 — a rank-320 factorization captures ~27% of bigram structure

```
val loss 5.3828   ppl 217.6   1,310,720 params   54s on a T4
```

| | cross-entropy | effective choices |
|---|---|---|
| unigram | 6.0380 | 419 |
| **rung 2** | **5.3828** | **218** |
| full bigram table | 3.6140 | 37 |

```
available gain, unigram -> bigram :  6.0380 - 3.6140 = 2.4240 nats
rung 2 captured                   :  6.0380 - 5.3828 = 0.6552 nats  = 27%
```

A 4096x4096 bigram table (16.8M params) squeezed into 4096x320 (1.31M params,
12.8x smaller) retains roughly a quarter of the conditional information. The
bigram bound is optimistic — measured on training data with only 3.0% of
possible bigrams observed — so 27% is a floor.

### Rung 2 is architecture-limited, not data-limited

```
final train loss 5.3708
val loss         5.3828      gap 0.012
```

Essentially no overfitting at 15 tokens/param. The model has extracted
everything a context-free predictor can and is bounded by what it *is*, not by
how much it has seen. This is the correct state to be in before adding
attention: any improvement at rung 3 is attributable to capability, not to
more data.



### The parameter economics are wildly uneven

| rung | added | params added | nats bought | **nats per million params** |
|---|---|---|---|---|
| 2 | embedding | 1,310,720 | 0.6552 | 0.50 |
| 3 | attention | 409,600 | 1.4722 | **3.59** |
| 4 | FFN | 819,200 | 0.1956 | 0.24 |
| 5 | RoPE | **0** | 0.4202 | **infinite** |

Share of all model gain (measured from the unigram floor of 6.0380):

```
attention  53.7%     embedding  23.9%     RoPE  15.3%     FFN  7.1%
```

**Attention is 63% of the gain for 16% of the parameters.** It is ~15x more
parameter-efficient than the FFN at depth 1. **RoPE delivered more than twice
the FFN's gain for zero parameters** and only 4% more wall clock (145s vs 139s).

Caveat before generalizing: this is one layer, so the FFN has nothing composed
to process yet, and there is no normalization — ReLU squared is unusually
scale-sensitive because squaring amplifies. Both handicaps are tested at
rungs 6 and 7.

Rung 5 at 3.2948 is the first rung below the bigram bound of 3.6140.

### Position encoding, isolated with a synthetic task

Task: `target[t] = input[t-2]`, alphabet of 8, iid tokens. The current token
carries **zero** information about the target, so a context-free model is pinned
at `ln(8) = 2.0794` and memorization is impossible.

| rung | context available | loss | effective choices | trajectory |
|---|---|---|---|---|
| 2 | none | 2.0665 | 7.9 | flat at 2.07 from step 100 |
| 3 | unordered | 1.5458 | 4.7 | 2.14 -> 1.55, grinding |
| 5 | **ordered** | **0.0144** | **1.01** | 0.64 -> 0.03 -> 0.01, solved |

The rung 3 result corrected a wrong prediction. I expected attention without
position encoding to stay at the floor; it beat it by 0.53 nats. **The bag of
words is itself informative** — the target token is a *member* of the prefix
multiset, so knowing which tokens appear (and how often) shifts the posterior
even without knowing where any of them are.

### RoPE's relative-position property, verified

Same content vectors, same offset of 3, absolute positions spanning 100x:

```
q at   5, k at   2:  -9.889071
q at  50, k at  47:  -9.889072
q at 200, k at 197:  -9.889067
q at 500, k at 497:  -9.889047
```

Identical to six significant figures. The drift in the final digits is fp32
rotation error accumulating with position.

### One line of gradient clipping was the difference between 8.0065 and 0.0144

My side-test helper omitted `clip_grad_norm_` while the main harness had it.
Rung 5 on the copy task:

```
without clipping, lr 1e-2:  8.0065     <- ln(4096) = 8.3178, total collapse
with    clipping, lr 1e-2:  0.0144     <- solved
```

Same learning rate. Same everything else. The lr sweep proves it was not the
learning rate — rung 5 gives 0.0144 at 1e-2, 3e-3, and 1e-3 alike once clipped.

Which rungs collapsed is diagnostic: **rung 3 (no FFN) survived; rung 5 (ReLU²
FFN, no norms) exploded.** Squaring amplifies, and nothing was holding the
residual stream's scale. Direct evidence for the normalization hypothesis before
rung 6 was even run.

Two harness defects, both mine:
1. no gradient clipping in the side-test helper
2. returning only the final loss, which makes divergence indistinguishable from
   "learned poorly"

Fix for (2): always print the loss trajectory. A single number hides explosions.

Contrast the two failure modes the lr sweep separates:

```
rung 3:  1.5793 / 1.5458 / 1.6046  across 10x lr   -> CAPABILITY-limited
rung 5:  8.0065 without clipping, 0.0144 with      -> OPTIMIZATION-limited
```
