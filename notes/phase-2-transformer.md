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
| 3 | + 1 attention layer | q,k,v,o + causal mask | _todo_ | |
| 4 | + FFN | up, ReLU², down | _todo_ | |
| 5 | + RoPE | rotary positions | _todo_ | |
| 6 | 8 layers | depth, pre-norm, residuals | _todo_ | |

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

