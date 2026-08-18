# Resume here

Last session: 2026-08-19. **Phase 1 and Phase 2 complete. Phase 3 just started.**

## What is already safe (nothing to redo)

| what | where |
|---|---|
| tokenizer + 477M token stream | HF `euclidstellar/tinystories-bpe4096` |
| rung 7 fp32 checkpoint (44.7 MB) | HF `euclidstellar/tinystories-1bit-llm` |
| local 20M-token slice | `data/` in this repo |
| all findings and numbers | `notes/phase-1-tokenizer.md`, `notes/phase-2-transformer.md` |

## The ladder, final

| rung | model | val loss | ppl | params added | nats/M params |
|---|---|---|---|---|---|
| 0 | uniform | 8.3178 | 4096 | — | — |
| 1 | unigram | 6.0380 | 419 | — | — |
| 2 | embedding + tied head | 5.3828 | 218 | 1,310,720 | 0.50 |
| 3 | + attention | 3.9106 | 49.9 | 409,600 | **3.59** |
| 4 | + FFN | 3.7150 | 41.1 | 819,200 | 0.24 |
| 5 | + RoPE | 3.2948 | 27.0 | **0** | infinite |
| 6 | + norms | 2.4304 | 11.4 | 2,560 | **337.7** |
| 7 | 8 layers | **2.0651** | **7.9** | 8,617,280 | 0.042 |

Rung 7 = 11,159,360 params exactly. Causality max delta 0.00e+00.
0.737 bits/char. Generates coherent children's prose; fails on entity tracking.

## Next step: Phase 3, the BitLinear swap

Already established and verified:

- `weight_ternary` produces exactly 3 values; zero fraction 31.05%, which matches
  the closed-form prediction `2*Phi(0.5*sqrt(2/pi)) - 1 = 0.3101` for Gaussian weights
- STE gradient with `.detach()` = 1.0; without it = 7.12e-06 (**140,000x less**).
  The leak is through the differentiable absmean *scale*, not through `round()`
- `BitLinear` class written

**Immediately next:** swap `nn.Linear` -> `BitLinear` for `q, k, v, o, up, down`
in `Block`. Six constructor lines. Embedding, LM head and all norms stay fp32.
Target split: **9,830,400 ternary (88.1%) / 1,328,960 full precision (11.9%)**.

Then: train the ternary arm on the same 20M tokens with the same harness and
compare against 2.0651.

## To get back to a working Kaggle session tomorrow

Fresh session loses all Python state. Re-run, in order:

1. **Cell 1** — setup (pulls data from HF, defines `get_batch`, `dev`)
2. The class definitions: `Rung2`, `Rung3`, `Rung4`, `Rung5`, `RMSNorm`, `Rung6`,
   `Block`, `Rung7`
3. **Cell 4** — the shared `run()` / `evaluate()` harness
4. Cell 17/18 — quant primitives and `BitLinear`

No need to retrain rungs 2-6; their numbers are in the table above. To get rung 7
back without an 820s retrain, load the checkpoint from HF:

```python
from huggingface_hub import hf_hub_download
import torch
p = hf_hub_download("euclidstellar/tinystories-1bit-llm", "rung7_fp32.pt")
ck = torch.load(p, map_location=dev, weights_only=False)
m7 = Rung7(**{k: v for k, v in ck["config"].items() if k != "maxT"}).to(dev)
m7.load_state_dict(ck["model"])
vl7 = ck["val_loss"]
```

## Open threads

- **Phase 3 demolition list**: remove `.detach()`; ternarize the embedding (should
  collapse); delete SubLN (norms scored 337.7 nats/M, so stakes are real);
  ternary vs binary; absmean vs absmax; activations int8 -> int4 -> int2
- **Prediction to check**: depth contributed only 0.042 nats/M at 1.79 tokens/param.
  At the real run's 100M tokens (9 tokens/param) its share should grow. The current
  ladder is 11x under-trained with zero overfitting gap.
- Rung classes still live only in the Kaggle notebook -- worth copying into
  `bitllm/` so they are versioned.
