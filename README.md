# 1-bit LLM, from scratch

An ~11M-parameter BitNet b1.58 language model, built from scratch to learn how
LLMs and ternary quantization actually work. Trained on TinyStories, on a free
Kaggle T4.

Architecture follows the BitNet b1.58 2B4T report ([arXiv:2504.12285](https://arxiv.org/abs/2504.12285)):
**W1.58A8** — ternary weights via per-tensor absmean, int8 activations via
per-token absmax — with subln normalization, squared ReLU, RoPE, no bias terms,
and tied embeddings. Trained quantized from scratch, not post-quantized.

## The experiment

Three arms, identical in every respect except how weights are quantized:

| arm | what it is | cost |
|-----|-----------|------|
| `fp32` | control. proves the task is learnable at this size | ~42 min on T4 |
| `ternary` | QAT — quantized in the forward pass, trained through it | ~42 min on T4 |
| `ptq` | fp32 weights ternarized *afterwards* | free, no training |

The third arm collapses. Putting that collapse beside a working QAT model is the
whole argument for training quantized from scratch.

## Config

| | |
|---|---|
| params | 11,159,360 (9,830,400 ternary / 1,328,960 full precision) |
| dims | 320 embd, 8 layers, 8 heads, 1280 FFN, 512 context |
| vocab | 4,096 BPE trained on TinyStories |
| tokens | 100M (21% of TinyStories, each seen once) — 9.0 tokens/param |

## Layout

```
bitllm/
  data.py     Phase 1 — TinyStories -> BPE -> flat uint16 token stream
  train.py    Phase 5 — training loop, checkpointing, resume
  sample.py   Phase 7 — generate text, inspect weights
  viz.py      Phase 7 — print ternary weights as -/./+ , memory report
reference/
  model.py    answer key for Phase 2 (the transformer)
  quant.py    answer key for Phase 3 (BitLinear + the STE)
```

`bitllm/model.py` and `bitllm/quant.py` are **deliberately absent** — writing
them is Phases 2 and 3. `reference/` holds worked versions to diff against
afterwards. `train.py` will not import until you've written them.

## Notes worth keeping

**`torch.cuda.is_bf16_supported()` lies on a T4.** It defaults to
`including_emulation=True`, so a Turing card (SM 7.5) reports `True` despite
having no bf16 tensor-core path. Ask for `including_emulation=False` and use
fp16 + GradScaler instead.

**Vocabulary size is a now-decision.** It sets the embedding table's shape,
which is baked into the checkpoint. 4,096 keeps the un-quantizable share at
11.9%; BitNet's own 128,256-token table is *larger* than its entire ternary
body, which is why their headline memory figure is labelled "Memory (Non-emb)".
