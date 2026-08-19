---
license: mit
language: en
library_name: bitllm
tags:
  - bitnet
  - 1-bit
  - ternary
  - quantization-aware-training
  - tinystories
  - from-scratch
datasets:
  - euclidstellar/tinystories-bpe4096
pipeline_tag: text-generation
---

# TinyStories 1-bit LLM

An **11,159,360-parameter BitNet b1.58 language model built from scratch** — no
`nn.Transformer`, no pretrained weights, no fine-tuning. Trained on a free Kaggle
T4.

The deliverable is **`model_packed.bin`, 2,313,205 bytes**, in which 11,141,120
weights are stored at **1.6 bits each**.

```
11,141,120 ternary weights  @ 1.6 bits/weight   (99.1% of the log2(3) floor)
    18,240 fp32 norm params @ 32 bits          (0.79% of the file)
-------------------------------------------------------------
 2,313,205 bytes    9.65x smaller than the same model at fp16
```

## Results

| model | val loss | ppl | size |
|---|---|---|---|
| uniform baseline (knows nothing) | 8.3178 | 4096 | — |
| unigram baseline (counting only) | 6.0380 | 419 | — |
| **PTQ-ternary** (quantized *after* training) | **5.0229** | **151.9** | — |
| fp16 at equal memory (d=128, 2.1M params) | 2.3607 | 10.6 | 4.21 MB |
| **QAT-ternary + ternary embedding** ← this model | **2.3107** | **10.1** | **2.31 MB** |
| QAT-ternary, fp16 embedding | 2.1760 | 8.8 | 4.61 MB |
| fp32 control | 2.0553 | 7.8 | 22.32 MB |

All arms: identical architecture, data, hyperparameters, and seed. The only
variable is quantization.

### Quantization-aware training is worth 2.85 nats over post-training quantization

`QAT-ternary` (2.1760) and `PTQ-ternary` (5.0229) have **identical forward
passes** — both multiply by ternary weights. The only difference is whether the
training loop knew it. That is worth 2.85 nats and a 17x perplexity gap.

For scale: post-quantized, the full 8-layer model (5.0229) scores **worse than a
single full-precision attention layer** (3.9106) from the ablation ladder.

### Ternary loses at equal parameters and wins at equal memory

| comparison | result |
|---|---|
| equal parameter count | ternary **loses** 0.1207 nats |
| equal memory budget | ternary **wins** 0.17–0.23 nats, across a 2x range of budgets |

2.31 MB of ternary buys 11.2M parameters; 2.31 MB of fp16 buys 1.1M. Reporting
only one of these comparisons misleads in either direction.

### Quantizing the embedding table (beyond the BitNet papers)

BitNet holds embeddings at bf16. Training the tied embedding as ternary through
the same STE:

```
embedding at fp16     2.1789 nats   4.61 MB   57.7% of the file is full precision
embedding at int8     2.1795        3.28 MB   1.11%      (+0.0006 nats — free)
embedding at ternary  2.3107        2.31 MB   0.79%      (+0.1347 nats)
```

QAT on the embedding recovered **76%** of what post-hoc quantization cost
(0.5596 → 0.1347 nats). In blind reading, the ternary-embedding output is not
reliably distinguishable from the fp16-embedding output.

## Limitations — read these

**It is not a good model.** It is a rigorous demonstration.

- **11x under-trained.** 20M tokens on 11.2M parameters = 1.79 tokens/parameter
  against Chinchilla's ~20. The train/val gap is 0.011 — no overfitting at all,
  meaning the model never even fit its training data.
- **Entity tracking fails in every arm, including fp32.** Characters change name
  mid-story, referents mutate between clauses, objects become other objects.
  Quantization is not the cause; training budget is.
- **TinyStories domain only.** Simple narrative English, ~4k vocabulary. It knows
  no facts, cannot answer questions, cannot hold a conversation, and will produce
  confident nonsense outside children's-story distribution.
- **Slower, not faster.** On a T4 in PyTorch this is **1.7x slower** than fp32
  inference. See below.
- **No standard benchmarks.** Validation loss on held-out TinyStories only.

## The speed result, honestly

| configuration | tok/s | ms/forward | vs fp32 |
|---|---|---|---|
| fp32 weights, no quantization | 91,926 | 89.1 | 1.00x |
| ternary weights, QAT forward | 53,500 | 153.1 | **1.72x slower** |
| packed weights, activation quant only | 54,217 | 151.1 | **1.70x slower** |

Decomposing the 64 ms of overhead:

```
activation quantization   62.0 ms    96.9%
weight quantization        2.0 ms     3.1%
```

The packed model skips weight quantization **entirely** and gains 1.3%. Weights
were never the cost — per-token absmax activation quantization is, because it
reduces over tensors 19x larger than the weights at all 48 BitLinear layers.

A GPU has thousands of hardware multipliers idle either way, so "we eliminated
the multiplier" buys nothing on silicon designed around multipliers. This is
exactly why `bitnet.cpp` exists. **Memory is the win available today; compute
needs custom kernels or custom hardware.**

## A note on how "1-bit" memory figures are reported

BitNet b1.58 2B4T's published efficiency figure is **"Memory (Non-emb) 0.4GB"**.
Computing from their own `config.json` (`vocab_size` 128256, `hidden_size` 2560,
30 layers, `tie_word_embeddings` true):

```
ternary body     1,848,115,200 params -> 366.1 MB   <- reproduces their 0.4GB
full precision     328,775,680 params -> 657.6 MB   <- excluded from the headline
packed total                             1023.7 MB
full-precision share of the file:           64.2%
```

The figure is accurate and the exclusion is labelled. It is also **the right
number for a compute claim and the wrong number for a file-size claim** — the
embedding is a gather, not a matmul, so it carries no multiplications, but a
1 GB file is still a 1 GB file.

This model's equivalent share is **0.79%**, because the embedding is quantized.

## Architecture

Follows [BitNet b1.58 2B4T](https://arxiv.org/abs/2504.12285):

| | |
|---|---|
| weights | ternary `{-1,0,+1}`, per-tensor absmean |
| activations | int8, per-token absmax (**W1.58A8**) |
| normalization | RMSNorm + SubLN before each sublayer's output projection |
| FFN | squared ReLU (not SwiGLU) |
| positions | RoPE |
| biases | none, anywhere |
| embeddings | tied, and ternary (beyond the paper) |
| training | quantized from scratch, straight-through estimator |
| dims | 320 embd, 8 layers, 8 heads, 1280 FFN, 256 context, 4096 vocab |

## Usage

```bash
pip install git+https://github.com/EuclidStellar/1-bit-LLM.git
```

```python
import torch
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from bitllm import load_packed_model

path = hf_hub_download("euclidstellar/tinystories-1bit-llm", "model_packed.bin")
model, header = load_packed_model(path)          # 2.31 MB
tok = Tokenizer.from_file(hf_hub_download(
    "euclidstellar/tinystories-bpe4096", "tokenizer.json", repo_type="dataset"))

ids = torch.tensor([tok.encode("Once upon a time").ids])
out = model.generate(ids, max_new=150, temperature=0.8, top_k=100)
print(tok.decode(out[0].tolist()))
```

`load_packed_model` builds a model with `weight_mode="none"` — the packed weights
already *are* the quantized weights, so re-quantizing them would recompute the
absmean scale as `g*(1 - zero_fraction) ≈ 0.686g` and shrink every weight 31%.

## Files

| file | size | what |
|---|---|---|
| `model_packed.bin` | 2.31 MB | **the 1-bit model.** Inference only |
| `qat_ternary_embed.pt` | 44.7 MB | fp32 master weights, resumable |
| `qat_ternary.pt` | 44.7 MB | ternary body, fp16 embedding |
| `rung7_fp32.pt` | 44.7 MB | fp32 control. Load into a ternary model for the PTQ arm |
| `fp16_d128.pt` | 8.45 MB | equal-memory comparison model |
| `results.json` | 100 kB | every number, plus the exact training recipe and seed |

Checkpoints store **fp32 master weights** because ternary weights cannot be
updated — an optimizer step of 1e-5 on a value that is exactly -1, 0 or +1 does
nothing. The fp32 master accumulates gradient until a weight crosses a rounding
boundary and flips state.

## Training

```
data       TinyStories, own 4096-token byte-level BPE, 477,236,558 tokens
budget     19,996,672 tokens  (2,441 steps x batch 32 x context 256)
optimizer  AdamW(0.9, 0.95), wd 0.1, grad clip 1.0
schedule   100-step linear warmup, cosine decay to 10%
lr         1e-3, identical across all arms
seed       1337
hardware   one Kaggle T4, 913 seconds
```

Fidelity of the packed file: loss 2.315797 against the source model's 2.315795,
and a max logit deviation of 1.144e-05 with activation quantization disabled —
fp32 accumulation noise.

## Reproducing

Code, all findings, and per-phase notes:
[github.com/EuclidStellar/1-bit-LLM](https://github.com/EuclidStellar/1-bit-LLM)

The `notes/` directory documents the full build: a 7-rung ablation ladder that
attaches a measured value to every architecture component, and the bugs found
along the way — including one line of gradient clipping that was the difference
between 8.0065 and 0.0144 on a probe task.

## Citation

Architecture from Ma et al., *The Era of 1-bit LLMs*, and the
[BitNet b1.58 2B4T technical report](https://arxiv.org/abs/2504.12285).
Data from Eldan & Li, *TinyStories*.
