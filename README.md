# A 1-bit LLM, from scratch

An **11,159,360-parameter BitNet b1.58 language model** built from nothing — no
`nn.Transformer`, no pretrained weights, no tutorial followed. Trained on one free
Kaggle T4 in about 15 minutes.

The deliverable is **2,313,205 bytes** in which 11,141,120 weights are stored at
**1.6 bits each**. It writes children's stories, badly, in a browser tab, at
**~1,045 tokens a second** with no server.

```
try it     https://gaurav.bar/things/1-bit-llm
weights    https://huggingface.co/euclidstellar/tinystories-1bit-llm
```

This repo is the whole journey, including the parts that were wrong. If you only
read one file, read [`notes/mistakes.md`](notes/mistakes.md).

**The whole thing written up as one post, in plain language** — what ternary
weights cost, what they buy, and where I was wrong:
[`blog/1-bit-llm-from-scratch.md`](blog/1-bit-llm-from-scratch.md)

---

## Results

| model | val loss | ppl | size |
|---|---|---|---|
| uniform baseline (knows nothing) | 8.3178 | 4096 | — |
| unigram baseline (counting only) | 6.0380 | 419 | — |
| **PTQ-ternary** (quantized *after* training) | **5.0229** | **151.9** | — |
| fp16 at equal memory (d=128, 2.1M params) | 2.3607 | 10.6 | 4.21 MB |
| **QAT-ternary + ternary embedding** ← ships | **2.3107** | **10.1** | **2.31 MB** |
| QAT-ternary, fp16 embedding | 2.1760 | 8.8 | 4.61 MB |
| fp32 control | 2.0553 | 7.8 | 22.32 MB |

Every arm: identical architecture, data, hyperparameters and seed. The only free
variable is quantization.

**Three findings worth the trouble:**

- **Quantization-aware training beats post-hoc by 2.85 nats.** Same weights, same
  forward pass — the only difference is *when* the quantizer runs. A post-quantized
  8-layer transformer scores worse than a single full-precision attention layer.
- **Ternary loses at equal parameters and wins at equal memory.** −0.12 nats one
  way, +0.17 the other, because 2.31 MB buys 11.1M ternary parameters or 1.1M fp16
  ones. Reporting either alone misleads.
- **BitNet's own 2B model is 64.2% full precision by packed bytes.** Computed from
  their published `config.json`; it reproduces their "Memory (Non-emb) 0.4GB"
  figure exactly. The excluded embedding table is 657.6 MB — larger than the
  400 MB ternary body beside it. This model's equivalent share is **0.79%**,
  because the embedding is quantized too, which the papers do not do.

---

## The journey, in order

| # | what | notes | code |
|---|---|---|---|
| 1 | tokenizer and the token stream | [`phase-1-tokenizer.md`](notes/phase-1-tokenizer.md) | [`01_data_kaggle.md`](notebooks/01_data_kaggle.md), [`bitllm/data.py`](bitllm/data.py) |
| 2 | the transformer, built **seven times** | [`phase-2-transformer.md`](notes/phase-2-transformer.md) | [`00_ladder_kaggle.py`](notebooks/00_ladder_kaggle.py) |
| 3 | making it 1-bit | [`phase-3-bitlinear.md`](notes/phase-3-bitlinear.md) | [`bitllm/quant.py`](bitllm/quant.py), [`bitllm/model.py`](bitllm/model.py) |
| 3b | reading what it writes | [`phase-3-samples.md`](notes/phase-3-samples.md) | [`bitllm/sample.py`](bitllm/sample.py) |
| 4 | packing to 2.31 MB | [`phase-4-packing.md`](notes/phase-4-packing.md) | [`bitllm/pack.py`](bitllm/pack.py) |
| 5 | 20 → 1,045 tok/s in a browser | [`phase-5-browser.md`](notes/phase-5-browser.md) | [`web/`](web/), [`wasm/kernel.c`](wasm/kernel.c) |
| — | **everything that went wrong** | [`mistakes.md`](notes/mistakes.md) | |

**The training run itself:** [`notebooks/02_train_kaggle.py`](notebooks/02_train_kaggle.py)
— every published number came from these cells, with the measured loss inline
against each arm.

### Phase 2 is the part I would show someone first

Reading a finished transformer teaches you very little; correct code looks
inevitable. So it was built seven times, each version one component bigger, each
retrained on an identical budget:

| rung | added | val loss | params added | **nats per million params** |
|---|---|---|---|---|
| 0 | uniform guess | 8.3178 | 0 | — |
| 1 | unigram frequencies | 6.0380 | 0 | — |
| 2 | embedding + tied head | 5.3828 | 1,310,720 | 0.50 |
| 3 | + one attention layer | 3.9106 | 409,600 | **3.59** |
| 4 | + FFN (squared ReLU) | 3.7150 | 819,200 | 0.24 |
| 5 | + RoPE | 3.2948 | **0** | **infinite** |
| 6 | + norms (RMSNorm + SubLN) | 2.4304 | 2,560 | **337.7** |
| 7 | 8 layers | 2.0651 | 8,617,280 | 0.042 |

```
attention + norms + RoPE  =    412,160 params ( 3.7%)  ->  69.5% of all gain
FFN + depth               =  9,436,480 params (84.6%)  ->  14.1% of all gain
```

Honest caveat: 1.79 tokens/param is ~11× under-trained with a train/val gap of
0.014, so the model never even fit its data. Depth and FFN capacity are exactly
what needs data. This ranks components **at this budget**, not in general.

---

## Decoder-only — there is no encoder and no cross-attention

A question that has come up, so: this is a **decoder-only** transformer, like GPT
and LLaMA. `LM` in [`bitllm/model.py`](bitllm/model.py) is the whole model —
embedding, eight identical blocks, final norm, tied head. There is no encoder
stack, no decoder stack, and no cross-attention anywhere.

The tell is in `Block.attn()`:

```python
q = self.q(x)     # Q from x
k = self.k(x)     # K from x  <- the same x
v = self.v(x)     # V from x  <- the same x
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

Q, K and V all come from one tensor. That is self-attention. Cross-attention
takes two inputs — Q from the decoder, K and V from the encoder's output — and it
exists so a decoder can look at something an encoder produced. There is nothing
here to look at: the task is next-token prediction over a single continuous
stream. Translation needs cross-attention because the output sequence must attend
to a different input sequence. Story continuation does not.

`is_causal=True` does the work instead: position *t* may only see positions ≤ *t*.
Remove that one flag and loss collapses toward zero, because the model is reading
the answer.

| family | attention | example |
|---|---|---|
| encoder-only | bidirectional self-attn, no mask | BERT |
| encoder-decoder | self-attn + **cross-attention** | original Transformer, T5 |
| **decoder-only** | causal self-attn only | GPT, LLaMA, BitNet, **this** |

To read attention appearing from nothing, see
[`notebooks/00_ladder_kaggle.py`](notebooks/00_ladder_kaggle.py) — rung 2 is the
model *without* it, rung 3 adds it by hand, and the loss difference between them
is 1.4722 nats.

## Architecture

Follows the [BitNet b1.58 2B4T report](https://arxiv.org/abs/2504.12285):

| | |
|---|---|
| weights | ternary `{-1,0,+1}`, per-tensor absmean |
| activations | int8, per-token absmax (**W1.58A8**) |
| normalization | RMSNorm + SubLN before each sublayer's output projection |
| FFN | squared ReLU, not SwiGLU |
| positions | RoPE |
| biases | none, anywhere |
| embeddings | tied, **and ternary** — beyond the paper |
| training | quantized from scratch, straight-through estimator |
| dims | 320 embd · 8 layers · 8 heads · 1280 FFN · 256 ctx · 4096 vocab |

```
11,141,120 ternary weights  @ 1.6 bits   (99.1% of the log2(3) floor)
    18,240 fp32 norm params @ 32 bits    (0.79% of the file)
 2,313,205 bytes total      9.65x smaller than the same model at fp16
```

## Speed, honestly

| where | tok/s | note |
|---|---|---|
| browser, WASM+relaxedSIMD | **1,045** | 11.57 GMAC/s, one thread, no server |
| T4, `torch.compile` + CUDA graphs | 736 | launch-bound: 0.2% of the card's compute |
| T4, eager PyTorch | 58 | |
| T4, packed ternary vs fp32 | **1.7× slower** | 96.9% of the overhead is activation quantization, not weights |

That last row is the interesting one. **A GPU has thousands of idle multipliers
either way**, so "we eliminated the multiplier" buys nothing on silicon designed
around multipliers — which is precisely why `bitnet.cpp` exists and why BitNet's
efficiency argument is about custom kernels and custom hardware. Memory is the win
available today.

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
print(tok.decode(model.generate(ids, max_new=150)[0].tolist()))
```

## Reproducing it

Free, start to finish. Kaggle notebook, GPU T4, **Internet ON**.

```
notebooks/01_data_kaggle.md      TinyStories -> 4096 BPE -> 477,236,558 tokens   ~30 min
notebooks/00_ladder_kaggle.py    the seven rungs                                 ~25 min
notebooks/02_train_kaggle.py     all five arms, packing, upload                   ~2 h
notebooks/03_build_wasm.py       compile kernel.c to SIMD128 wasm                 ~2 min
```

Under four hours of a 30-hour weekly quota. Phases 2 and 3 — where the learning
is — need no GPU at all.

## What it is not

**It is not a good model.** 20M tokens against 11.16M parameters is ~11× less than
compute-optimal, deliberately, because the point was measuring what ternary
weights cost rather than maximizing quality. Entity tracking fails in **every**
arm including fp32 — characters change name, objects become other objects, a
rabbit flies. It knows no facts, cannot answer questions, and has never seen
anything outside simple narrative English.

## Layout

```
bitllm/          model, quantizers, data pipeline, bit-packing, sampling, viz
notebooks/       every step, as run, with measured numbers inline
notes/           findings per phase, plus mistakes.md
wasm/            kernel.c + both compiled kernels (simd128 and relaxed)
web/             the browser demo: 4 JS modules, no dependencies
reference/       worked model.py and quant.py, held back during phase 2-3
space/           a Gradio app (unused: HF now gates Gradio Spaces behind PRO)
```

## Credits

Architecture from Ma et al., *The Era of 1-bit LLMs*, and the
[BitNet b1.58 2B4T technical report](https://arxiv.org/abs/2504.12285). Data from
Eldan & Li, *TinyStories*. Built with Claude Code.
