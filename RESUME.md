# Resume here

**Phases 1-4 complete. Project A is done.** The core experimental result is done and
published. What remains is ablations, packed inference, and writing it up.

## Everything is durable

| what | where |
|---|---|
| tokenizer + 477,236,558 token stream | HF `euclidstellar/tinystories-bpe4096` |
| fp32 control checkpoint | HF `euclidstellar/tinystories-1bit-llm/rung7_fp32.pt` |
| QAT-ternary checkpoint | HF `.../qat_ternary.pt` |
| equal-memory comparison model | HF `.../fp16_d128.pt` |
| every number + exact recipe | HF `.../results.json` |
| **the 1-bit model itself** | HF `.../model_packed.bin` -- **2.313 MB** |
| code + findings | github.com/EuclidStellar/1-bit-LLM |

Checkpoints hold **fp32 master weights** (44.7 MB). Ternary quantization happens
in the forward pass. The 4.61 MB packed figure is an inference artifact and
belongs to Phase 4.

## The headline result

| arm | val loss | ppl | packed |
|---|---|---|---|
| fp32 control | **2.0553** | 7.8 | 22.32 MB (fp16) |
| **QAT-ternary** | **2.1760** | 8.8 | **4.61 MB** |
| PTQ-ternary | **5.0229** | 151.9 | 4.61 MB |
| fp16 d=128 (equal memory) | 2.3607 | 10.6 | 4.21 MB |

```
QAT vs PTQ: 2.85 nats, 17x perplexity -- from WHEN the quantizer runs, nothing else
equal parameter count : ternary LOSES by 0.1207 nats
equal memory budget   : ternary WINS  by ~0.168 nats  (4.85x more params per byte)
QAT training overhead : 1.12x wall clock
```

## Restore a working Kaggle session

New notebook, **GPU T4 x2**, **Internet ON**.

```python
import sys, os, subprocess, importlib
REPO_DIR = "/kaggle/working/repo"
subprocess.run(["rm", "-rf", REPO_DIR])
subprocess.run(["git", "clone", "https://github.com/EuclidStellar/1-bit-LLM.git", REPO_DIR])
sys.path.insert(0, REPO_DIR)
importlib.invalidate_caches()          # REQUIRED -- see notes/phase-3
from bitllm import LM, BitLinear, fp32_model, ternary_model, weight_ternary
```

Then cell A (data + `get_batch` + `run`/`evaluate`/`causality_test`) from the
session log, then load any checkpoint:

```python
from huggingface_hub import hf_hub_download
ck = torch.load(hf_hub_download("euclidstellar/tinystories-1bit-llm",
                                "qat_ternary.pt"),
                map_location=dev, weights_only=False)
m_qat = ternary_model().to(dev); m_qat.load_state_dict(ck["model"])
```

`state_dict` keys are identical across arms, so any checkpoint loads into either
model. Loading `rung7_fp32.pt` into `ternary_model()` **is** the PTQ arm.

## Next, in order of value

**1. int8 embeddings (one forward pass, no training).** Ternarizing the embedding
collapses the model; int8 is a mild perturbation. If near-free:
packed 4.61 -> 3.28 MB, fp share 57.7% -> 40.5%, compression 4.85x -> 6.8x.
Not a step the paper takes.

```python
from bitllm import weight_intk
v0 = evaluate(m_qat)
bk = m_qat.embed.weight.data.clone()
m_qat.embed.weight.data = weight_intk(m_qat.embed.weight.data, 8)
print(f"fp16 embed {v0:.4f} -> int8 embed {evaluate(m_qat):.4f}")
m_qat.embed.weight.data = bk
```

**2. Generate from all three arms**, scored on the rubric: grammatical sentences,
dialogue attribution, **entity consistency** (fp32 already fails here -- expect it
to degrade first), story endings, invented non-words.

**3. Demolition list**, ~3 min each:

| ablation | why it matters |
|---|---|
| delete the SubLN pair | norms scored 337.7 nats/M in fp32, highest on the ladder; low-bit is where they should matter most |
| ternarize the embedding | asserts the 11.9% exclusion is load-bearing |
| ternary -> binary | 31% of weights ARE zeros; this removes a third of the expressiveness |
| activations int8 -> int4 -> int2 | why BitNet keeps activations at 8 bits |
| absmean -> absmax scaling | why BitNet chose the mean |
| remove `.detach()` from the STE | gradient 1.0 -> 7.12e-06; learning should stop |

**4. Measure the fair equal-memory point** rather than interpolating it:
`d=136, n_head=4` gives 2,340,424 params at 4.68 MB, within 1.5% of 4.61 MB.

**5. Phase 4** -- pack weights into real bits, integer inference path, measure
bytes on disk and tokens/sec.

**6. Phase 8** -- HF model card and a Gradio Space.

## Open predictions to check

- Depth contributed only 0.042 nats/M at 1.79 tokens/param. At 100M tokens
  (9 tokens/param) its share should grow. The Phase 2 ladder is 11x under-trained
  with a train/val gap of 0.014 -- no overfitting at all.
- Zero fraction stayed at ~31.4% after training, matching the Gaussian
  closed-form 31.01%. So it is a poor progress signal at this scale.
- Gradient norms are 24.6x larger at block 0 than block 7 (RMSNorm backward
  scales inversely with residual magnitude). Adam should cancel it. Worth
  re-measuring on the trained model.

## Blog-ready findings

1. `torch.cuda.is_bf16_supported()` returns True on a T4 and means "emulated"
2. One line of `clip_grad_norm_` was the difference between 8.0065 and 0.0144
3. 3.7% of the parameters delivered 70% of the gain (attention + norms + RoPE)
4. Normalization *replaced* part of the FFN rather than unlocking it
5. PTQ-ternary performs worse than a single fp32 attention layer
6. BitNet's own 2B model is **64.2% full precision by packed bytes**, and the
   published "Memory (Non-emb) 0.4GB" reproduces exactly from their config.json
7. The ~31% ternary zero fraction is derivable in closed form for Gaussian weights
8. `importlib.invalidate_caches()` after cloning a repo inside a notebook


---

# PROJECT A IS COMPLETE (2026-08-20)

## The deliverable

`euclidstellar/tinystories-1bit-llm/model_packed.bin` -- **2,313,205 bytes**

```
11,141,120 ternary weights at 1.6 bits/weight   (99.1% of the log2(3) floor)
    18,240 fp32 norm parameters
        49 packed tensors: 48 BitLinear + the tied embedding
9.65x smaller than the same model at fp16 (22.32 MB)
19.3x smaller than the fp32 training checkpoint (44.67 MB)
numerically exact: loss 2.315797 vs source 2.315795
```

## Every result

| | val loss | ppl | size |
|---|---|---|---|
| uniform baseline | 8.3178 | 4096 | -- |
| unigram baseline | 6.0380 | 419 | -- |
| PTQ-ternary | 5.0229 | 151.9 | -- |
| fp16 d=128 (equal memory) | 2.3607 | 10.6 | 4.21 MB |
| **QAT ternary + ternary embed** | **2.3107** | **10.1** | **2.31 MB** |
| QAT-ternary, fp16 embed | 2.1760 | 8.8 | 4.61 MB |
| fp32 control | 2.0553 | 7.8 | 22.32 MB |

```
QAT vs PTQ, body           2.85 nats from WHEN the quantizer runs, nothing else
QAT vs PTQ, embedding      76.0% of 0.5596 nats recovered
equal parameter count      ternary loses 0.1207 nats
equal memory budget        ternary wins ~0.17-0.23 nats across a 2x range
speed on a T4              1.7x SLOWER, and 96.9% of that is activation quant
```

## What is left: Phase 8, packaging

1. **HF model card** -- architecture, the Pareto table, the three-arm comparison,
   the honest speed result, limitations (under-trained at 1.79 tokens/param,
   entity tracking fails, TinyStories domain only)
2. **Gradio Space** loading `model_packed.bin` -- 2.31 MB loads instantly
3. **Blog post** from `notes/` -- eight findings are already written up
4. Optional: measure the fair equal-memory point (`d=136, n_head=4`,
   2,340,424 params at 4.68 MB) instead of interpolating it

## Project B, separate, if you want a GOOD model

Quantization is not the bottleneck -- training is. The fp32 control at 2.0553 is
itself bad (repeats "machine" eight times, loses entity identity). Levers, ranked:

| lever | cost per arm | tokens/param |
|---|---|---|
| 100M tokens | 76 min | 9.0 |
| 223M tokens | 2.8 h | 20 (Chinchilla) |
| 477M tokens (full corpus) | 6.1 h | 42.7 |
| context 256 -> 512 | ~1.4x on top | -- (**you trained at 256; design said 512**) |
| depth 8 -> 16 layers | ~1.8x on top | -- (also takes fp share 57.7% -> 40.9%) |

Recorded prediction to check: depth contributed only 0.042 nats/M at 1.79
tokens/param, worst on the ladder. Its share should grow with data.
