# Phase 1 — Tokenizer and the token stream

Measured on Kaggle, T4 x2 session, TinyStories, 2026-08-18.

## Outcome

| | |
|---|---|
| corpus | TinyStories, 1.8 GB train + 19 MB valid |
| vocabulary | 4,096 (256 byte seeds + 1 special + **3,839 learned merges**) |
| compression | **4.04 chars/token** |
| train stream | **477,236,558 tokens** (954 MB uint16) |
| val stream | 4,801,337 tokens |
| max token id | 4095 against vocab 4096 — full vocabulary in use, nothing out of range |

---

## Finding 1 — `torch.cuda.is_bf16_supported()` reports True on a T4, and it's lying

The T4 reported `bf16: True`. Turing (SM 7.5) has **no bf16 tensor-core path**. From
the torch source:

```python
def is_bf16_supported(including_emulation: bool = True):
    if torch.cuda.get_device_properties(device).major >= 8:
        return True                  # Ampere+ : native
    if not including_emulation:
        return False                 # T4 is major=7 -> False here
    return _check_bf16_tensor_supported(device)   # -> True, but EMULATED
```

The default `including_emulation=True` answers "can bf16 tensors exist", not "is
bf16 fast". Selecting bf16 on that basis costs you the fp16 tensor cores
(~65 TFLOPS) in exchange for software emulation, with no error and no warning.

**Always pass `including_emulation=False`.** On a T4 the correct choice is
fp16 + GradScaler.

## Finding 2 — merges compose, and you can watch it happen

Traced on `"little little little littlest happy happy happier happiest"`:

```
step 1   'l' + 'i'      seen 4x   (tied with 3 other pairs)  -> 'li'
step 2   'li' + 't'     seen 4x                              -> 'lit'
step 3   'lit' + 't'    seen 4x                              -> 'litt'
step 4   'litt' + 'l'   seen 4x                              -> 'littl'
step 5   'littl' + 'e'  seen 4x                              -> 'little'
```

Six merges to build one word, each consuming the previous merge's output. The
same chain appears in the real 4,096-vocab table: merge 2 `Ġ`+`t`, merge 1
`h`+`e`, then **merge 7** `Ġt`+`he` -> `Ġthe`. Also merge 9 `Ġa`+`nd` -> `Ġand`
from merges 3 and 5, and merge 26 `Ġwa`+`s` -> `Ġwas` from merge 14.

Two consequences nobody programmed: `littlest` reuses the `little` token and
keeps `st` separate, and `happier`/`happiest` share `Ġha`. Stem/suffix structure
falls out of frequency counting alone.

## Finding 3 — `add_prefix_space=False` changes the vocabulary

```
Ġonce         id 2969
ĠOnce         NOT a single token      <- "Once upon a time" is the corpus's
Ġupon         id 453                     most common opening
Ġtime         id 404
```

Because stories start at the beginning of a line, `Once` has no preceding space
to absorb — so it is tokenized as `Once` (no `Ġ`), and `ĠOnce` only occurs
mid-sentence, where it is rare. Case is not shared either: `Ġonce` and `ĠOnce`
are unrelated vocabulary entries. One boolean flag, and the tokenization of the
corpus's signature phrase changes.

## Finding 4 — 4.04 chars/token independently reproduces the paper's token count

```
1.8 GB / 4.04 chars per token  ~=  478M tokens
measured                            477,236,558
TinyStories paper documents         ~480M
```

Within 0.6%, using a **4,096**-token vocabulary against the paper's 10,000. A
vocabulary 2.4x smaller compressed essentially as well — because TinyStories was
deliberately built with a restricted lexicon, so 4,096 already covers nearly all
of it. On general web text a 4k vocab would compress far worse. This is a
property of the corpus, not of BPE.

## Finding 5 — the longest learned tokens expose the corpus's design intent

```
_uncomfortable  _compassionate  _accidentally  _disappointed
_enthusiastic   _embarrassed    _adventurous   _caterpillar
_butterflies    _remembered     _understood    _eventually
```

Two clusters: **emotional states** and **narrative connectives**. Frequency
analysis alone reveals that TinyStories was engineered to teach emotional
vocabulary and story structure to small models. The corpus's construction shows
through its own statistics.

## Finding 6 — the 1-bit memory claim has an asterisk, and it is the embedding table

This is the most consequential finding of the phase.

```
embedding  = vocab x d          <- can NEVER be ternarized
per layer  = 12 x d^2           <- 4d^2 attention + 8d^2 FFN, all ternarizable
```

Vocabulary adds **only** un-quantizable parameters. Depth adds **only**
quantizable ones. Holding d=320, L=8:

| vocab | embedding params | full-precision share | packed size |
|---|---|---|---|
| 1,024 | 327,680 | 3.4% | 2.64 MB |
| **4,096** | **1,310,720** | **11.9%** | **4.61 MB** |
| 16,384 | 5,242,880 | 34.9% | 12.47 MB |
| 32,768 | 10,485,760 | 51.7% | 22.96 MB |
| 128,256 | 41,041,920 | **80.7%** | 84.07 MB |

And depth runs the other way — holding vocab=4,096, d=320:

| layers | full-precision share |
|---|---|
| 4 | 21.2% |
| **8** | **11.9%** |
| 16 | 6.4% |
| 24 | 4.4% |

At BitNet b1.58 2B4T's actual `config.json` — `vocab_size: 128256`,
`hidden_size: 2560`, `tie_word_embeddings: true` — the tied embedding table is
**328M parameters, ~656 MB at bf16**, against a ternary body of ~400 MB. The
un-quantizable part is *larger than the thing being compressed*.

Which is exactly why their headline efficiency figure is labelled
**"Memory (Non-emb)"** and excludes it.

**Rule that follows: to strengthen the 1-bit claim, add layers, never vocabulary.**

## Finding 7 — the ~50M token floor is a confound argument, not a Chinchilla argument

Below roughly 50M tokens the fp32 arm memorizes the training set while the
ternary arm cannot. The measured gap then reflects **capacity to memorize**, not
capacity to learn — contaminating the only comparison the project exists to make.
This is what disqualified every genuinely niche corpus considered: Cricsheet IPL
(<1M tokens), CCCBR change ringing (20k methods), Lojban (22k sentences),
stitch-maps knitting (4,728 patterns), SCP Foundation (~8M tokens).

Niche means small. That is what the word means.

## Decisions locked by this phase

| decision | value | reversible? |
|---|---|---|
| vocabulary size | 4,096 | **no** — shapes the embedding table, baked into every checkpoint |
| tokenizer | byte-level BPE, `add_prefix_space=False` | no — ids would all change |
| separator | `<|endoftext|>` at id 0 | no |
| dtype on disk | uint16 | yes, but no reason to |
| mixed precision | fp16 + GradScaler | yes |
