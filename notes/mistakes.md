# Everything that went wrong

Kept because the corrections are the most useful part of this project. Every
entry is something that was measured wrong, predicted wrong, or shipped broken —
with what it cost.

---

## Traps in the tooling

**`torch.cuda.is_bf16_supported()` returns `True` on a T4.** It defaults to
`including_emulation=True`, and a Turing card (SM 7.5) has no bf16 tensor-core
path at all. Trusting it means giving up ~65 TFLOPS of fp16 tensor cores for
software emulation, silently. Ask for `including_emulation=False`.

**`importlib.invalidate_caches()` after cloning a repo inside a notebook.** A
first attempt that `rm -rf`s a directory and then fails to clone leaves it on
`sys.path` while empty. Python's `FileFinder` caches that empty listing in
`sys.path_importer_cache`, keyed by path string, and it survives removing and
re-adding the entry to `sys.path`. The files then exist and the import still
fails. Cost: three rounds of confusion.

**Kaggle Quick Save does not snapshot `/kaggle/working`.** It persists notebook
code and rendered output, nothing else. Only a completed *Save & Run All* produces
version output files. Combined with `/kaggle/working` being session-scoped, that
is two independent ways to lose data — and it destroyed 477,236,558 tokens and
cost a 30-minute regeneration. The fix was to stop treating Kaggle as storage:
data lives on HF Hub now, Kaggle is pure compute.

**Hugging Face usernames are case-sensitive and unrelated to GitHub.** GitHub
`EuclidStellar` vs HF `euclidstellar` produced `403 Forbidden: you don't have the
rights to create a dataset under the namespace "EuclidStellar"` — which reads
like a permissions problem and is not. Derive it: `api.whoami()["name"]`.

**`isinstance()` fails across module reloads.** After `del sys.modules['bitllm*']`
and a re-import, the reload creates a *new* `BitLinear` class object while
existing models hold instances of the old one. Same name, different class,
`isinstance` returns `False`. The bit-packer therefore found zero quantized layers
and stored all 9,830,400 body weights as fp16 — a **19,959,424-byte file instead
of 2,276,524**, with no error. Duck-type on an attribute instead.

**A catch-all SPA rewrite eats static files.** `{"source": "/(.*)", "destination":
"/index.html"}` sent `/1bit/index.html` to the React app, which 404'd. Needs a
negative lookahead.

---

## Predictions that were wrong

**"Removing the STE makes the gradient exactly zero."** Measured `7.12e-06`, not
`0`. `round()` does contribute exactly nothing — but `weight_ternary` divides and
then multiplies by `mean(|W|)`, and **that scale is differentiable**. The leak is
through the scale, not the quantizer. Detach only the scale and it *is* exactly
zero. The practical conclusion survives (1.0 vs 7e-06 is 140,000× less signal)
but the mechanism was different.

**"Attention without position encoding will stay at the floor."** On a
copy-from-2-back task with an 8-token alphabet, rung 3 beat the `ln(8)` floor by
0.53 nats. The **prefix multiset is informative even unordered** — the target is a
*member* of it, so knowing which tokens appear shifts the posterior without
locating any of them.

**"Normalization will unlock the FFN."** It did the opposite: the FFN's marginal
contribution *shrank* 40% (+0.1956 → +0.1175 nats). Part of what an unnormalized
FFN buys is **scale correction, not computation** — and 2,560 dedicated norm
parameters do that job better and far cheaper.

**"Ternarize the embedding and the model collapses into noise."** Repeated several
times before measuring it. Actual cost: **+0.5596 nats**, ppl 8.8 → 15.5.
Bad — 4.6× the cost of int8 — but nowhere near collapse (PTQ on the body was ppl
151.9; the unigram floor is 419). The rule is real; its usual statement is too
strong.

**"Softmax is the browser bottleneck."** Fused scores+softmax+mix into one WASM
call with a libm-free `exp`. No change.

**"It's memory-bound."** Dense f32 weights read 44.6 MB per token — 26.8 GB/s at
600 tok/s, and nothing near 44 MB fits in cache. Reading int8 states directly cut
traffic 4×. Result: **3% slower.**

Both of those were asserted instead of measured. The profiler that followed found
the real answer in one round, and showed that *both changes had actually worked* —
their gains were invisible because the sampler swamped them.

---

## Tests that tested nothing

**A test any model passes by memorizing.** "target[t] = input[t−2]" over a
4096-token vocabulary with ~500 positions means every input token in the batch is
unique, so it collapses to a lookup table. Rung 2 — which has no context at all —
scored `0.0000`. Fixed by shrinking the alphabet to 8 so tokens repeat and
memorization is impossible.

**A helper missing `clip_grad_norm_`.** The main harness had it; the side-test
helper did not. Same learning rate, same everything: **8.0065 without clipping,
0.0144 with.** Worse-than-random versus solved, from one line. Which rungs
collapsed was itself diagnostic — rung 3 (no FFN) survived, rung 5 (ReLU² FFN, no
norms) exploded.

**Returning only the final loss.** Makes divergence indistinguishable from
"learned poorly". Always print the trajectory.

**Expecting bit-exact logits from a model containing step functions.** After
bit-packing, `max logit delta` was `1.781e-01` and I called it a possible bug. The
int8 activation quantizer contains `round()`: an activation near a boundary flips a
**whole quantization level** in response to a 1-ulp weight difference, and 48
BitLinears with per-token scaling guarantee some sit on boundaries. Measured
amplification: **44×**. The right fidelity test is the loss — which agreed to
`1.2e-05` — not per-element agreement. When the weights were *genuinely* wrong the
loss was off by a full nat.

**Computing GMAC/s from generation throughput.** For three rounds. Generation had
a sampler in the denominator, so the kernel looked like 6.7 GMAC/s when the
matvecs were actually running at 13–14.

**Comparing runs of different lengths.** 600 → 580 → 535 tok/s looked like a
regression across two changes. It wasn't: attention cost scales with ring fill, and
a 2048-token run keeps a full 256-token window for 87% of its tokens versus 57% for
a 600-token run. Steady-state is now reported separately.

---

## Bugs that shipped

**Double quantization on load.** `load_packed` returns `states × scale`. Loading
that into a model whose forward pass quantizes *again* leaves the states alone but
recomputes the scale as `mean(|states·g|) = g·(1 − zero_fraction) ≈ 0.686g` —
every weight shrank 31%, and since the embedding is tied to the LM head, so did
the logits. Val loss 3.3545 instead of 2.3158.

**`await requestAnimationFrame` in the generation loop.** Blocks until the next
repaint (~16.7 ms), so yielding every 4 tokens capped throughput at ~60 × 4 =
240–480 tok/s regardless of model speed. **1,045 became 483.** `MessageChannel`
posts a macrotask with no frame wait and no 4 ms `setTimeout` clamp.

**Banning `<|endoftext|>` to get one long story.** The model was trained on
~227-token stories and has no concept of a longer one. Denied its ending, it wrote
"The end." and then looped on it forever. Banning the stop token does not make a
model write more; it makes it unable to stop.

**A hardcoded status string.** The UI read "plain JavaScript" while WASM was
active, because a `.replace()` never matched. Cost a round of diagnosing a
regression that didn't exist.

**Publishing code that had never been run.** `bitllm/train.py` — argparse,
checkpoint/resume, AMP — was written during planning and never executed. Every
published number came from cells pasted into a notebook that were never committed.
Anyone opening the repo for "the training code" would have read the wrong file.
Fixed by writing down what was actually run and annotating the other.

---

## The one that cost the most

**Four rounds searching for a creative dataset when the binding constraint was
size.** Cricsheet IPL (<1M tokens), CCCBR change ringing (20k methods), Lojban
(22k sentences), stitch-maps knitting (4,728 patterns), SCP Foundation (~8M
tokens) — every one rejected on "not creative enough" while the actual
disqualifier went unchecked.

And the reason size matters is not Chinchilla-optimality, it is a **confound**:
below roughly 50M tokens the fp32 arm memorizes and the ternary arm cannot, so the
measured gap reflects *capacity to memorize* rather than capacity to learn — which
contaminates the only comparison the project exists to make.

Niche means small. That is what the word means.
