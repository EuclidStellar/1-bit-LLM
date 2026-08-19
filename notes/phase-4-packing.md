# Phase 4 — Packing the weights, and the inference reality

Goal: turn the claimed 2.24 MB into a file, and test whether "no multipliers"
translates into speed on real hardware.

## The packed file

**FINAL** (norms at fp32; see the fidelity section for why):

```
packing scheme            base 3, five ternary weights per uint8 (3^5 = 243 <= 255)
bits per ternary weight   1.6000        floor log2(3) = 1.5850 -> 99.1% efficient
packed tensors            49            48 BitLinear + the tied embedding
ternary weights           11,141,120
fp32 weights                  18,240    the 41 RMSNorm vectors, nothing else

blob                       2,301,184 B   <- predicted to the byte
header (JSON)                 12,021 B
TOTAL                      2,313,205 B  = 2.313 MB

fp32 training checkpoint      44.67 MB
same model at fp16            22.32 MB
compression vs fp16            9.65x
```

Fidelity, final:

```
val loss   packed 2.315797   source 2.315795      difference 2e-6 nats
max logit delta, act quant OFF   1.144e-05        fp32 accumulation noise
```

The intermediate fp16-norms build measured 2,276,524 B / 9.80x compression but
deviated 4.025e-03 in logits. fp32 norms cost 36,480 bytes (+1.6% of the file)
and improved fidelity **352x**. Worth it: the norms are 0.16% of parameters, and
"numerically exact packed inference" is a materially stronger claim than
"matches to 4e-3".

Loss is preserved: **packed 2.315807 vs original 2.315795**, a difference of
1.2e-5 nats. The packed file stores `states` and one fp32 `scale` per tensor, and
`states * scale` reproduces `weight_ternary(w)` exactly -- so the packed model is
not an approximation of the QAT model, it *is* the QAT model at 1.6 bits/weight.

Rejected alternative: 2 bits per weight, 4 per byte. Wastes one code in four,
costs 2.0 bits/weight (79% efficient), and would have produced 2.79 MB.

## Two bugs worth writing down

### isinstance() silently fails across module reloads

`save_packed` detected quantized layers with `isinstance(mod, BitLinear)`. In a
notebook that had done `del sys.modules['bitllm*']` and re-imported, the reload
creates a **new** `BitLinear` class object while the existing model holds
instances of the old one. Same name, different class, `isinstance` returns False.

Consequence: all 48 body weights were stored as fp16 instead of packed. The file
came out at 19,959,424 bytes, which decomposes exactly:

```
9,830,400 body at fp16  = 19,660,800
1,310,720 embed packed  =    262,144
   18,240 norms at fp16 =     36,480
                          -----------
                          19,959,424
```

No error, no warning. Fix: duck-type on `getattr(mod, "weight_mode", "none")`
instead, plus an assert that refuses to write a file claiming to be packed when
nothing was packed.

### Double quantization on load

`load_packed` returns `states * scale`. Loading that into a model whose forward
pass quantizes again leaves the states unchanged but recomputes the scale:

```
new_g = mean(|states * g|) = g * (1 - zero_fraction) ~ 0.686 * g
```

Every weight shrank 31%, and because the embedding is tied to the LM head the
logits shrank too. Symptom: val loss 3.3545 instead of 2.3158, max logit delta
9.111. Fix: `inference_config()` sets `weight_mode` and `embed_mode` to "none"
while preserving `act_bits`, since activation quantization is a runtime operation
that the QAT model was trained with.

## Numerical fidelity: the activation quantizer amplifies 44x

```
max logit delta, act quant ON  : 1.781e-01   mean 1.408e-02
max logit delta, act quant OFF : 4.025e-03   mean 3.220e-04
```

`act_quant` contains `round()`. An activation sitting near a quantization
boundary flips by a **full step** in response to a 1-ulp weight difference, so
tiny differences become discrete jumps. 48 BitLinears with per-token scaling
guarantee some activations sit on boundaries.

The residual 4e-3 with quantization off is **the fp16 norm storage** -- 5e-4
relative precision compounded through 41 norm layers and amplified by the LM
head. Storing norms at fp32 instead costs 36 KB on a 2.277 MB file (+1.6%) and
makes the packed model numerically exact. Worth doing.

**Lesson: with discrete quantizers in the graph, per-element output agreement is
the wrong fidelity test.** Compare the loss. When the weights were genuinely
wrong the loss was off by a whole nat (3.3545 vs 2.3158); when they are right it
agrees to 1.2e-5 despite max logit deltas of 0.178.

## The speed result: 1.7x SLOWER, and the weights are not the reason

Batch 32 x 256 on a T4, forward only:

| configuration | tok/s | ms/forward | vs fp32 |
|---|---|---|---|
| fp32 weights, no quantization | 91,926 | 89.1 | 1.00x |
| ternary weights, QAT forward | 53,500 | 153.1 | **1.72x slower** |
| packed -> unpacked, act quant only | 54,217 | 151.1 | **1.70x slower** |

Decomposing the 64.0 ms of overhead:

```
activation quantization   62.0 ms    96.9%
weight quantization        2.0 ms     3.1%
```

The packed model skips `weight_ternary` **entirely** and gains only **1.3%**. The
weight quantizer was never the cost. It is element counts, per block:

```
weight tensors     :  1,228,800 elements
activation tensors : 23,500,000 elements     19x more
```

`act_quant` runs an `amax` reduction plus round and clamp over tensors 19x larger
than the weights, at every one of 48 BitLinears.

### Why this is the expected answer, not a failed experiment

A T4 has thousands of hardware multipliers that sit idle whether or not you use
them, and no unpacking unit. "We eliminated the multiplier" buys nothing on
silicon designed around multipliers. This is precisely why `bitnet.cpp` exists
and why BitNet's efficiency argument is about **custom kernels and custom
hardware**, not about PyTorch on a GPU.

**The honest summary: on a T4 in PyTorch, a 1-bit LLM is 1.7x slower than fp32
and 9.80x smaller.** Memory is the win that is available today; compute requires
hardware that does not already give multiplication away for free.
