# Phase 5 — Getting it into a browser, 20 to 1,045 tok/s

Goal: the 2.31 MB file should run where anyone can reach it, with no server, fast
enough that the speed is the point rather than an apology.

Final: **1,045 tok/s average, 1,038 steady-state on a full 256-token window,
11.57 GMAC/s**, in a tab, on one thread.

## The route

| tok/s | change | why it worked |
|---|---|---|
| 20 | plain JS, literal add/subtract/skip | — |
| 110 | branchless, eight accumulators | 31% of weights are zero, so the ternary branch is unpredictable and mispredicted on ~2/3 of 11M iterations per token. A dense f32 multiply is *faster* than the branch it replaces. Eight accumulators break the serial dependency on float-add latency. |
| 600 | WASM + SIMD128 | 15 KB of C. Four-wide f32 lanes, no bounds checks, and a fused attention head. |
| 900 | fixed the sampler | It was **sorting all 4096 logits to select 100** — ~49,000 comparator-closure calls per token. Replaced with a size-k min-heap. |
| **1,045** | relaxed-SIMD FMA | `f32x4_relaxed_madd` is one instruction where SIMD128 needs a multiply and an add. |

## The profiler is the only reason the last two rows exist

Two hypotheses asserted and both wrong before I measured anything:

1. **"Softmax is the bottleneck."** Fused scores+softmax+mix into one WASM call
   with a libm-free `exp`. Result: no change.
2. **"It is memory-bound."** The dense f32 expansion reads 44.6 MB of weights per
   token; at 600 tok/s that is 26.8 GB/s, and nothing near 44 MB fits in cache.
   Added `matvec_i8` to read the int8 states directly, a 4x traffic reduction.
   Result: 3% **slower**.

Then a profiler timing every kernel op at its real size and call count:

```
matvec 320x320  (q,k,v,o)   0.0075 ms x 32 = 0.240 ms  28.3%   13.65 GMAC/s
matvec 1280x320 (up)        0.0300 ms x  8 = 0.240 ms  28.3%   13.65 GMAC/s
matvec 320x1280 (down)      0.0288 ms x  8 = 0.230 ms  27.1%   14.25 GMAC/s
matvec 4096x320 (head)      0.0935 ms x  1 = 0.094 ms  11.0%   14.02 GMAC/s
attn_head                   ~0     ms x 64 = 0.000 ms   0.0%   (fused, free)
add_inplace / _rope (JS)                     0.040 ms   4.8%
                                            ---------
modelled total                               0.849 ms
measured forward()                           0.870 ms   98% accounted
```

`forward()` was **0.870 ms = 1,149 tok/s** while generation ran at **535**. Half
the time was outside the model entirely, in a sort nobody had looked at. Matvecs
are 94.7% of the forward pass at 13-14 GMAC/s, near the SIMD128 ceiling -- the
kernel was never the problem, and both "failed" changes had in fact worked. Their
gains were invisible because sampling swamped them.

**Lesson: measure the thing you are optimising, not the thing you ship.** Three
rounds were spent computing GMAC/s from generation throughput, which had a
sampler in the denominator.

## Context: a ring buffer, not a sliding window

The model's attention window is 256 tokens. The first attempt at longer output
re-prefilled the last 64 tokens whenever the cache filled -- 33% overhead, and it
destroyed three quarters of the context every 192 tokens, so long generations came
out as a pile of separate stories.

The correct approach was dismissed too early. **RoPE scores depend only on the
relative offset (t - p), never on absolute position.** So absolute position may
run to 4,000+ provided attention reaches back at most 256 tokens, because every
offset then stays inside the trained range. A plain ring buffer: drop the oldest
key as each new one arrives, no re-prefill, no re-rotation, no memmove. Full
256-token context **and** the overhead gone.

## Two failure modes that were mine, not the model's

**Banning `<|endoftext|>` made it unable to stop.** Asked for one continuous story
instead of several, the obvious move was to ban the end-of-story token. But the
model was trained on ~227-token stories and has no concept of a longer one -- it
reaches its natural terminus, writes "The end.", and then loops on it forever.
Fixed with n-gram blocking (looping becomes structurally impossible) plus a
frequency penalty, so it is forced to keep finding something new to say.

**`await requestAnimationFrame` throttled generation to the display refresh
rate.** Yielding every 4 tokens to paint capped throughput at ~60 x 4 = 240-480
tok/s regardless of model speed: 1,045 became 483. `MessageChannel.postMessage`
posts a macrotask with no frame wait and no 4 ms `setTimeout` clamp, so the
browser can paint and generation resumes immediately. Paints now run on a 16 ms
budget -- ~60/sec at ~16 tokens each, about 3% of throughput.

## Where the ceiling is

13-14 GMAC/s on the matvecs is close to what four-wide SIMD can do, and matvecs
are 94.7% of the work. Past this you need `SharedArrayBuffer` threads (COOP/COEP
headers, plus 49 barrier syncs per token that would likely eat the gain) or
WebGPU with fused per-block shaders. Neither is cleverness applied to the current
design; both are a different design.
