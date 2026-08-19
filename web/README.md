# Browser demo

An 11M-parameter BitNet b1.58 model running entirely client-side. Weights are
fetched once from the Hugging Face model repo (2.31 MB) and every token after
that is computed locally -- no server, no API.

| file | what |
|---|---|
| `index.html` | the page: generation UI, TPS counter, ternary weight viewer |
| `bitllm.js` | base-3 unpacking, RMSNorm, RoPE, causal attention with a KV cache, squared ReLU, per-token int8 activation quantization |
| `tokenizer.js` | byte-level BPE, encode and decode, read from `tokenizer.json` |

On load the page replays the PyTorch reference (`reference.json` in the model
repo) through its own forward pass and compares logits. A wrong implementation
fails visibly instead of quietly producing slightly-wrong stories.

## Run locally

`fetch` will not work from `file://`, so serve it:

```bash
cd web && python3 -m http.server 8000
# http://localhost:8000
```

## Deploy

```bash
npx vercel --prod          # from this directory
```

Static only -- zero function invocations, zero CPU-hours, so it stays inside
Vercel's Hobby allowances no matter how much traffic it gets.

## Stage 1 vs stage 2

This is **stage 1: plain JavaScript, correctness first.** Expect roughly 15-40
tok/s single-threaded. Stage 2 ports the six matmuls per block to WASM+SIMD with
web workers (~800-1,600 tok/s) using this implementation as the reference.
