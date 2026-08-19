// ---------------------------------------------------------------------------
// kernel.c -- SIMD128 inner loops for the browser BitNet model.
//
// Only the hot loops live here. The control flow stays in JavaScript, where it
// has already been verified logit-by-logit against PyTorch -- so if matvec is
// correct, the whole forward pass stays correct.
//
// Memory is IMPORTED from JavaScript (env.memory), so JS owns the layout:
// weights, KV cache and scratch all live in one linear heap and these functions
// take raw float pointers into it. Nothing is copied per call.
//
// Build (see wasm/build.md):
//   clang --target=wasm32 -O3 -msimd128 -nostdlib -ffreestanding \
//         -Wl,--no-entry -Wl,--import-memory -Wl,--export-dynamic \
//         -o kernel.wasm kernel.c
// ---------------------------------------------------------------------------

#include <wasm_simd128.h>

#define EXPORT(name) __attribute__((export_name(name)))

// Rounding without libm. Built with -nostdlib, so __builtin_roundf emits a call
// to roundf that never resolves. WASM has f32x4.nearest as a real instruction,
// so splat-round-extract costs nothing and links cleanly.
//
// f32x4.nearest is roundTiesToEven -- which is what torch.round does. JavaScript
// Math.round is ties-away-from-zero, so this path actually matches PyTorch more
// closely than the pure-JS implementation it replaces.
static inline float nearest_f32(float v) {
  return wasm_f32x4_extract_lane(wasm_f32x4_nearest(wasm_f32x4_splat(v)), 0);
}

// y[o] = sum_i x[i] * W[o*inF + i]
//
// Four independent v128 accumulators: 16 floats per iteration. The four chains
// are independent so the pipeline can overlap them, which matters more than the
// raw lane count -- a single accumulator serializes on f32x4_add latency.
EXPORT("matvec")
void matvec(const float *x, const float *W, float *y, int outF, int inF) {
  for (int o = 0; o < outF; ++o) {
    const float *w = W + (long)o * inF;
    v128_t a0 = wasm_f32x4_splat(0.0f), a1 = wasm_f32x4_splat(0.0f);
    v128_t a2 = wasm_f32x4_splat(0.0f), a3 = wasm_f32x4_splat(0.0f);
    v128_t a4 = wasm_f32x4_splat(0.0f), a5 = wasm_f32x4_splat(0.0f);
    v128_t a6 = wasm_f32x4_splat(0.0f), a7 = wasm_f32x4_splat(0.0f);
    int i = 0;
    for (; i + 32 <= inF; i += 32) {
      a0 = wasm_f32x4_add(a0, wasm_f32x4_mul(wasm_v128_load(x + i),
                                             wasm_v128_load(w + i)));
      a1 = wasm_f32x4_add(a1, wasm_f32x4_mul(wasm_v128_load(x + i + 4),
                                             wasm_v128_load(w + i + 4)));
      a2 = wasm_f32x4_add(a2, wasm_f32x4_mul(wasm_v128_load(x + i + 8),
                                             wasm_v128_load(w + i + 8)));
      a3 = wasm_f32x4_add(a3, wasm_f32x4_mul(wasm_v128_load(x + i + 12),
                                             wasm_v128_load(w + i + 12)));
      a4 = wasm_f32x4_add(a4, wasm_f32x4_mul(wasm_v128_load(x + i + 16),
                                             wasm_v128_load(w + i + 16)));
      a5 = wasm_f32x4_add(a5, wasm_f32x4_mul(wasm_v128_load(x + i + 20),
                                             wasm_v128_load(w + i + 20)));
      a6 = wasm_f32x4_add(a6, wasm_f32x4_mul(wasm_v128_load(x + i + 24),
                                             wasm_v128_load(w + i + 24)));
      a7 = wasm_f32x4_add(a7, wasm_f32x4_mul(wasm_v128_load(x + i + 28),
                                             wasm_v128_load(w + i + 28)));
    }
    for (; i + 4 <= inF; i += 4)
      a0 = wasm_f32x4_add(a0, wasm_f32x4_mul(wasm_v128_load(x + i),
                                             wasm_v128_load(w + i)));
    v128_t a = wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_add(a0, a1),
                                             wasm_f32x4_add(a2, a3)),
                              wasm_f32x4_add(wasm_f32x4_add(a4, a5),
                                             wasm_f32x4_add(a6, a7)));
    float s = wasm_f32x4_extract_lane(a, 0) + wasm_f32x4_extract_lane(a, 1)
            + wasm_f32x4_extract_lane(a, 2) + wasm_f32x4_extract_lane(a, 3);
    for (; i < inF; ++i) s += x[i] * w[i];
    y[o] = s;
  }
}

// RMSNorm: no mean subtraction, no bias. Matches the PyTorch implementation.
EXPORT("rmsnorm")
void rmsnorm(const float *x, const float *w, float *out, int n, float eps) {
  v128_t s0 = wasm_f32x4_splat(0.0f), s1 = wasm_f32x4_splat(0.0f);
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    v128_t p = wasm_v128_load(x + i), q = wasm_v128_load(x + i + 4);
    s0 = wasm_f32x4_add(s0, wasm_f32x4_mul(p, p));
    s1 = wasm_f32x4_add(s1, wasm_f32x4_mul(q, q));
  }
  v128_t sv = wasm_f32x4_add(s0, s1);
  float ss = wasm_f32x4_extract_lane(sv, 0) + wasm_f32x4_extract_lane(sv, 1)
           + wasm_f32x4_extract_lane(sv, 2) + wasm_f32x4_extract_lane(sv, 3);
  for (; i < n; ++i) ss += x[i] * x[i];

  // f32.sqrt is a WASM instruction, so this needs no libm either
  float inv = 1.0f / __builtin_sqrtf(ss / (float)n + eps);
  v128_t iv = wasm_f32x4_splat(inv);
  i = 0;
  for (; i + 4 <= n; i += 4)
    wasm_v128_store(out + i, wasm_f32x4_mul(wasm_f32x4_mul(wasm_v128_load(x + i), iv),
                                            wasm_v128_load(w + i)));
  for (; i < n; ++i) out[i] = x[i] * inv * w[i];
}

// Per-token absmax int8 activation quantization -- BitNet's A8.
// A step function, so it must match PyTorch bit for bit or whole quantization
// levels flip rather than values nudging.
EXPORT("act_quant")
void act_quant(const float *x, float *out, int n) {
  const float qmax = 127.0f;

  v128_t m = wasm_f32x4_splat(0.0f);
  int i = 0;
  for (; i + 4 <= n; i += 4)
    m = wasm_f32x4_max(m, wasm_f32x4_abs(wasm_v128_load(x + i)));
  float amax = wasm_f32x4_extract_lane(m, 0);
  float m1 = wasm_f32x4_extract_lane(m, 1);
  float m2 = wasm_f32x4_extract_lane(m, 2);
  float m3 = wasm_f32x4_extract_lane(m, 3);
  if (m1 > amax) amax = m1;
  if (m2 > amax) amax = m2;
  if (m3 > amax) amax = m3;
  for (; i < n; ++i) { float a = x[i] < 0 ? -x[i] : x[i]; if (a > amax) amax = a; }

  float s = amax > 1e-5f ? amax : 1e-5f;
  float k = qmax / s, invk = s / qmax;
  v128_t kv = wasm_f32x4_splat(k), iv = wasm_f32x4_splat(invk);
  v128_t hi = wasm_f32x4_splat(qmax), lo = wasm_f32x4_splat(-qmax);
  i = 0;
  for (; i + 4 <= n; i += 4) {
    v128_t q = wasm_f32x4_nearest(wasm_f32x4_mul(wasm_v128_load(x + i), kv));
    q = wasm_f32x4_pmin(hi, wasm_f32x4_pmax(lo, q));
    wasm_v128_store(out + i, wasm_f32x4_mul(q, iv));
  }
  for (; i < n; ++i) {
    float q = nearest_f32(x[i] * k);
    if (q > qmax) q = qmax; else if (q < -qmax) q = -qmax;
    out[i] = q * invk;
  }
}

// Attention over the KV ring for one head.
//   scores[t] = dot(q + qo, kc + t*D + qo, hd) * scale
EXPORT("attn_scores")
void attn_scores(const float *q, const float *kc, float *scores,
                 int nc, int D, int qo, int hd, float scale) {
  for (int t = 0; t < nc; ++t) {
    const float *k = kc + (long)t * D + qo;
    const float *qq = q + qo;
    v128_t a0 = wasm_f32x4_splat(0.0f), a1 = wasm_f32x4_splat(0.0f);
    int i = 0;
    for (; i + 8 <= hd; i += 8) {
      a0 = wasm_f32x4_add(a0, wasm_f32x4_mul(wasm_v128_load(qq + i),
                                             wasm_v128_load(k + i)));
      a1 = wasm_f32x4_add(a1, wasm_f32x4_mul(wasm_v128_load(qq + i + 4),
                                             wasm_v128_load(k + i + 4)));
    }
    v128_t a = wasm_f32x4_add(a0, a1);
    float s = wasm_f32x4_extract_lane(a, 0) + wasm_f32x4_extract_lane(a, 1)
            + wasm_f32x4_extract_lane(a, 2) + wasm_f32x4_extract_lane(a, 3);
    for (; i < hd; ++i) s += qq[i] * k[i];
    scores[t] = s * scale;
  }
}

// out[qo..qo+hd) = sum_t w[t] * vc[t*D + qo ..]
EXPORT("attn_mix")
void attn_mix(const float *w, const float *vc, float *out,
              int nc, int D, int qo, int hd) {
  for (int i = 0; i < hd; ++i) out[qo + i] = 0.0f;
  for (int t = 0; t < nc; ++t) {
    v128_t wv = wasm_f32x4_splat(w[t]);
    const float *v = vc + (long)t * D + qo;
    float *o = out + qo;
    int i = 0;
    for (; i + 4 <= hd; i += 4)
      wasm_v128_store(o + i, wasm_f32x4_add(wasm_v128_load(o + i),
                                            wasm_f32x4_mul(wv, wasm_v128_load(v + i))));
    for (; i < hd; ++i) o[i] += w[t] * v[i];
  }
}

// x += y, elementwise. The residual stream.
EXPORT("add_inplace")
void add_inplace(float *x, const float *y, int n) {
  int i = 0;
  for (; i + 4 <= n; i += 4)
    wasm_v128_store(x + i, wasm_f32x4_add(wasm_v128_load(x + i), wasm_v128_load(y + i)));
  for (; i < n; ++i) x[i] += y[i];
}

// squared ReLU, in place. BitNet uses this instead of SwiGLU, for sparsity.
EXPORT("relu2")
void relu2(float *x, int n) {
  v128_t z = wasm_f32x4_splat(0.0f);
  int i = 0;
  for (; i + 4 <= n; i += 4) {
    v128_t r = wasm_f32x4_max(wasm_v128_load(x + i), z);
    wasm_v128_store(x + i, wasm_f32x4_mul(r, r));
  }
  for (; i < n; ++i) { float r = x[i] > 0 ? x[i] : 0; x[i] = r * r; }
}

// copy a row out of the embedding table
EXPORT("gather_row")
void gather_row(const float *table, float *out, int row, int n) {
  const float *src = table + (long)row * n;
  int i = 0;
  for (; i + 4 <= n; i += 4) wasm_v128_store(out + i, wasm_v128_load(src + i));
  for (; i < n; ++i) out[i] = src[i];
}


// ---------------------------------------------------------------------------
// exp() without libm, and a fully fused attention head.
// ---------------------------------------------------------------------------

static inline float pow2i(int i) {
  if (i < -126) return 0.0f;
  if (i > 127) i = 127;
  union { unsigned u; float f; } r;
  r.u = ((unsigned)(i + 127)) << 23;      // f32.reinterpret_i32, one instruction
  return r.f;
}

// exp(x) via 2^(x*log2e), split into an integer power of two and a minimax
// polynomial on [-0.5, 0.5]. Accurate to ~1e-7 relative, which is far beyond
// what softmax needs after the max has been subtracted (so x <= 0 always).
static inline float fast_expf(float x) {
  if (x < -87.0f) return 0.0f;
  float t = x * 1.44269504088896f;                 // log2(e)
  int i = (int)(t < 0.0f ? t - 0.5f : t + 0.5f);   // nearest
  float f = t - (float)i;
  float p = 1.0f + f * (0.69314718f + f * (0.24022651f + f * (0.05550411f
          + f * (0.00961812f + f * 0.00133336f))));
  return p * pow2i(i);
}

// One head of causal attention, start to finish: scores, softmax, weighted sum.
//
// Replaces attn_scores + a JavaScript softmax + attn_mix. That was 16,384
// Math.exp calls and 192 WASM boundary crossings per token; this is 64 calls and
// no JS exp at all.
EXPORT("attn_head")
void attn_head(const float *q, const float *kc, const float *vc, float *out,
               float *sc, int nc, int D, int qo, int hd, float scale) {
  const float *qq = q + qo;

  float mx = -1e30f;
  for (int t = 0; t < nc; ++t) {
    const float *k = kc + (long)t * D + qo;
    v128_t a0 = wasm_f32x4_splat(0.0f), a1 = wasm_f32x4_splat(0.0f);
    int i = 0;
    for (; i + 8 <= hd; i += 8) {
      a0 = wasm_f32x4_add(a0, wasm_f32x4_mul(wasm_v128_load(qq + i),
                                             wasm_v128_load(k + i)));
      a1 = wasm_f32x4_add(a1, wasm_f32x4_mul(wasm_v128_load(qq + i + 4),
                                             wasm_v128_load(k + i + 4)));
    }
    v128_t a = wasm_f32x4_add(a0, a1);
    float d = wasm_f32x4_extract_lane(a, 0) + wasm_f32x4_extract_lane(a, 1)
            + wasm_f32x4_extract_lane(a, 2) + wasm_f32x4_extract_lane(a, 3);
    for (; i < hd; ++i) d += qq[i] * k[i];
    d *= scale;
    sc[t] = d;
    if (d > mx) mx = d;
  }

  float sum = 0.0f;
  for (int t = 0; t < nc; ++t) { float e = fast_expf(sc[t] - mx); sc[t] = e; sum += e; }
  float inv = 1.0f / sum;

  float *o = out + qo;
  for (int i = 0; i < hd; ++i) o[i] = 0.0f;
  for (int t = 0; t < nc; ++t) {
    v128_t wv = wasm_f32x4_splat(sc[t] * inv);
    const float *v = vc + (long)t * D + qo;
    int i = 0;
    for (; i + 4 <= hd; i += 4)
      wasm_v128_store(o + i, wasm_f32x4_add(wasm_v128_load(o + i),
                                            wasm_f32x4_mul(wv, wasm_v128_load(v + i))));
    for (; i < hd; ++i) o[i] += sc[t] * inv * v[i];
  }
}
