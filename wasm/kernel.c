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
    int i = 0;
    for (; i + 16 <= inF; i += 16) {
      a0 = wasm_f32x4_add(a0, wasm_f32x4_mul(wasm_v128_load(x + i),
                                             wasm_v128_load(w + i)));
      a1 = wasm_f32x4_add(a1, wasm_f32x4_mul(wasm_v128_load(x + i + 4),
                                             wasm_v128_load(w + i + 4)));
      a2 = wasm_f32x4_add(a2, wasm_f32x4_mul(wasm_v128_load(x + i + 8),
                                             wasm_v128_load(w + i + 8)));
      a3 = wasm_f32x4_add(a3, wasm_f32x4_mul(wasm_v128_load(x + i + 12),
                                             wasm_v128_load(w + i + 12)));
    }
    v128_t a = wasm_f32x4_add(wasm_f32x4_add(a0, a1), wasm_f32x4_add(a2, a3));
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
  for (int l = 1; l < 4; ++l) {
    float v = l == 1 ? wasm_f32x4_extract_lane(m, 1)
            : l == 2 ? wasm_f32x4_extract_lane(m, 2)
                     : wasm_f32x4_extract_lane(m, 3);
    if (v > amax) amax = v;
  }
  for (; i < n; ++i) { float a = x[i] < 0 ? -x[i] : x[i]; if (a > amax) amax = a; }

  float s = amax > 1e-5f ? amax : 1e-5f;
  float k = qmax / s, invk = s / qmax;
  for (i = 0; i < n; ++i) {
    float q = __builtin_roundf(x[i] * k);
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
