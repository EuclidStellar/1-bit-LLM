// ---------------------------------------------------------------------------
// bitllm.js -- an 11M-parameter BitNet b1.58 model, in the browser, in plain JS.
//
// Reads model_packed.bin directly: base-3 packed ternary weights, five per byte.
// Verified logit-by-logit against the PyTorch reference before it is trusted.
// ---------------------------------------------------------------------------

const MAGIC = "BITNET1\0";

// ---------- base-3 unpacking ------------------------------------------------
// Inverse of the Python packer: digit 0 is least significant, five per byte.
function unpackBase3(bytes, n) {
  const out = new Int8Array(n);
  let k = 0;
  for (let i = 0; i < bytes.length && k < n; i++) {
    let b = bytes[i];
    for (let d = 0; d < 5 && k < n; d++) {
      const r = b % 3;
      out[k++] = r - 1;          // {0,1,2} -> {-1,0,+1}
      b = (b - r) / 3;
    }
  }
  return out;
}

// ---------- model file ------------------------------------------------------
/**
 * @param heap  optional Heap. If given, every tensor is written into the shared
 *              linear memory and `offsets` maps names to float offsets, so the
 *              WASM kernel and the JS engine read the exact same bytes.
 */
export async function loadModel(url, onProgress, heap = null) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${res.status} ${res.statusText}`);

  const total = +res.headers.get("content-length") || 0;
  const chunks = [];
  let got = 0;
  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    if (onProgress) onProgress(got, total);
  }
  const buf = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }

  const magic = new TextDecoder().decode(buf.subarray(0, 8));
  if (magic !== MAGIC) throw new Error(`bad magic ${JSON.stringify(magic)}`);
  const headerLen = new DataView(buf.buffer, 8, 4).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(buf.subarray(12, 12 + headerLen)));
  const blobStart = 12 + headerLen;

  const T = {}, offsets = {}, i8offsets = {};
  const place = (name, n) => {
    if (!heap) return new Float32Array(n);
    const off = heap.alloc(n);
    offsets[name] = off;
    return heap.view(off, n);
  };

  for (const t of header.tensors) {
    const raw = buf.subarray(blobStart + t.offset, blobStart + t.offset + t.nbytes);
    if (t.kind === "ternary") {
      const states = unpackBase3(raw, t.n);
      const data = place(t.name, t.n);
      const sc = t.scale;
      for (let i = 0; i < t.n; i++) data[i] = states[i] * sc;
      if (heap) {
        // the int8 states themselves: what the SIMD kernel reads, a quarter of
        // the bytes of the dense f32 copy
        const bo = heap.allocBytes(t.n);
        heap.i8(bo, t.n).set(states);
        i8offsets[t.name] = bo;
      }
      T[t.name] = { data, shape: t.shape, states, scale: sc };
    } else {
      // aligned copy: subarray offsets are not guaranteed 4-byte aligned
      const copy = new Uint8Array(raw.length);
      copy.set(raw);
      const src = t.kind === "fp32"
        ? new Float32Array(copy.buffer)
        : Float32Array.from(new Uint16Array(copy.buffer), fp16ToFp32);
      const data = place(t.name, src.length);
      if (data !== src) data.set(src);
      T[t.name] = { data, shape: t.shape };
    }
  }
  T["head.weight"] = T["embed.weight"];                  // tied
  offsets["head.weight"] = offsets["embed.weight"];
  i8offsets["head.weight"] = i8offsets["embed.weight"];
  return { tensors: T, offsets, i8offsets, header, bytes: got };
}

function fp16ToFp32(h) {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * Math.pow(2, -24) * f;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * Math.pow(2, e - 25) * (1024 + f);
}

// ---------- primitives ------------------------------------------------------

// RMSNorm, as PyTorch computes it: no mean subtraction, no bias.
function rmsnorm(x, w, out, n, eps = 1e-6) {
  let ss = 0;
  for (let i = 0; i < n; i++) ss += x[i] * x[i];
  const inv = 1 / Math.sqrt(ss / n + eps);
  for (let i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
}

// Per-token absmax int8 activation quantization -- BitNet's A8. Must match
// PyTorch exactly: this is a step function, so being slightly off flips whole
// quantization levels rather than nudging a value.
function actQuant(x, out, n, bits = 8) {
  const qmax = (1 << (bits - 1)) - 1;             // 127
  let amax = 0;
  for (let i = 0; i < n; i++) { const a = x[i] < 0 ? -x[i] : x[i]; if (a > amax) amax = a; }
  const s = amax > 1e-5 ? amax : 1e-5;
  const k = qmax / s, inv = s / qmax;
  for (let i = 0; i < n; i++) {
    let q = Math.round(x[i] * k);
    if (q > qmax) q = qmax; else if (q < -qmax) q = -qmax;
    out[i] = q * inv;
  }
}

// Reference implementation: literal add / subtract / skip, no multiplication by
// any weight. Correct, and SLOW in JavaScript -- with 31% zeros and ~34.5% each
// of +/-1 the branch is unpredictable, so the CPU mispredicts on roughly two
// thirds of 11M iterations per token. Kept because it is the honest expression
// of what ternary weights mean; deliberately not in the hot path.
export function ternaryMatmulReference(x, states, scale, outF, inF, y) {
  for (let o = 0; o < outF; o++) {
    const base = o * inF;
    let acc = 0;
    for (let i = 0; i < inF; i++) {
      const st = states[base + i];
      if (st === 1) acc += x[i];
      else if (st === -1) acc -= x[i];
    }
    y[o] = acc * scale;
  }
  return y;
}

// Eight independent accumulators. A single `acc` serializes on the floating-point
// add's latency; eight let the pipeline overlap them, which is worth more than
// the instruction count suggests.
function dot8(x, W, base, inF) {
  let a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0, i = 0;
  const lim = inF - 7;
  for (; i < lim; i += 8) {
    const b = base + i;
    a0 += x[i]     * W[b];
    a1 += x[i + 1] * W[b + 1];
    a2 += x[i + 2] * W[b + 2];
    a3 += x[i + 3] * W[b + 3];
    a4 += x[i + 4] * W[b + 4];
    a5 += x[i + 5] * W[b + 5];
    a6 += x[i + 6] * W[b + 6];
    a7 += x[i + 7] * W[b + 7];
  }
  let acc = (a0 + a1) + (a2 + a3) + ((a4 + a5) + (a6 + a7));
  for (; i < inF; i++) acc += x[i] * W[base + i];
  return acc;
}

// Hot path: the pre-scaled Float32Array of states*scale, dense and branchless.
// Numerically identical to the reference above, several times faster.
function denseMatmul(x, W, outF, inF, y) {
  for (let o = 0; o < outF; o++) y[o] = dot8(x, W, o * inF, inF);
}

// ---------- the model -------------------------------------------------------
export class BitLM {
  constructor(model, cfg) {
    this.T = model.tensors;
    const c = model.header.inference_config || model.header.config || {};
    this.vocab = c.vocab ?? 4096;
    this.d = c.d ?? 320;
    this.nLayer = c.n_layer ?? 8;
    this.nHead = c.n_head ?? 8;
    this.mult = c.mult ?? 4;
    this.actBits = c.act_bits ?? 8;
    this.cap = cfg?.cap ?? 256;              // trained context = attention window
    this.hd = this.d / this.nHead;
    this.ffn = this.d * this.mult;
    const half = this.hd / 2;

    // RoPE inverse frequencies. Angles are computed per position on the fly --
    // 40 trig calls per token is nothing against 11M MACs -- because absolute
    // position now grows without bound (see the ring buffer note in forward()).
    this.invFreq = new Float32Array(half);
    for (let i = 0; i < half; i++) this.invFreq[i] = 1 / Math.pow(10000, i / half);
    this.rc = new Float32Array(half);
    this.rs = new Float32Array(half);

    // KV cache as a RING BUFFER of `cap` slots.
    //
    // RoPE attention scores depend only on the RELATIVE offset (t - p), never on
    // absolute position. So absolute position may grow to 4,000+ freely, as long
    // as we only attend to the last `cap` tokens -- because then every offset is
    // in 0..cap-1, exactly the range the model was trained on. No re-prefill, no
    // re-rotation of cached keys, no memmove.
    this.kc = [];
    this.vc = [];
    for (let l = 0; l < this.nLayer; l++) {
      this.kc.push(new Float32Array(this.cap * this.d));
      this.vc.push(new Float32Array(this.cap * this.d));
    }

    const D = this.d;
    this.x = new Float32Array(D);
    this.h = new Float32Array(D);
    this.q = new Float32Array(D);
    this.k = new Float32Array(D);
    this.v = new Float32Array(D);
    this.att = new Float32Array(D);
    this.y = new Float32Array(D);
    this.u = new Float32Array(this.ffn);
    this.un = new Float32Array(this.ffn);
    this.scores = new Float32Array(this.cap);
    this.logits = new Float32Array(this.vocab);

    // Two pre-allocated quantization buffers. Calling subarray() per BitLinear
    // allocated a fresh view 48 times per token -- pure GC pressure.
    this.qD = new Float32Array(D);
    this.qF = new Float32Array(this.ffn);

    this.pos = 0;          // absolute position of the next token
    this.nCached = 0;      // valid ring slots
  }

  reset() { this.pos = 0; this.nCached = 0; }

  get contextUsed() { return this.nCached; }

  macsPerToken() {
    return this.nLayer * (4 * this.d * this.d + 2 * this.d * this.ffn)
         + this.d * this.vocab;
  }

  _bit(name, x, outF, inF, y) {
    const buf = inF === this.d ? this.qD : this.qF;
    actQuant(x, buf, inF, this.actBits);
    denseMatmul(buf, this.T[name].data, outF, inF, y);
  }

  _rope(vec, p) {
    const half = this.hd / 2, H = this.nHead, hd = this.hd;
    for (let i = 0; i < half; i++) {
      const th = p * this.invFreq[i];
      this.rc[i] = Math.cos(th);
      this.rs[i] = Math.sin(th);
    }
    for (let hh = 0; hh < H; hh++) {
      const b = hh * hd;
      for (let i = 0; i < half; i++) {
        const c = this.rc[i], s = this.rs[i];
        const a = vec[b + i], d = vec[b + half + i];
        vec[b + i] = a * c - d * s;
        vec[b + half + i] = d * c + a * s;
      }
    }
  }

  /** One token in, logits out. Advances the ring by one position. */
  forward(tokenId) {
    const D = this.d, H = this.nHead, hd = this.hd, ffn = this.ffn, T = this.T;
    const p = this.pos;
    const slot = p % this.cap;

    const E = T["embed.weight"].data;
    const eoff = tokenId * D;
    for (let i = 0; i < D; i++) this.x[i] = E[eoff + i];

    for (let l = 0; l < this.nLayer; l++) {
      const P = "blocks." + l + ".";

      rmsnorm(this.x, T[P + "attn_norm.weight"].data, this.h, D);
      this._bit(P + "q.weight", this.h, D, D, this.q);
      this._bit(P + "k.weight", this.h, D, D, this.k);
      this._bit(P + "v.weight", this.h, D, D, this.v);
      this._rope(this.q, p);
      this._rope(this.k, p);                 // rotated at its ABSOLUTE position
      this.kc[l].set(this.k, slot * D);
      this.vc[l].set(this.v, slot * D);

      const nc = Math.min(p + 1, this.cap);
      const scale = 1 / Math.sqrt(hd);
      const kc = this.kc[l], vc = this.vc[l];
      for (let hh = 0; hh < H; hh++) {
        const qo = hh * hd;
        let mx = -Infinity;
        for (let t = 0; t < nc; t++) {
          const ko = t * D + qo;
          let sc = 0;
          for (let i = 0; i < hd; i++) sc += this.q[qo + i] * kc[ko + i];
          sc *= scale;
          this.scores[t] = sc;
          if (sc > mx) mx = sc;
        }
        let sum = 0;
        for (let t = 0; t < nc; t++) {
          const e = Math.exp(this.scores[t] - mx);
          this.scores[t] = e; sum += e;
        }
        const inv = 1 / sum;
        for (let i = 0; i < hd; i++) this.att[qo + i] = 0;
        for (let t = 0; t < nc; t++) {
          const w = this.scores[t] * inv, vo = t * D + qo;
          for (let i = 0; i < hd; i++) this.att[qo + i] += w * vc[vo + i];
        }
      }

      rmsnorm(this.att, T[P + "subln.weight"].data, this.y, D);      // SubLN
      this._bit(P + "o.weight", this.y, D, D, this.att);
      for (let i = 0; i < D; i++) this.x[i] += this.att[i];          // residual

      rmsnorm(this.x, T[P + "ffn_norm.weight"].data, this.h, D);
      this._bit(P + "up.weight", this.h, ffn, D, this.u);
      for (let i = 0; i < ffn; i++) { const r = this.u[i] > 0 ? this.u[i] : 0; this.u[i] = r * r; }
      rmsnorm(this.u, T[P + "ffn_subln.weight"].data, this.un, ffn);
      this._bit(P + "down.weight", this.un, D, ffn, this.y);
      for (let i = 0; i < D; i++) this.x[i] += this.y[i];            // residual
    }

    rmsnorm(this.x, T["final_norm.weight"].data, this.h, D);
    // tied head is a plain nn.Linear in PyTorch, so NO activation quantization
    denseMatmul(this.h, T["head.weight"].data, this.vocab, D, this.logits);
    this.pos++;
    this.nCached = Math.min(this.pos, this.cap);
    return this.logits;
  }
}

// ---------- repetition control ----------------------------------------------

/**
 * Blocks any token that would complete an n-gram already seen.
 *
 * Necessary because this model was trained on ~227-token stories and has no
 * concept of a longer one. Forced past its natural ending it degenerates into
 * "The end. The end. The end." -- banning the end-of-story token does not make
 * it write more, it makes it unable to stop. This makes looping impossible, so
 * it has to keep finding something new to say.
 */
export class NoRepeatNGram {
  constructor(n = 3) { this.n = n; this.seen = new Map(); this.hist = []; }

  _key(arr, from) { return arr.slice(from, from + this.n - 1).join(","); }

  banned() {
    if (this.hist.length < this.n - 1) return null;
    const k = this._key(this.hist, this.hist.length - (this.n - 1));
    return this.seen.get(k) || null;
  }

  push(tok) {
    this.hist.push(tok);
    if (this.hist.length >= this.n) {
      const k = this._key(this.hist, this.hist.length - this.n);
      let set = this.seen.get(k);
      if (!set) { set = new Set(); this.seen.set(k, set); }
      set.add(tok);
    }
  }
}

/** Frequency penalty over a sliding window of recent tokens. */
export function frequencyPenalty(logits, hist, alpha, window = 128) {
  if (alpha <= 0) return;
  const from = Math.max(0, hist.length - window);
  const count = new Map();
  for (let i = from; i < hist.length; i++)
    count.set(hist[i], (count.get(hist[i]) || 0) + 1);
  for (const [t, c] of count) logits[t] -= alpha * c;
}

// ---------- sampling --------------------------------------------------------
export function sample(logits, temperature = 0.8, topK = 100, rng = Math.random,
                       banned = null) {
  const n = logits.length;
  if (banned) for (const b of banned) logits[b] = -Infinity;

  if (temperature <= 0) {
    let bi = 0;
    for (let i = 1; i < n; i++) if (logits[i] > logits[bi]) bi = i;
    return bi;
  }
  let idx;
  if (topK > 0 && topK < n) {
    idx = new Array(n);
    for (let i = 0; i < n; i++) idx[i] = i;
    idx.sort((a, b) => logits[b] - logits[a]);
    idx = idx.slice(0, topK);
  } else {
    idx = new Array(n);
    for (let i = 0; i < n; i++) idx[i] = i;
  }
  let mx = -Infinity;
  for (const i of idx) if (logits[i] > mx) mx = logits[i];
  let sum = 0;
  const pr = new Float64Array(idx.length);
  for (let j = 0; j < idx.length; j++) {
    const e = Math.exp((logits[idx[j]] - mx) / temperature);
    pr[j] = e; sum += e;
  }
  let r = rng() * sum;
  for (let j = 0; j < idx.length; j++) { r -= pr[j]; if (r <= 0) return idx[j]; }
  return idx[idx.length - 1];
}
