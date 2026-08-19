// ---------------------------------------------------------------------------
// bitllm.js -- an 11M-parameter BitNet b1.58 model, in the browser, in plain JS.
//
// Reads model_packed.bin directly: base-3 packed ternary weights, five per byte.
// No PyTorch, no ONNX, no WASM. Stage 1 is written for CORRECTNESS -- it is
// verified logit-by-logit against the PyTorch reference before it is trusted.
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
export async function loadModel(url, onProgress) {
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

  const T = {};             // name -> {data: Float32Array, shape, states?}
  for (const t of header.tensors) {
    const raw = buf.subarray(blobStart + t.offset, blobStart + t.offset + t.nbytes);
    if (t.kind === "ternary") {
      const states = unpackBase3(raw, t.n);
      const data = new Float32Array(t.n);
      const s = t.scale;
      for (let i = 0; i < t.n; i++) data[i] = states[i] * s;
      T[t.name] = { data, shape: t.shape, states, scale: s };
    } else {
      // aligned copy: subarray offsets are not guaranteed 4-byte aligned
      const copy = new Uint8Array(raw.length);
      copy.set(raw);
      const data = t.kind === "fp32"
        ? new Float32Array(copy.buffer)
        : Float32Array.from(new Uint16Array(copy.buffer), fp16ToFp32);
      T[t.name] = { data, shape: t.shape };
    }
  }
  T["head.weight"] = T["embed.weight"];          // tied
  return { tensors: T, header, bytes: got };
}

function fp16ToFp32(h) {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * Math.pow(2, -24) * f;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * Math.pow(2, e - 25) * (1024 + f);
}

// ---------- primitives ------------------------------------------------------

// RMSNorm, computed the way PyTorch does it: fp32, no mean subtraction, no bias.
function rmsnorm(x, w, out, eps = 1e-6) {
  let ss = 0;
  for (let i = 0; i < x.length; i++) ss += x[i] * x[i];
  const inv = 1 / Math.sqrt(ss / x.length + eps);
  for (let i = 0; i < x.length; i++) out[i] = x[i] * inv * w[i];
  return out;
}

// Per-token absmax int8 activation quantization -- BitNet's A8. Must be
// replicated exactly or logits diverge: this is a step function, so being
// slightly off flips whole quantization levels.
function actQuant(x, out, bits = 8) {
  const qmax = (1 << (bits - 1)) - 1;             // 127
  let amax = 0;
  for (let i = 0; i < x.length; i++) { const a = Math.abs(x[i]); if (a > amax) amax = a; }
  const s = Math.max(amax, 1e-5);
  const k = qmax / s, inv = s / qmax;
  for (let i = 0; i < x.length; i++) {
    let q = Math.round(x[i] * k);
    if (q > qmax) q = qmax; else if (q < -qmax) q = -qmax;
    out[i] = q * inv;
  }
  return out;
}

// Reference implementation: literal add / subtract / skip, no multiplication by
// any weight. Correct, and SLOW in JavaScript -- with 31% zeros and ~34.5% each
// of +/-1 the branch is essentially unpredictable, so the CPU mispredicts on
// roughly two thirds of 11 million iterations per token. Kept because it is the
// honest expression of what ternary weights mean; not used in the hot path.
export function ternaryMatmulReference(x, states, scale, outF, inF, y) {
  for (let o = 0; o < outF; o++) {
    const base = o * inF;
    let acc = 0;
    for (let i = 0; i < inF; i++) {
      const s = states[base + i];
      if (s === 1) acc += x[i];
      else if (s === -1) acc -= x[i];
    }
    y[o] = acc * scale;
  }
  return y;
}

// Four independent accumulators. One `acc` serializes on the add's latency;
// four let the pipeline overlap them, which is worth more than it looks.
function dot4(x, W, base, inF) {
  let a0 = 0, a1 = 0, a2 = 0, a3 = 0, i = 0;
  const lim = inF - 3;
  for (; i < lim; i += 4) {
    a0 += x[i]     * W[base + i];
    a1 += x[i + 1] * W[base + i + 1];
    a2 += x[i + 2] * W[base + i + 2];
    a3 += x[i + 3] * W[base + i + 3];
  }
  let acc = a0 + a1 + a2 + a3;
  for (; i < inF; i++) acc += x[i] * W[base + i];
  return acc;
}

// Hot path: the pre-scaled Float32Array of states*scale, dense and branchless.
// Numerically identical to the reference above, several times faster.
function denseMatmul(x, W, outF, inF, y) {
  for (let o = 0; o < outF; o++) y[o] = dot4(x, W, o * inF, inF);
  return y;
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
    this.maxT = cfg?.maxT ?? 256;
    this.hd = this.d / this.nHead;
    this.ffn = this.d * this.mult;

    // RoPE tables
    const half = this.hd / 2;
    this.cos = new Float32Array(this.maxT * half);
    this.sin = new Float32Array(this.maxT * half);
    for (let p = 0; p < this.maxT; p++)
      for (let i = 0; i < half; i++) {
        const th = p / Math.pow(10000, i / half);
        this.cos[p * half + i] = Math.cos(th);
        this.sin[p * half + i] = Math.sin(th);
      }

    // KV cache and scratch
    this.kc = [], this.vc = [];
    for (let l = 0; l < this.nLayer; l++) {
      this.kc.push(new Float32Array(this.maxT * this.d));
      this.vc.push(new Float32Array(this.maxT * this.d));
    }
    const D = this.d;
    this.x = new Float32Array(D); this.h = new Float32Array(D);
    this.xq = new Float32Array(Math.max(D, this.ffn));
    this.q = new Float32Array(D); this.k = new Float32Array(D);
    this.v = new Float32Array(D); this.att = new Float32Array(D);
    this.y = new Float32Array(D); this.u = new Float32Array(this.ffn);
    this.un = new Float32Array(this.ffn);
    this.scores = new Float32Array(this.maxT);
    this.logits = new Float32Array(this.vocab);
    this.pos = 0;
  }

  reset() { this.pos = 0; }

  _bit(name, x, outF, inF, y) {
    const t = this.T[name];
    actQuant(x, this.xq.subarray(0, inF), this.actBits);
    return denseMatmul(this.xq, t.data, outF, inF, y);
  }

  /** MACs per generated token, for reporting an honest throughput figure. */
  macsPerToken() {
    const D = this.d, F = this.ffn;
    return this.nLayer * (4 * D * D + 2 * D * F) + D * this.vocab;
  }

  _rope(vec, p) {
    const half = this.hd / 2;
    for (let hh = 0; hh < this.nHead; hh++) {
      const b = hh * this.hd;
      for (let i = 0; i < half; i++) {
        const c = this.cos[p * half + i], s = this.sin[p * half + i];
        const a = vec[b + i], d = vec[b + half + i];
        vec[b + i] = a * c - d * s;
        vec[b + half + i] = d * c + a * s;
      }
    }
  }

  /** One token in, logits out. Advances the KV cache by one position. */
  forward(tokenId) {
    const { d: D, nHead: H, hd, ffn, T } = this;
    const p = this.pos;
    if (p >= this.maxT) throw new Error("context full");

    // embedding lookup (full precision, never quantized)
    const E = T["embed.weight"].data;
    for (let i = 0; i < D; i++) this.x[i] = E[tokenId * D + i];

    for (let l = 0; l < this.nLayer; l++) {
      const P = `blocks.${l}.`;

      rmsnorm(this.x, T[P + "attn_norm.weight"].data, this.h);
      this._bit(P + "q.weight", this.h, D, D, this.q);
      this._bit(P + "k.weight", this.h, D, D, this.k);
      this._bit(P + "v.weight", this.h, D, D, this.v);
      this._rope(this.q, p);
      this._rope(this.k, p);
      this.kc[l].set(this.k, p * D);
      this.vc[l].set(this.v, p * D);

      // causal attention over the cache, per head
      const scale = 1 / Math.sqrt(hd);
      const kc = this.kc[l], vc = this.vc[l];
      for (let hh = 0; hh < H; hh++) {
        const qo = hh * hd;
        let mx = -Infinity;
        for (let t = 0; t <= p; t++) {
          const ko = t * D + qo;
          let s = 0;
          for (let i = 0; i < hd; i++) s += this.q[qo + i] * kc[ko + i];
          s *= scale;
          this.scores[t] = s;
          if (s > mx) mx = s;
        }
        let sum = 0;
        for (let t = 0; t <= p; t++) { const e = Math.exp(this.scores[t] - mx); this.scores[t] = e; sum += e; }
        const inv = 1 / sum;
        for (let i = 0; i < hd; i++) this.att[qo + i] = 0;
        for (let t = 0; t <= p; t++) {
          const w = this.scores[t] * inv, vo = t * D + qo;
          for (let i = 0; i < hd; i++) this.att[qo + i] += w * vc[vo + i];
        }
      }

      rmsnorm(this.att, T[P + "subln.weight"].data, this.y);   // SubLN
      this._bit(P + "o.weight", this.y, D, D, this.att);
      for (let i = 0; i < D; i++) this.x[i] += this.att[i];    // residual

      rmsnorm(this.x, T[P + "ffn_norm.weight"].data, this.h);
      this._bit(P + "up.weight", this.h, ffn, D, this.u);
      for (let i = 0; i < ffn; i++) { const r = this.u[i] > 0 ? this.u[i] : 0; this.u[i] = r * r; }
      rmsnorm(this.u, T[P + "ffn_subln.weight"].data, this.un);
      this._bit(P + "down.weight", this.un, D, ffn, this.y);
      for (let i = 0; i < D; i++) this.x[i] += this.y[i];      // residual
    }

    rmsnorm(this.x, T["final_norm.weight"].data, this.h);
    // tied head: plain nn.Linear in PyTorch, so NO activation quantization here
    denseMatmul(this.h, T["head.weight"].data, this.vocab, D, this.logits);
    this.pos++;
    return this.logits;
  }
}

// ---------- sampling --------------------------------------------------------
export function sample(logits, temperature = 0.8, topK = 100, rng = Math.random) {
  const n = logits.length;
  if (temperature <= 0) {
    let bi = 0;
    for (let i = 1; i < n; i++) if (logits[i] > logits[bi]) bi = i;
    return bi;
  }
  const idx = topK > 0 && topK < n
    ? Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]).slice(0, topK)
    : Array.from(logits.keys());
  let mx = -Infinity;
  for (const i of idx) if (logits[i] > mx) mx = logits[i];
  let sum = 0;
  const pr = idx.map(i => { const e = Math.exp((logits[i] - mx) / temperature); sum += e; return e; });
  let r = rng() * sum;
  for (let j = 0; j < idx.length; j++) { r -= pr[j]; if (r <= 0) return idx[j]; }
  return idx[idx.length - 1];
}
