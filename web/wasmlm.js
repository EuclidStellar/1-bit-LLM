// ---------------------------------------------------------------------------
// wasmlm.js -- the same forward pass, with the inner loops in SIMD128 WASM.
//
// Control flow is identical to bitllm.js (which is verified against PyTorch);
// only the loops move. Both engines read the same Heap, so comparing their
// logits is a real test rather than a coincidence of shared code.
//
// RoPE and softmax stay in JavaScript: RoPE needs trig, softmax needs exp, and
// together they are a low single-digit percentage of the work.
// ---------------------------------------------------------------------------

/**
 * Try each kernel URL in order and keep the first that instantiates.
 *
 * kernel_relaxed.wasm uses relaxed-SIMD f32x4_relaxed_madd, which halves the
 * instruction count in every dot product. Not every browser enables relaxed
 * SIMD, and a module using it fails to instantiate where it is unsupported --
 * hence the ordered fallback rather than a feature test.
 */
export async function loadBestKernel(urls, heap) {
  const errors = [];
  for (const url of urls) {
    try {
      return { ...(await loadKernel(url, heap)), url };
    } catch (e) {
      errors.push(`${url.split("/").pop()}: ${e.message}`);
    }
  }
  throw new Error("no kernel loaded -- " + errors.join(" | "));
}

export async function loadKernel(url, heap) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`kernel fetch ${res.status}`);
  const bytes = await res.arrayBuffer();
  const { instance } = await WebAssembly.instantiate(bytes,
    { env: { memory: heap.memory } });

  const need = ["matvec", "rmsnorm", "act_quant", "add_inplace",
                "relu2", "gather_row"];
  const missing = need.filter(n => typeof instance.exports[n] !== "function");
  if (missing.length) throw new Error("kernel missing exports: " + missing);

  // wasm-ld puts the module's shadow stack low in linear memory; move the JS
  // allocator above it before anything is allocated.
  const hb = instance.exports.__heap_base;
  heap.setBase(typeof hb === "object" ? hb.value : hb);

  return { K: instance.exports, size: bytes.byteLength,
           caps: { i8: typeof instance.exports.matvec_i8 === "function",
                   fusedAttn: typeof instance.exports.attn_head === "function" } };
}

export class WasmBitLM {
  /**
   * @param model  result of loadModel(url, onProgress, heap) -- offsets, not arrays
   * @param heap   the shared Heap
   * @param K      kernel exports from loadKernel
   */
  constructor(model, heap, K, cfg) {
    const c = model.header.inference_config || model.header.config || {};
    this.heap = heap; this.K = K; this.O = model.offsets;
    this.I = model.i8offsets || {};
    this.SC = {};
    for (const tt of model.header.tensors)
      if (tt.kind === "ternary") this.SC[tt.name] = tt.scale;
    this.SC["head.weight"] = this.SC["embed.weight"];
    // int8 states are a quarter of the bytes of the dense f32 copy, and weight
    // streaming is the bottleneck at this size
    this.useI8 = typeof K.matvec_i8 === "function"
              && Object.keys(this.I).length > 0;
    this.fused = typeof K.attn_head === "function";
    this.vocab = c.vocab ?? 4096;
    this.d = c.d ?? 320;
    this.nLayer = c.n_layer ?? 8;
    this.nHead = c.n_head ?? 8;
    this.mult = c.mult ?? 4;
    this.cap = cfg?.cap ?? 256;
    this.hd = this.d / this.nHead;
    this.ffn = this.d * this.mult;
    const D = this.d, F = this.ffn, half = this.hd / 2;

    // RoPE frequencies. Angles are recomputed per position because absolute
    // position grows without bound -- see the ring-buffer note in forward().
    this.invFreq = new Float32Array(half);
    for (let i = 0; i < half; i++) this.invFreq[i] = 1 / Math.pow(10000, i / half);
    this.rc = new Float32Array(half);
    this.rs = new Float32Array(half);

    // everything below lives in the shared heap
    const A = n => heap.alloc(n);
    this.oX = A(D); this.oH = A(D); this.oQ = A(D); this.oK = A(D);
    this.oV = A(D); this.oAtt = A(D); this.oY = A(D);
    this.oU = A(F); this.oUn = A(F);
    this.oQD = A(D); this.oQF = A(F);
    this.oLog = A(this.vocab); this.oSc = A(this.cap);

    this.oKC = []; this.oVC = [];
    for (let l = 0; l < this.nLayer; l++) {
      this.oKC.push(A(this.cap * D));
      this.oVC.push(A(this.cap * D));
    }

    this.f32 = heap.f32;
    this.vQ = heap.view(this.oQ, D);        // RoPE works on views
    this.vK = heap.view(this.oK, D);
    this.vSc = heap.view(this.oSc, this.cap);
    this.logits = heap.view(this.oLog, this.vocab);

    this.pos = 0;
    this.nCached = 0;
  }

  reset() { this.pos = 0; this.nCached = 0; }
  get contextUsed() { return this.nCached; }

  macsPerToken() {
    return this.nLayer * (4 * this.d * this.d + 2 * this.d * this.ffn)
         + this.d * this.vocab;
  }

  /** RoPE angles depend only on the position, which is fixed for the whole
   *  token -- so compute them once per forward, not once per layer. */
  _ropeTables(p) {
    const half = this.hd / 2;
    for (let i = 0; i < half; i++) {
      const th = p * this.invFreq[i];
      this.rc[i] = Math.cos(th);
      this.rs[i] = Math.sin(th);
    }
  }

  _rope(vec) {
    const half = this.hd / 2, H = this.nHead, hd = this.hd;
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

  /** One matvec, from int8 states when the kernel supports it. */
  _mv(name, srcOff, dstOff, outF, inF) {
    if (this.useI8)
      this.K.matvec_i8(srcOff * 4, this.I[name], dstOff * 4, outF, inF,
                       this.SC[name]);
    else
      this.K.matvec(srcOff * 4, this.O[name] * 4, dstOff * 4, outF, inF);
  }

  forward(tokenId) {
    const K = this.K, O = this.O, P = o => o * 4;
    const D = this.d, H = this.nHead, hd = this.hd, F = this.ffn;
    const p = this.pos, slot = p % this.cap;

    if (this.useI8 && typeof K.gather_row_i8 === "function")
      K.gather_row_i8(this.I["embed.weight"], P(this.oX), tokenId, D,
                      this.SC["embed.weight"]);
    else
      K.gather_row(P(O["embed.weight"]), P(this.oX), tokenId, D);
    this._ropeTables(p);
    const hasFused = this.fused;

    for (let l = 0; l < this.nLayer; l++) {
      const B = "blocks." + l + ".";

      K.rmsnorm(P(this.oX), P(O[B + "attn_norm.weight"]), P(this.oH), D, 1e-6);

      K.act_quant(P(this.oH), P(this.oQD), D);
      this._mv(B + "q.weight", this.oQD, this.oQ, D, D);
      this._mv(B + "k.weight", this.oQD, this.oK, D, D);
      this._mv(B + "v.weight", this.oQD, this.oV, D, D);

      this._rope(this.vQ);
      this._rope(this.vK);                 // rotated at its ABSOLUTE position
      this.f32.copyWithin(this.oKC[l] + slot * D, this.oK, this.oK + D);
      this.f32.copyWithin(this.oVC[l] + slot * D, this.oV, this.oV + D);

      // Ring buffer: RoPE scores depend only on the relative offset (t - p), so
      // absolute position may grow without bound as long as attention reaches
      // back at most `cap` tokens -- every offset then stays in the trained range.
      const nc = Math.min(p + 1, this.cap);
      const scale = 1 / Math.sqrt(hd);
      for (let hh = 0; hh < H; hh++) {
        const qo = hh * hd;
        if (hasFused) {
          // scores + softmax + weighted sum in one call, no JS exp
          K.attn_head(P(this.oQ), P(this.oKC[l]), P(this.oVC[l]), P(this.oAtt),
                      P(this.oSc), nc, D, qo, hd, scale);
        } else {
          K.attn_scores(P(this.oQ), P(this.oKC[l]), P(this.oSc),
                        nc, D, qo, hd, scale);
          let mx = -Infinity;
          for (let t = 0; t < nc; t++) if (this.vSc[t] > mx) mx = this.vSc[t];
          let sum = 0;
          for (let t = 0; t < nc; t++) {
            const e = Math.exp(this.vSc[t] - mx);
            this.vSc[t] = e; sum += e;
          }
          const inv = 1 / sum;
          for (let t = 0; t < nc; t++) this.vSc[t] *= inv;
          K.attn_mix(P(this.oSc), P(this.oVC[l]), P(this.oAtt), nc, D, qo, hd);
        }
      }

      K.rmsnorm(P(this.oAtt), P(O[B + "subln.weight"]), P(this.oY), D, 1e-6);
      K.act_quant(P(this.oY), P(this.oQD), D);
      this._mv(B + "o.weight", this.oQD, this.oAtt, D, D);
      K.add_inplace(P(this.oX), P(this.oAtt), D);

      K.rmsnorm(P(this.oX), P(O[B + "ffn_norm.weight"]), P(this.oH), D, 1e-6);
      K.act_quant(P(this.oH), P(this.oQD), D);
      this._mv(B + "up.weight", this.oQD, this.oU, F, D);
      K.relu2(P(this.oU), F);
      K.rmsnorm(P(this.oU), P(O[B + "ffn_subln.weight"]), P(this.oUn), F, 1e-6);
      K.act_quant(P(this.oUn), P(this.oQF), F);
      this._mv(B + "down.weight", this.oQF, this.oY, D, F);
      K.add_inplace(P(this.oX), P(this.oY), D);
    }

    K.rmsnorm(P(this.oX), P(O["final_norm.weight"]), P(this.oH), D, 1e-6);
    // tied head is a plain nn.Linear in PyTorch: no activation quantization
    this._mv("embed.weight", this.oH, this.oLog, this.vocab, D);

    this.pos++;
    this.nCached = Math.min(this.pos, this.cap);
    return this.logits;
  }
}
