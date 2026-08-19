// ---------------------------------------------------------------------------
// heap.js -- one linear memory shared by the WASM kernel and the JS fallback.
//
// The whole point: weights, the KV cache and every scratch buffer live in a
// single WebAssembly.Memory. The WASM kernel reads them as raw pointers and the
// JS engine reads the same bytes through Float32Array views. Nothing is copied
// per call, and both engines are provably looking at identical data -- which is
// what makes verifying one against the other meaningful.
// ---------------------------------------------------------------------------

const PAGE = 65536;                       // WASM page size, 64 KB

export class Heap {
  /** @param {number} floats how many f32 slots to reserve */
  constructor(floats) {
    const pages = Math.ceil((floats * 4) / PAGE) + 24;    // slack for the stack
    this.memory = new WebAssembly.Memory({ initial: pages });
    this.f32 = new Float32Array(this.memory.buffer);
    this.cap = this.f32.length;
    this.pages = pages;
    // Start well above 0: wasm-ld places the module's shadow stack low in
    // linear memory, and allocating from 0 would let JS scribble over it.
    // Raised to the real value once the module reports __heap_base.
    this.top = 1 << 18;                                   // 256 K floats = 1 MB
  }

  /** Move the allocation floor above whatever the WASM module reserves. */
  setBase(heapBaseBytes) {
    if (typeof heapBaseBytes === "number" && heapBaseBytes > 0) {
      const floors = Math.ceil(heapBaseBytes / 4) + 4096;  // 16 KB of padding
      if (floors > this.top) this.top = floors;
    }
    this.top = (this.top + 3) & ~3;
  }

  alloc(n) {
    const off = this.top;
    let t = off + n;
    t = (t + 3) & ~3;                     // keep 16-byte alignment for SIMD
    if (t > this.cap)
      throw new Error(`heap overflow: need ${t} floats, have ${this.cap}`);
    this.top = t;
    return off;
  }

  /** A Float32Array view -- shares bytes with WASM, no copy. */
  view(off, n) { return this.f32.subarray(off, off + n); }

  /** Byte pointer, for passing to WASM. */
  ptr(off) { return off * 4; }

  get usedMB() { return (this.top * 4) / 1e6; }
  get totalMB() { return (this.cap * 4) / 1e6; }
}

/** Floats needed for a given model shape, so the heap is sized exactly once. */
export function heapFloatsFor({ vocab = 4096, d = 320, nLayer = 8, mult = 4,
                                cap = 256 } = {}) {
  const ffn = d * mult;
  const weights = vocab * d                          // embedding, tied with head
                + nLayer * (4 * d * d + 2 * d * ffn) // q,k,v,o,up,down
                + nLayer * (3 * d + ffn) + d;        // norms
  const kv = nLayer * 2 * cap * d;
  const scratch = 7 * d + 3 * ffn + vocab + cap + 4096;
  return weights + kv + scratch;
}
