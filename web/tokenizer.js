// ---------------------------------------------------------------------------
// tokenizer.js -- byte-level BPE, encode and decode, from tokenizer.json.
//
// The page has to tokenize the user's prompt, so decode alone is not enough.
// This replicates HF tokenizers' ByteLevel(add_prefix_space=false) + BPE.
// ---------------------------------------------------------------------------

// GPT-2 byte<->unicode table. Bytes that are not printable ASCII get mapped
// into the 256+ range so every byte becomes exactly one BMP character.
// Byte 32 (space) lands on U+0120, which is the "G with dot above" glyph you
// see all over the vocabulary.
function byteMaps() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  }
  const b2u = new Array(256), u2b = new Map();
  for (let i = 0; i < bs.length; i++) {
    const ch = String.fromCharCode(cs[i]);
    b2u[bs[i]] = ch;
    u2b.set(ch, bs[i]);
  }
  return { b2u, u2b };
}

// GPT-2's pre-tokenization pattern. ByteLevel applies this before byte-encoding,
// which is why " the" is one piece and a line-initial "the" is another.
const SPLIT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

const EOT = "<|endoftext|>";

export class Tokenizer {
  constructor(json) {
    this.vocab = json.model.vocab;                       // token string -> id
    this.ids = new Array(Object.keys(this.vocab).length);
    for (const [t, i] of Object.entries(this.vocab)) this.ids[i] = t;

    // merge rank table: "a b" -> priority (lower rank applies first)
    this.ranks = new Map();
    json.model.merges.forEach((m, i) => {
      const [a, b] = Array.isArray(m) ? m : m.split(" ");
      this.ranks.set(a + " " + b, i);
    });

    const { b2u, u2b } = byteMaps();
    this.b2u = b2u;
    this.u2b = u2b;
    this.enc = new TextEncoder();
    this.dec = new TextDecoder("utf-8", { fatal: false });
    this.cache = new Map();
  }

  get size() { return this.ids.length; }

  // Greedy BPE: repeatedly merge the adjacent pair with the lowest rank.
  // This is the same algorithm that produced the merge table during training.
  _bpe(piece) {
    if (this.cache.has(piece)) return this.cache.get(piece);
    const parts = Array.from(piece);
    for (;;) {
      let best = Infinity, bi = -1;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = this.ranks.get(parts[i] + " " + parts[i + 1]);
        if (r !== undefined && r < best) { best = r; bi = i; }
      }
      if (bi < 0) break;
      parts.splice(bi, 2, parts[bi] + parts[bi + 1]);
    }
    this.cache.set(piece, parts);
    return parts;
  }

  encode(text) {
    const out = [];
    for (const m of text.matchAll(SPLIT)) {
      let mapped = "";
      for (const b of this.enc.encode(m[0])) mapped += this.b2u[b];
      for (const tk of this._bpe(mapped)) {
        const id = this.vocab[tk];
        if (id !== undefined) {
          out.push(id);
        } else {
          for (const ch of tk) {              // fall back to single bytes
            const single = this.vocab[ch];
            if (single !== undefined) out.push(single);
          }
        }
      }
    }
    return out;
  }

  decode(ids, keepEot = false) {
    let mapped = "";
    for (const id of ids) {
      const t = this.ids[id];
      if (t === undefined) continue;
      if (t === EOT) { if (keepEot) mapped += "\n"; continue; }
      mapped += t;
    }
    const bytes = new Uint8Array(mapped.length * 2);
    let n = 0;
    for (const ch of mapped) {
      const b = this.u2b.get(ch);
      if (b !== undefined) bytes[n++] = b;
    }
    return this.dec.decode(bytes.subarray(0, n));
  }

  get eotId() {
    const id = this.vocab[EOT];
    return id === undefined ? -1 : id;
  }
}
