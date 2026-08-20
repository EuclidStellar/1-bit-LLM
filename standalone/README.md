# standalone/index.html

One self-contained page. No build step, no dependencies, no server code —
`heap.js`, `bitllm.js`, `tokenizer.js` and `wasmlm.js` from `../web/` are
inlined into the single `<script type="module">` block, so the file is the
whole app.

At runtime it fetches three things over plain HTTPS, all public:

| file | from | size |
|---|---|---|
| `model_packed.bin` | `huggingface.co/euclidstellar/tinystories-1bit-llm` | 2.31 MB |
| `kernel_relaxed.wasm` (falls back to `kernel.wasm`) | same repo | ~7 KB |
| `tokenizer.json` | `huggingface.co/datasets/euclidstellar/tinystories-bpe4096` | ~180 KB |

Nothing is uploaded. Generation happens in the tab.

## Deploy

Drag `index.html` onto <https://app.netlify.com/drop>. That is the entire
deploy — one HTML file needs no `_headers`, no redirects, no config. (Dropping
the whole folder also works, but then `build.py` and this README are served
too.)
Hugging Face serves `resolve/main/` with `Access-Control-Allow-Origin: *`,
so the fetches work from any origin, and the WASM is instantiated from an
`ArrayBuffer` rather than streamed, so the host's MIME types don't matter.

Netlify hands back a `random-name.netlify.app` URL; rename it under
Site configuration → Change site name.

## Regenerating it

`standalone/build.py` re-inlines from `../web/`, so edits belong in the
module files, not here.
