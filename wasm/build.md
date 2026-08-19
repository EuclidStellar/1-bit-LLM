# Building kernel.wasm

Apple clang has no wasm32 target, so this is compiled on Kaggle (Linux, full
LLVM) and the resulting `kernel.wasm` is committed to the repo.

```
clang --target=wasm32 -O3 -msimd128 -nostdlib -ffreestanding \
      -Wl,--no-entry -Wl,--import-memory -Wl,--export-dynamic \
      -o kernel.wasm kernel.c
```

`--import-memory` matters: JavaScript owns the heap layout, so weights, the KV
cache and all scratch live in one `WebAssembly.Memory` that JS allocates and
these functions take raw pointers into. Nothing is copied per call.

If clang lacks the wasm target, `pip install ziglang` provides a full one:

```
python -m ziglang cc --target=wasm32-freestanding -O3 -msimd128 \
       -nostdlib -Wl,--no-entry -Wl,--import-memory -Wl,--export-dynamic \
       -o kernel.wasm kernel.c
```
