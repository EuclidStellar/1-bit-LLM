# ---------------------------------------------------------------------------
# Compile wasm/kernel.c to SIMD128 WebAssembly, on Kaggle.
#
# Why here and not locally: Apple clang has no wasm32 target ("No available
# targets are compatible with triple wasm32"), and Kaggle has no clang at all.
# ziglang ships a complete LLVM with wasm support in one pip install, which is
# the most reliable path on either machine.
#
# Produces TWO kernels. Relaxed SIMD adds f32x4_relaxed_madd -- a fused
# multiply-add, one instruction where base SIMD128 needs a mul and an add, which
# halves the instruction count in every dot product. A module using it FAILS TO
# INSTANTIATE where the browser lacks support, so the page tries relaxed first
# and falls back rather than feature-testing. That fallback was worth 900 -> 1045
# tok/s on Chrome.
#
# Both compiled kernels are committed at wasm/kernel.wasm and
# wasm/kernel_relaxed.wasm, so this only needs re-running if kernel.c changes.
# ---------------------------------------------------------------------------

import os, sys, subprocess

D = "/kaggle/working/wasmbuild"
subprocess.run(["rm", "-rf", D], check=False)
subprocess.run(["git", "clone", "-q",
                "https://github.com/EuclidStellar/1-bit-LLM.git", D], check=True)
SRC = f"{D}/wasm/kernel.c"

print("installing ziglang ...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ziglang"], check=True)

BASE = [sys.executable, "-m", "ziglang", "cc", "-target", "wasm32-freestanding",
        "-O3", "-nostdlib",
        "-Wl,--no-entry",          # a library, not a program
        "-Wl,--import-memory",     # JS owns the heap: weights, KV cache, scratch
        "-Wl,--export-dynamic"]    # honour __attribute__((export_name(...)))

TARGETS = [("/kaggle/working/kernel.wasm",         ["-msimd128"]),
           ("/kaggle/working/kernel_relaxed.wasm", ["-msimd128", "-mrelaxed-simd",
                                                    "-DUSE_RELAXED"])]

for out, extra in TARGETS:
    if os.path.exists(out):
        os.remove(out)
    r = subprocess.run(BASE + extra + ["-o", out, SRC], capture_output=True, text=True)
    ok = r.returncode == 0 and os.path.exists(out)
    print(f"{os.path.basename(out):<24} rc={r.returncode} {'OK' if ok else 'FAILED'}"
          + (f"  {os.path.getsize(out):,} bytes" if ok else ""))
    if not ok:
        print("   " + r.stderr.strip()[:800])

# Verify the exports, because a missing one degrades SILENTLY: wasmlm.js checks
# `typeof K.attn_head === "function"` and quietly uses the slower path if absent.
# That cost a full round of thinking a change had done nothing.
b = open("/kaggle/working/kernel.wasm", "rb").read()
assert b[:4] == b"\x00asm", "not a wasm module"
for name in ["matvec", "matvec_i8", "attn_head", "gather_row", "gather_row_i8",
             "rmsnorm", "act_quant", "add_inplace", "relu2"]:
    print(f"  export {name:<15} {'present' if name.encode() in b else 'MISSING'}")
print(f"  imports memory  {'yes' if b'memory' in b else 'NO'}")

# A NOTE ON -nostdlib: __builtin_roundf emits a call to roundf, which never
# resolves without libm -- the first build failed on exactly that. WASM has
# f32x4.nearest as a real instruction, so kernel.c splats-rounds-extracts
# instead. Incidental win: f32x4.nearest is roundTiesToEven, matching
# torch.round, whereas JavaScript's Math.round is ties-away-from-zero.

from kaggle_secrets import UserSecretsClient
from huggingface_hub import HfApi
api = HfApi(token=UserSecretsClient().get_secret("hf_token"))
for f in ["kernel.wasm", "kernel_relaxed.wasm"]:
    p = f"/kaggle/working/{f}"
    if os.path.exists(p):
        api.upload_file(path_or_fileobj=p, path_in_repo=f,
                        repo_id=f"{api.whoami()['name']}/tinystories-1bit-llm")
        print("uploaded", f)
