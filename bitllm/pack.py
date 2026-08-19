"""
Bit-packing ternary weights, and an integer inference path.

This is the module that turns the project's headline number from arithmetic into
a file. Everything before this stores fp32 master weights and merely *computes*
with ternary values; here the ternary states are written as actual bits.

Packing scheme: **base 3, five weights per byte.** 3^5 = 243 <= 255, so five
ternary digits fit in one uint8 with 12 codes spare. That is 8/5 = 1.6 bits per
weight against the log2(3) = 1.585 information-theoretic floor -- 99.1%
efficient. The obvious alternative, 2 bits per weight packed 4-per-byte, wastes
one of four codes and costs 2.0 bits/weight (79% efficient).

CRITICAL: pack from `quantize_weight(w)` states, never from the STE output.
`ste(w, q)` equals `q` in exact arithmetic but not in fp32 -- counting unique
values on an STE output gives 7, not 3, because `q - w` rounds and then adding
`w` rounds again.
"""

import json
import struct

import numpy as np
import torch

MAGIC = b"BITNET1\0"
POW3 = np.array([1, 3, 9, 27, 81], dtype=np.uint16)


# --------------------------------------------------------------------------
# ternary states
# --------------------------------------------------------------------------

def ternary_states(w: torch.Tensor):
    """Split a weight tensor into int8 states in {-1,0,+1} and one fp32 scale.

    This is exactly `quant.weight_ternary` factored so the states and the scale
    can be stored separately: the states become packed bits, the scale becomes
    four bytes per tensor.
    """
    g = w.detach().abs().mean().clamp(min=1e-5)
    states = (w.detach() / g).round().clamp(-1, 1).to(torch.int8)
    return states, float(g)


# --------------------------------------------------------------------------
# base-3 packing
# --------------------------------------------------------------------------

def pack_base3(states: torch.Tensor):
    """Pack int8 states in {-1,0,+1} into uint8, five per byte.

    Returns (packed uint8 array, n_values, pad). Shifting to {0,1,2} first keeps
    the digits unsigned; the maximum encodable byte is 2*(1+3+9+27+81) = 242.
    """
    flat = (states.reshape(-1).to(torch.int16) + 1).cpu().numpy().astype(np.uint16)
    assert flat.min() >= 0 and flat.max() <= 2, "states must be in {-1,0,+1}"
    n = flat.size
    pad = (-n) % 5
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint16)])
    packed = (flat.reshape(-1, 5) * POW3).sum(axis=1).astype(np.uint8)
    return packed, n, pad


def unpack_base3(packed: np.ndarray, n: int) -> np.ndarray:
    """Inverse of pack_base3. Returns int8 states in {-1,0,+1}, length n."""
    b = packed.astype(np.uint16).copy()
    digits = np.empty((b.size, 5), dtype=np.int8)
    for i in range(5):
        digits[:, i] = (b % 3).astype(np.int8)
        b //= 3
    return digits.reshape(-1)[:n] - 1


# --------------------------------------------------------------------------
# file format
# --------------------------------------------------------------------------

def _is_quantized_bitlinear(mod):
    """Duck-typed check, deliberately NOT isinstance().

    isinstance() compares class *identity*, and re-importing bitllm after
    `del sys.modules[...]` creates a brand-new BitLinear class object. A model
    built before the reload holds instances of the old one, so isinstance
    returns False for every module -- silently. That bug stored 9,830,400 body
    weights as fp16 instead of packing them, producing a 19.97 MB file instead
    of 2.27 MB with no error at all.
    """
    return (hasattr(mod, "weight")
            and getattr(mod, "weight_mode", "none") not in (None, "none"))


def inference_config(train_config):
    """Config for a model that CONSUMES packed weights.

    The packed file already holds quantized weights, so the inference model must
    NOT quantize again: `weight_mode` and `embed_mode` become "none" while
    `act_bits` is preserved, because activation quantization happens at runtime
    and is part of the computation the QAT model was trained with.

    Re-quantizing an already-ternary tensor leaves the states unchanged but
    recomputes the scale as mean(|states*g|) = g*(1 - zero_fraction) ~ 0.686g,
    shrinking every weight by ~31%.
    """
    c = dict(train_config)
    c["weight_mode"] = "none"
    c["embed_mode"] = "none"
    c.setdefault("act_bits", 8)
    return c


def save_packed(model, path, config=None, meta=None, full_dtype="fp32"):
    """Write a self-describing packed model.

        MAGIC (8 bytes) | header_len uint32 | header JSON | blobs

    Ternary tensors store packed bits plus one fp32 scale. Everything else --
    the normalization vectors, and the embedding if it was not trained ternary
    -- stores at `full_dtype`.

    `full_dtype` defaults to **fp32**. At fp16 the 18,240 norm parameters carry
    ~5e-4 relative error, which compounds through 41 norm layers and is amplified
    by the LM head: measured max logit deviation 4.0e-3 from the source model
    even with activation quantization disabled. fp32 costs 36 KB on a 2.3 MB file
    (+1.6%) and makes the packed model numerically exact. The norms are 0.16% of
    parameters; exactness is worth more than 1.6% of the file.
    """
    tensors, blobs, offset = [], [], 0
    quantized_names = set()
    for name, mod in model.named_modules():
        if _is_quantized_bitlinear(mod):
            quantized_names.add(name + ".weight")
    assert quantized_names, ("no quantized modules found -- refusing to write a "
                            "file that claims to be packed but is not")
    if getattr(model, "embed_mode", "none") == "ternary":
        quantized_names.add("embed.weight")

    for name, t in model.state_dict().items():
        if name == "head.weight":
            continue                              # tied to embed.weight
        if name in quantized_names:
            states, scale = ternary_states(t)
            packed, n, pad = pack_base3(states)
            raw = packed.tobytes()
            tensors.append({"name": name, "shape": list(t.shape), "kind": "ternary",
                            "scale": scale, "n": n, "pad": pad,
                            "offset": offset, "nbytes": len(raw)})
        else:
            np_dt = {"fp16": torch.float16, "fp32": torch.float32}[full_dtype]
            raw = t.detach().cpu().to(np_dt).numpy().tobytes()
            tensors.append({"name": name, "shape": list(t.shape),
                            "kind": full_dtype,
                            "offset": offset, "nbytes": len(raw)})
        blobs.append(raw)
        offset += len(raw)

    ITEM = {"ternary": 0, "fp16": 2, "fp32": 4}
    n_tern = sum(t["n"] for t in tensors if t["kind"] == "ternary")
    n_full = sum(t["nbytes"] // ITEM[t["kind"]]
                 for t in tensors if t["kind"] != "ternary")
    header = json.dumps({"tensors": tensors, "format_version": 2,
                         "full_dtype": full_dtype,
                         "config": config or {},
                         "inference_config": inference_config(config or {}),
                         "meta": meta or {}, "tied_head": True,
                         "counts": {"ternary_weights": n_tern,
                                    "full_precision_weights": n_full,
                                    "full_dtype": full_dtype}}).encode()
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(header)))
        f.write(header)
        for b in blobs:
            f.write(b)
    return {"path": path, "header_bytes": len(header) + 12,
            "blob_bytes": offset, "total_bytes": len(header) + 12 + offset,
            "ternary_weights": n_tern, "full_precision_weights": n_full,
            "full_dtype": full_dtype,
            "packed_tensors": len(quantized_names),
            "bits_per_ternary_weight": round(
                sum(t["nbytes"] for t in tensors if t["kind"] == "ternary")
                * 8 / max(n_tern, 1), 4)}


def load_packed(path):
    """Read a packed model back. Returns (state_dict, header).

    Ternary tensors are reconstructed as `states * scale`, which reproduces the
    quantized weight exactly -- so a model loaded from this file computes bit-
    identically to the QAT model it came from.
    """
    with open(path, "rb") as f:
        assert f.read(8) == MAGIC, "not a BITNET1 file"
        hlen = struct.unpack("<I", f.read(4))[0]
        header = json.loads(f.read(hlen))
        blob = f.read()

    sd = {}
    for t in header["tensors"]:
        raw = blob[t["offset"]: t["offset"] + t["nbytes"]]
        if t["kind"] == "ternary":
            states = unpack_base3(np.frombuffer(raw, dtype=np.uint8), t["n"])
            w = torch.from_numpy(states.astype(np.float32)).reshape(*t["shape"])
            sd[t["name"]] = w * t["scale"]
        else:
            dt = {"fp16": np.float16, "fp32": np.float32}[t["kind"]]
            a = np.frombuffer(raw, dtype=dt).reshape(*t["shape"])
            sd[t["name"]] = torch.from_numpy(a.astype(np.float32))
    if header.get("tied_head"):
        sd["head.weight"] = sd["embed.weight"]
    return sd, header


# --------------------------------------------------------------------------
# the integer inference path
# --------------------------------------------------------------------------

def ternary_matmul_masked(x, states, scale):
    """Practical form: y = scale * (sum of x over +1 columns - sum over -1).

    Multiplication by the weights is gone -- the only operands are 0/1 masks, so
    every product is x*1 or x*0. On a GPU this still dispatches to a float
    matmul because that is the hardware available; the point is that the
    *arithmetic* requires no multiplier.
    """
    pos = (states == 1).to(x.dtype)
    neg = (states == -1).to(x.dtype)
    return scale * (x @ pos.T - x @ neg.T)


def ternary_matmul_explicit(x, states, scale):
    """Provably add/subtract/skip, with no matmul anywhere. O(out*in) in Python,
    so this is a correctness reference for small tensors only -- it exists to
    demonstrate that the arithmetic really is additive."""
    out_f, in_f = states.shape
    y = torch.zeros(*x.shape[:-1], out_f, dtype=x.dtype, device=x.device)
    for o in range(out_f):
        row = states[o]
        acc = torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)
        for i in torch.nonzero(row == 1, as_tuple=False).flatten().tolist():
            acc = acc + x[..., i]          # ADD
        for i in torch.nonzero(row == -1, as_tuple=False).flatten().tolist():
            acc = acc - x[..., i]          # SUBTRACT
        y[..., o] = acc                    # state 0 contributes nothing: SKIP
    return scale * y


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def selftest(verbose=True):
    """Round-trip and arithmetic checks. Cheap, no model needed."""
    torch.manual_seed(0)
    ok = True

    for n in (1, 4, 5, 6, 100, 4096, 9973):
        s = torch.randint(-1, 2, (n,), dtype=torch.int8)
        packed, nn, pad = pack_base3(s)
        back = unpack_base3(packed, nn)
        same = np.array_equal(back, s.numpy())
        ok &= same
        if verbose and not same:
            print(f"  FAIL round-trip at n={n}")
    if verbose:
        print(f"  round-trip over 7 sizes: {'PASS' if ok else 'FAIL'}")

    s = torch.randint(-1, 2, (10_000,), dtype=torch.int8)
    packed, _, _ = pack_base3(s)
    bits = packed.nbytes * 8 / s.numel()
    if verbose:
        print(f"  density: {bits:.4f} bits/weight  "
              f"(floor log2(3)={np.log2(3):.4f}, efficiency "
              f"{np.log2(3)/bits:.1%})")

    x = torch.randn(3, 16)
    w = torch.randn(8, 16)
    states, scale = ternary_states(w)
    ref = torch.nn.functional.linear(x, states.float() * scale)
    a = ternary_matmul_masked(x, states, scale)
    b = ternary_matmul_explicit(x, states, scale)
    da, db = (a - ref).abs().max().item(), (b - ref).abs().max().item()
    ok &= da < 1e-4 and db < 1e-4
    if verbose:
        print(f"  masked   vs F.linear: max delta {da:.2e}")
        print(f"  explicit vs F.linear: max delta {db:.2e}  (add/sub/skip only)")
    return ok


def selftest_file(tmp="/tmp/_bitllm_selftest.bin", verbose=True):
    """End-to-end: build a tiny model, pack it, reload it, compare logits.

    Runs on CPU in about a second and catches both bugs that got through the
    unit-level selftest: isinstance failing across module reloads, and double
    quantization on load.
    """
    from .model import LM

    torch.manual_seed(0)
    cfg = dict(vocab=64, d=32, n_layer=2, n_head=4, mult=4,
               weight_mode="ternary", act_bits=8, embed_mode="ternary")
    m = LM(**cfg).eval()
    info = save_packed(m, tmp, config=cfg)
    m2, header = load_packed_model(tmp)

    x = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        d = (m(x)[0] - m2(x)[0]).abs().max().item()
    packed_all = info["packed_tensors"] == 13     # 6 per block x 2 + embedding
    if verbose:
        print(f"  packed tensors: {info['packed_tensors']}  expect 13  "
              f"{'PASS' if packed_all else 'FAIL'}")
        print(f"  full_dtype: {info['full_dtype']}")
        print(f"  max logit delta after file round-trip: {d:.3e}")
    return packed_all and d < 1e-2


if __name__ == "__main__":
    print("bitllm.pack selftest:")
    a = selftest()
    print("bitllm.pack selftest_file:")
    b = selftest_file()
    print("PASS" if (a and b) else "FAIL")


def load_packed_model(path, device="cpu"):
    """Load a packed file into a model that does NOT re-quantize.

    Returns (model, header). The result computes bit-identically to the QAT
    model the file was written from, because the packed weights already ARE the
    quantized weights.
    """
    from .model import LM

    sd, header = load_packed(path)
    cfg = header.get("inference_config")
    if cfg is None:                      # files written before this existed
        cfg = inference_config(header.get("config", {}))
    cfg = {k: v for k, v in cfg.items() if k in LM.__init__.__code__.co_varnames}
    m = LM(**cfg).to(device)
    m.load_state_dict({k: v.to(device) for k, v in sd.items()})
    m.eval()
    return m, header
