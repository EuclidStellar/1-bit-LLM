"""
Look at the weights.

Ternary weights are one of the very few things in machine learning you can
read with your eyes. Three states print as three characters, so a weight
matrix becomes a picture you can watch evolve across training -- the
quantizer sorting weights into buckets, and the sparsity fraction settling.
"""

import torch

from .model import BitLM, BitLinear
from .quant import quantize_weight

GLYPH = {-1: "-", 0: "·", 1: "+"}


def ternary_states(module: BitLinear) -> torch.Tensor:
    """The {-1, 0, +1} state of every weight in a BitLinear."""
    with torch.no_grad():
        q = quantize_weight(module.weight, module.weight_mode)
        gamma = module.weight.abs().mean().clamp(min=1e-5)
        return (q / gamma).round().clamp(-1, 1).to(torch.int8)


def print_ternary(module: BitLinear, rows: int = 24, cols: int = 72,
                  title: str = ""):
    """Print the top-left corner of a weight matrix as a - / . / + grid."""
    s = ternary_states(module).cpu()
    r, c = min(rows, s.shape[0]), min(cols, s.shape[1])
    if title:
        print(f"\n{title}  [showing {r}x{c} of {s.shape[0]}x{s.shape[1]}]")
    for i in range(r):
        print("  " + "".join(GLYPH[int(v)] for v in s[i, :c]))


def sparsity_report(model: BitLM):
    """Per-layer breakdown of how the quantizer distributed the three states.

    The zero fraction is the interesting column. It is not fixed by the
    algorithm -- it emerges from how the weight distribution sits relative to
    the absmean threshold, and it drifts as training proceeds.
    """
    rows = []
    for name, m in model.named_modules():
        if not isinstance(m, BitLinear) or m.weight_mode == "none":
            continue
        s = ternary_states(m)
        n = s.numel()
        rows.append((
            name,
            n,
            (s == -1).sum().item() / n,
            (s == 0).sum().item() / n,
            (s == 1).sum().item() / n,
        ))

    if not rows:
        print("no quantized layers (weight_mode='none')")
        return

    print(f"\n{'layer':<34} {'weights':>10} {'-1':>7} {'0':>7} {'+1':>7}")
    print("-" * 70)
    for name, n, neg, zero, pos in rows:
        print(f"{name:<34} {n:>10,} {neg:>6.1%} {zero:>6.1%} {pos:>6.1%}")

    total = sum(r[1] for r in rows)
    wz = sum(r[1] * r[3] for r in rows) / total
    print("-" * 70)
    print(f"{'weighted mean zero fraction':<34} {total:>10,} {'':>7} {wz:>6.1%}")


def memory_report(model: BitLM):
    """What the model would actually weigh on disk once packed.

    Splits quantizable from full-precision, because the full-precision share
    is what decides whether the memory claim is real. BitNet's own headline
    number excludes the embedding table for exactly this reason.
    """
    r = model.param_report()
    packed = r["packed_bytes_quantizable"] + r["packed_bytes_fp16_rest"]
    fp16_all = r["total"] * 2

    print(f"\n{'':<28} {'params':>13} {'bytes':>12}")
    print("-" * 56)
    print(f"{'quantizable (BitLinear)':<28} {r['quantizable']:>13,} "
          f"{r['packed_bytes_quantizable'] / 1e6:>11.2f}M")
    print(f"{'full precision (fp16)':<28} {r['full_precision']:>13,} "
          f"{r['packed_bytes_fp16_rest'] / 1e6:>11.2f}M")
    print("-" * 56)
    print(f"{'packed total':<28} {r['total']:>13,} {packed / 1e6:>11.2f}M")
    print(f"{'same model, all fp16':<28} {r['total']:>13,} {fp16_all / 1e6:>11.2f}M")
    print(f"\ncompression vs fp16: {fp16_all / packed:.2f}x")
    fp_share = r["packed_bytes_fp16_rest"] / packed
    print(f"full-precision share of the packed file: {fp_share:.1%}")
    if fp_share > 0.5:
        print("  ^ the un-quantizable part dominates. Shrink the vocab or "
              "raise n_embd\n    if you want the memory story to be about "
              "the ternary weights.")
