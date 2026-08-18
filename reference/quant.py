"""
Quantization primitives for BitNet b1.58.

Everything here is *fake quant*: the forward pass computes a quantized value,
but gradients flow back to the underlying full-precision tensor through a
straight-through estimator. No weights are packed into real bits anywhere in
this file -- bit-packing belongs to the inference path, which is a separate
piece of work.

The scheme is W1.58A8, from the BitNet b1.58 2B4T technical report
(arXiv:2504.12285):

    weights      ternary {-1, 0, +1}, per-TENSOR absmean scaling
    activations  8-bit integers,      per-TOKEN absmax scaling

Why those two differ matters. A single scale for the whole weight tensor is
fine because weights are static and roughly well-behaved. Activations are
not: LLMs produce enormous outliers at a handful of positions, so each token
gets its own scale and a loud neighbour cannot crush it.
"""

import torch

EPS = 1e-5


# --------------------------------------------------------------------------
# the straight-through estimator
# --------------------------------------------------------------------------

def ste(x: torch.Tensor, x_q: torch.Tensor) -> torch.Tensor:
    """Forward returns ``x_q``; backward behaves as if nothing happened.

    ``(x_q - x)`` is detached, so it contributes a constant to the forward
    value and exactly zero to the gradient. The derivative of the whole
    expression with respect to ``x`` is therefore 1.

    This is the single trick that makes quantization-aware training possible.
    ``round()`` has zero gradient almost everywhere, so without this the
    network below a quantizer would receive no learning signal at all.

    The cost: gradients are now *approximate*. A latent weight can drift a
    long way without ever crossing a rounding boundary, so updates arrive in
    lumps and the loss curve is visibly noisier than fp32. That is expected,
    not a bug.
    """
    return x + (x_q - x).detach()


# --------------------------------------------------------------------------
# weight quantizers
# --------------------------------------------------------------------------

def weight_ternary(w: torch.Tensor) -> torch.Tensor:
    """BitNet b1.58: ternary weights via per-tensor absmean scaling.

        gamma = mean(|W|)                      -- one scalar for the tensor
        W_q   = round(clamp(W / gamma, -1, 1)) -- lands in {-1, 0, +1}
        return W_q * gamma                     -- back to the original range

    Re-applying ``gamma`` is what makes this a drop-in replacement for a
    normal weight: magnitudes stay comparable, so the rest of the network
    does not need retuning. The information content is still ~1.58 bits
    (log2 of three states) because only three distinct values exist.
    """
    gamma = w.abs().mean().clamp(min=EPS)
    return (w / gamma).round().clamp(-1, 1) * gamma


def weight_binary(w: torch.Tensor) -> torch.Tensor:
    """Original BitNet (2023): sign only, {-1, +1}. Exactly 1 bit.

    Note this is *not* ``torch.sign``, which maps 0 to 0 and would sneak a
    third state back in. Genuine binary has no zero, which is precisely why
    it trains worse: the network loses the ability to say "no connection".
    """
    gamma = w.abs().mean().clamp(min=EPS)
    return torch.where(w >= 0, 1.0, -1.0).to(w.dtype) * gamma


def weight_intk(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-tensor integer quantization to ``bits`` bits.

    Used for the bit-width sweep (int8, int4, ...) so the same model can be
    trained at several precisions and plotted as quality vs bits-per-weight.
    """
    qmax = 2 ** (bits - 1) - 1
    scale = (w.abs().amax() / qmax).clamp(min=EPS)
    return (w / scale).round().clamp(-qmax, qmax) * scale


# --------------------------------------------------------------------------
# activation quantizer
# --------------------------------------------------------------------------

def act_quant(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Per-token absmax quantization of activations.

    ``amax(dim=-1)`` takes the max over the feature dimension, giving one
    scale per token per batch element. Note that "1-bit LLM" never means
    1-bit activations -- BitNet keeps these at int8 for exactly the outlier
    reason described in the module docstring.
    """
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=EPS)
    return (x * qmax / scale).round().clamp(-qmax, qmax) * scale / qmax


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------

WEIGHT_MODES = ("none", "ternary", "binary", "int8", "int4", "int2")


def quantize_weight(w: torch.Tensor, mode: str) -> torch.Tensor:
    """Apply the weight quantizer named by ``mode``. ``"none"`` is a no-op."""
    if mode == "none":
        return w
    if mode == "ternary":
        return weight_ternary(w)
    if mode == "binary":
        return weight_binary(w)
    if mode.startswith("int"):
        return weight_intk(w, int(mode[3:]))
    raise ValueError(f"unknown weight mode {mode!r}; expected one of {WEIGHT_MODES}")


def bits_per_weight(mode: str) -> float:
    """Information content per weight, for labelling the sweep's x-axis."""
    import math
    return {
        "none": 32.0,
        "ternary": math.log2(3),   # 1.585
        "binary": 1.0,
        "int8": 8.0,
        "int4": 4.0,
        "int2": 2.0,
    }[mode]
