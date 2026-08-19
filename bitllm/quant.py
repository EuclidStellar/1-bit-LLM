"""
Quantization primitives for BitNet b1.58 (arXiv:2504.12285).

W1.58A8: ternary weights via per-tensor absmean, int8 activations via
per-token absmax. Everything here is *fake quant* -- the forward pass computes
a quantized value while gradients flow to the underlying fp32 tensor through a
straight-through estimator. No bit-packing; that belongs to the inference path.
"""

import torch

EPS = 1e-5


def ste(x, x_q):
    """Forward returns x_q. Backward behaves as if quantization never happened.

    (x_q - x) is detached, so it contributes to the forward value and exactly
    zero to the gradient. d(out)/dx is therefore 1.

    MEASURED: gradient mean with .detach() = 1.0, without it = 7.12e-06.
    That residual 7e-06 is NOT the rounding -- round() has exactly zero
    gradient. It leaks through the differentiable absmean *scale*, which
    appears twice in weight_ternary. Verify by detaching only the scale: then
    the gradient is exactly 0.
    """
    return x + (x_q - x).detach()


def weight_ternary(w):
    """Per-TENSOR absmean scaling to {-g, 0, +g}.

        g   = mean(|W|)
        W_q = round(clamp(W/g, -1, 1)) * g

    Note this is scale-equivariant: weight_ternary(c*w) == c*weight_ternary(w),
    because g scales with w. The ternary STATES are scale-invariant; only the
    magnitude carries through. That is why the 1/sqrt(2L) residual init scaling
    still works identically for the ternary and fp32 arms.

    For Gaussian weights the zero fraction is analytically 2*Phi(0.5*sqrt(2/pi))-1
    = 0.3101, and the scale is sigma*sqrt(2/pi) = 0.7979*sigma. MEASURED at init:
    0.3105 and 0.7957. It drifts once the weight distribution stops being
    Gaussian -- tracking that drift is how you watch the quantizer learn.
    """
    g = w.abs().mean().clamp(min=EPS)
    return (w / g).round().clamp(-1, 1) * g


def weight_binary(w):
    """Original BitNet (2023): sign only, {-g, +g}. Exactly 1 bit.

    Deliberately not torch.sign, which maps 0 -> 0 and would smuggle a third
    state back in. True binary has no zero, which is why it trains worse: the
    model loses the ability to express "no connection".
    """
    g = w.abs().mean().clamp(min=EPS)
    return torch.where(w >= 0, 1.0, -1.0).to(w.dtype) * g


def weight_intk(w, bits):
    """Symmetric per-tensor integer quantization, for the bit-width sweep."""
    qmax = 2 ** (bits - 1) - 1
    s = (w.abs().amax() / qmax).clamp(min=EPS)
    return (w / s).round().clamp(-qmax, qmax) * s


def act_quant(x, bits=8):
    """Per-TOKEN absmax scaling. One scale per position, so a single loud token
    cannot crush the scale for its neighbours. "1-bit LLM" never means 1-bit
    activations -- BitNet keeps these at int8 for exactly this reason."""
    qmax = 2 ** (bits - 1) - 1
    s = x.abs().amax(dim=-1, keepdim=True).clamp(min=EPS)
    return (x * qmax / s).round().clamp(-qmax, qmax) * s / qmax


WEIGHT_MODES = ("none", "ternary", "binary", "int8", "int4", "int2")


def quantize_weight(w, mode):
    if mode == "none":    return w
    if mode == "ternary": return weight_ternary(w)
    if mode == "binary":  return weight_binary(w)
    if mode.startswith("int"): return weight_intk(w, int(mode[3:]))
    raise ValueError(f"unknown weight mode {mode!r}; expected {WEIGHT_MODES}")


def bits_per_weight(mode):
    import math
    return {"none": 32.0, "ternary": math.log2(3), "binary": 1.0,
            "int8": 8.0, "int4": 4.0, "int2": 2.0}[mode]
