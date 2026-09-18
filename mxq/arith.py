"""Arithmetic helpers that pair with float_em: what a PE does between quantizations.
Bit-identical to the gemmini golden (`fp8_matmul_model.py`) functions named in each docstring.

    exact_add(a, b, e, m)         a + b exactly, then one rounding to float(e, m)        golden fp_add_exact, bf16_accum_add
    truncate_significand(x, m)    keep m fraction bits of |x|'s significand, drop the rest golden mx_product_quantize_trunc, step 1
    saturate_product(x, e, m)     clamp a product to the PE's largest float(e, m) value   golden mx_product_saturate
"""
import torch

from . import rounding
from .element_quant import float_em

__all__ = ["exact_add", "truncate_significand", "saturate_product"]


def exact_add(a: torch.Tensor, b: torch.Tensor, e: int, m: int, rounding_mode: str = "rne", grid: str = "ieee") -> torch.Tensor:
    """a + b computed exactly, then quantized once to float(e, m). Golden `fp_add_exact` / `bf16_accum_add`.

    Precondition: a and b are already on a float grid with significands of at most 24 bits (any float_em output,
    any float32). The sum is formed in float64: exact whenever the exponents differ by <= 29 bits; when they
    differ by more, the smaller operand is far below half an ulp of the result and cannot change the rounding,
    so the outcome is still the exact-add result. Operands that are not on such a grid (e.g. float64 values with
    more than 24 significant bits) are outside this guarantee.
    A zero addend returns the OTHER operand unchanged (golden short-circuit), so 0 + -0 -> -0, not +0.
    """
    a64, b64 = a.detach().to(torch.float64), b.detach().to(torch.float64)
    q = float_em.quantize(a64 + b64, e, m, rounding_mode=rounding_mode, grid=grid)
    return torch.where(a64 == 0, b.to(torch.float32), torch.where(b64 == 0, a.to(torch.float32), q))


def truncate_significand(x: torch.Tensor, m: int) -> torch.Tensor:
    """|x| = s * 2^E with s in [1, 2): keep m fraction bits of s, drop the rest (toward zero). No exponent clamp:
    a float32 subnormal is normalised first, so it is truncated at its own leading 1 and not flushed.
    Zeros, Inf and NaN pass through. Golden `mx_product_quantize_trunc` before its saturation step (MxFPMul).
    """
    x = x.to(torch.float32)
    ax = x.abs()
    nz = torch.isfinite(x) & (ax != 0)
    mant, ex = torch.frexp(torch.where(nz, ax, torch.ones_like(ax)))     # ax = mant * 2^ex, mant in [0.5, 1)
    k = rounding.round_int(mant * (1 << (m + 1)), "truncate")            # significand with m fraction bits, exact
    t = torch.ldexp(k / (1 << m), ex - 1)                                # back to the value
    return torch.where(nz, torch.copysign(t, x), x)


def saturate_product(x: torch.Tensor, e: int, m: int) -> torch.Tensor:
    """MxPEOutToRaw saturation after the product truncation. Golden `mx_product_saturate`.

    hardfloat treats biased exponent 2^e - 1 as special, so the PE saturates to mantissa 2^m - 2 at unbiased
    exponent bias + 1. For e4m3 that is 448, the format max. Values below max_normal pass through unchanged.
    Only (4, 3) is validated against hardware; other (e, m) follow the golden's general formula unchecked.
    Edge cases, as in the golden: +-Inf saturates (Inf > max_normal); NaN stays NaN; -0 becomes +0 (sign(-0) = 0).
    """
    x = x.to(torch.float32)
    bias = float_em.bias(e)
    is_mx_fp8 = (e == 4 and m == 3)
    emax = bias + 1 if is_mx_fp8 else bias
    scale = float(2 ** m)
    max_mant = (2 ** m - 2) if is_mx_fp8 else (2 ** m - 1)
    max_normal = (2.0 ** emax) * (1.0 + max_mant / scale)
    sat_val = (1.0 + (2 ** m - 2) / scale) * (2.0 ** (bias + 1))
    ax = x.abs()
    return torch.sign(x) * torch.where(ax > max_normal, torch.full_like(ax, sat_val), ax)
