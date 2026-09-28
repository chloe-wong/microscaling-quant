"""Arithmetic helpers that pair with float_em: what a PE does between quantizations.
Bit-identical to the gemmini golden (`fp8_matmul_model.py`) functions named in each docstring.

    exact_add(a, b, e, m)         a + b exactly, then one rounding to float(e, m)        golden fp_add_exact, bf16_accum_add
    truncate_significand(x, m)    keep m fraction bits of |x|'s significand, drop the rest golden mx_product_quantize_trunc, step 1
    flush_product(x, floor_exp)   zero a product below 2^floor_exp                        MxFPMul PROD_FLOOR
    saturate_product(x, e, m)     clamp a product to the PE's largest float(e, m) value   golden mx_product_saturate
"""
import torch

from . import rounding
from .element_quant import float_em

__all__ = ["exact_add", "truncate_significand", "flush_product", "saturate_product"]


def _fits_float32(e: int, m: int) -> bool:
    """Two float(e, m) grid values sum exactly in float32.

    A grid value is below 2^(emax + 1) and is a multiple of 2^(emin - m); with bias 2^(e-1) - 1 that is
    emax - emin = 2^e - 3, so the grid spans 2^e - 2 + m bits and the exact sum of two values needs one more
    for the carry: 2^e - 1 + m. float32 holds 24.

        e = 2: m <= 21;  e = 3: m <= 17;  e = 4: m <= 9;  e >= 5: never.
    """
    return 2 ** e - 1 + m <= 24


def _two_sum(a: torch.Tensor, b: torch.Tensor):
    """Knuth's TwoSum: s = fl(a + b) and the residual t with s + t = a + b exactly. Six adds and subtracts, no
    multiplies, exact under IEEE round-to-nearest whenever nothing overflows."""
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _round_pair(s: torch.Tensor, t: torch.Tensor, e: int, m: int) -> torch.Tensor:
    """s + t (exact, |t| <= ulp32(s) / 2) rounded to float(e, m), nearest-even.

    float_em._ieee's scaled-integer grid on s, with one case decided by t instead of by evenness: the lane
    grid's step at s is at least 2 ulp32(s) for m <= 22, so t is under a quarter step and can only matter
    when s lies exactly on a midpoint of the grid. There it breaks the tie -- up in magnitude when t has s's
    sign, down otherwise. k = |s| / step is exact with k < 2^(m + 1) < 2^23, so "exactly a midpoint" is
    frac(k) == 0.5, exact. No float32 subnormal is formed: an e <= 7 lane's finest step is 2^(emin - m) >= 2^-84.
    """
    bias = float_em.bias(e)
    emin, emax = 1 - bias, bias
    ax = s.abs()
    finite_nz = torch.isfinite(s) & (ax != 0)
    _, exp = torch.frexp(torch.where(finite_nz, ax, torch.ones_like(ax)))
    E = (exp - 1).to(torch.float32)
    step = torch.exp2(torch.clamp(E, min=float(emin)) - m)
    k = ax / step                                                 # exact: step is a power of two
    kf = torch.floor(k)
    tie = (k - kf) == 0.5
    up = (torch.sign(s) * t) > 0                                  # the residual pushes |s + t| past the midpoint
    k_out = torch.where(tie & (t != 0), torch.where(up, kf + 1, kf), torch.round(k))   # round: half-to-even
    val = k_out * step
    val = torch.where(val >= 2.0 ** (emax + 1), torch.full_like(val, float("inf")), val)
    out = torch.where(finite_nz, torch.copysign(val, s), s)      # NaN, Inf, +-0 pass through
    return torch.where(finite_nz & (val == 0), torch.zeros_like(out), out)   # underflow -> +0.0, as _ieee


def exact_add(a: torch.Tensor, b: torch.Tensor, e: int, m: int, rounding_mode: str = "rne", grid: str = "ieee") -> torch.Tensor:
    """a + b computed exactly, then quantized once to float(e, m). Golden `fp_add_exact` / `bf16_accum_add`.

    Precondition: a and b are already on a float grid with significands of at most 24 bits (any float_em output,
    any float32). A zero addend returns the OTHER operand unchanged (golden short-circuit), so 0 + -0 -> -0,
    not +0.

    Three paths, chosen on (e, m) alone so the choice is fixed when a window is compiled:

    float32, when `_fits_float32(e, m)`: the sum is exact in float32, and `_ieee` on float32 input is exact
    for e < 8 (its step 2^(emin - m) >= 2^-13 is a normal float32), so the value rounded is the same one
    float64 would form. This is every lane of schedule.HW_FINAL's family. Measured on an L40S, where
    float64 runs at 1/64 the rate of float32: 5.07x on HW_FINAL at every layer shape, 6.90x on an all-e4m4
    ladder, bit-identical on the hard-value corpus, the frozen loop, the hardware fixture and 40 random
    ladders straddling the bound.

    float32 pair, for the other lanes with e <= 7: the sum is held exactly as s = fl(a + b) plus its residual
    t (`_two_sum`), and rounded by `_round_pair`, where t decides only the exact-midpoint case. Measured:
    6.43x on an all-e5m6 ladder, 6.36x on all-e7m10, bit-identical -- the pair costs nothing measurable over
    the direct add. With this path every lane under bf16 accumulates without float64.

    float64, otherwise: exact whenever the exponents differ by <= 29 bits; when they differ by more, the
    smaller operand is far below half an ulp of the result and cannot change the rounding, so the outcome is
    still the exact-add result. Operands that are not on such a grid (e.g. float64 values with more than 24
    significant bits) are outside this guarantee. bf16 (e = 8) always takes this path: its subnormal step
    is 2^-133, itself a float32 subnormal, which fused kernels flush to zero.
    """
    if rounding_mode == "rne" and grid == "ieee" and _fits_float32(e, m):
        q = float_em.quantize(a + b, e, m, rounding_mode="rne", grid="ieee")
        return torch.where(a == 0, b.to(torch.float32), torch.where(b == 0, a.to(torch.float32), q))
    if rounding_mode == "rne" and grid == "ieee" and e <= 7:
        q = _round_pair(*_two_sum(a, b), e, m)
        return torch.where(a == 0, b.to(torch.float32), torch.where(b == 0, a.to(torch.float32), q))
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


def flush_product(x: torch.Tensor, floor_exp: int) -> torch.Tensor:
    """Zero a product below 2^floor_exp (MxFPMul PROD_FLOOR, 2^-16 for the (4, 3) product)."""
    x = x.to(torch.float32)
    return torch.where(x.abs() < 2.0 ** floor_exp, torch.zeros_like(x), x)


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
