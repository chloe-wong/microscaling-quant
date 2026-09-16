"""Element quantization to an arbitrary float(e, m) format: `round` picks the rounding rule, `grid` picks the
set of representable values.

    grid="qtorch"   qtorch 0.2.0 float_quantize semantics (MXQuant as shipped). Biased exponent clipped to
                    [127-L, 127+L], L = 2^(e-1)-1: top exponent reserved (e4m3 max 240), one binade of "fake
                    subnormals" at 2^-L with full mantissa, below 2^(-L-1) flush to zero, overflow saturates.
                    With round="ties_away" this is bit-identical to qtorch.quant.float_quantize(..., "nearest").
    grid="ieee"     IEEE-like format with bias 2^(e-1)-1, normals in [emin, emax] = [1-bias, bias], TRUE
                    subnormals with step 2^(emin-m), overflow -> Inf, NaN/Inf propagate. This is hardfloat's
                    RoundAnyRawFNToRecFN, i.e. the mesh lane accumulators; with round="rne" it is bit-identical
                    to the gemmini golden `fp_quantize_rne`. (8, 7) is bf16.

`round` is one of mxq.rounding.MODES: "ties_away" | "rne" | "truncate".
Input is cast to float32 unless it is float64, in which case the ieee grid rounds from float64 directly
(needed by exact_add, whose exact sum may not fit float32). Output is float32.
"""
import torch

from .. import rounding

__all__ = ["quantize", "max_value", "min_normal", "GRIDS"]

GRIDS = ("qtorch", "ieee")
_U32 = 0xFFFFFFFF


def max_value(e: int, m: int) -> float:
    """Largest finite value under grid="qtorch" (top exponent reserved)."""
    L = (1 << (e - 1)) - 1
    return (2.0 - 2.0 ** -m) * 2.0 ** L


def min_normal(e: int) -> float:
    """Smallest value with full mantissa under grid="qtorch"."""
    return 2.0 ** -((1 << (e - 1)) - 1)


def quantize(z: torch.Tensor, e: int, m: int, round: str = "ties_away", grid: str = "qtorch") -> torch.Tensor:
    if round not in rounding.MODES:
        raise ValueError(f"round must be one of {rounding.MODES}, got {round!r}")
    if not (1 <= e <= 8 and 1 <= m <= 22):
        raise ValueError(f"unsupported widths e={e}, m={m}")
    if grid == "qtorch":
        return _qtorch(z, e, m, round)
    if grid == "ieee":
        return _ieee(z, e, m, round)
    raise ValueError(f"grid must be one of {GRIDS}, got {grid!r}")


def _qtorch(z, e, m, round):
    """qtorch's float_kernel: round the float32 bit pattern at m fraction bits, then clip_exponent."""
    z32 = z.detach().to(torch.float32).contiguous()
    bits = z32.view(torch.int32).to(torch.int64) & _U32
    q = rounding.round_bits(bits, m, round)

    L = (1 << (e - 1)) - 1
    min_store, max_store = 127 - L, 127 + L
    exp_store = (q >> 23) & 0xFF
    sign = bits & 0x80000000
    max_num = sign | (max_store << 23) | (0x007FFFFF & ~((1 << (23 - m)) - 1))
    min_num = sign | (min_store << 23)
    middle = (min_store - 1) << 23
    q_mag = q & 0x7FFFFFFF

    q = torch.where(exp_store > max_store, max_num, q)
    below = exp_store < min_store
    q = torch.where(below & (q_mag > middle), min_num, q)
    q = torch.where(below & ~(q_mag > middle), torch.zeros_like(q), q)
    q = torch.where(q >= 1 << 31, q - (1 << 32), q).to(torch.int32)
    return q.view(torch.float32)


def _ieee(z, e, m, round):
    """Scaled-integer rounding: step = 2^(max(E, emin) - m), k = |x|/step (exact), round k, multiply back.
    Subnormals fall out because step stops shrinking at emin; a carry past 2^(m+1) is handled by the
    overflow test on the value itself."""
    x = z.detach()
    x = x if x.dtype == torch.float64 else x.to(torch.float32)
    bias = (1 << (e - 1)) - 1
    emin, emax = 1 - bias, bias

    ax = x.abs()
    finite_nz = torch.isfinite(x) & (ax != 0)
    _, exp = torch.frexp(torch.where(finite_nz, ax, torch.ones_like(ax)))
    E = (exp - 1).to(x.dtype)                                     # unbiased exponent, exact
    step = torch.exp2(torch.clamp(E, min=float(emin)) - m)
    k = rounding.round_int(ax / step, round)                      # ax/step is exact (step is a power of two)
    val = k * step
    val = torch.where(val >= 2.0 ** (emax + 1), torch.full_like(val, float("inf")), val)

    out = torch.where(finite_nz, torch.copysign(val, x), x)      # NaN, Inf, +-0 pass through
    if e < 8:                                                     # golden scalar path: underflow -> +0.0;
        out = torch.where(finite_nz & (val == 0), torch.zeros_like(out), out)   # its e=8 bit path keeps the sign
    return out.to(torch.float32)
