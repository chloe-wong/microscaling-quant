"""mxq.rounding — the three rounding rules used across the MX datapath, each in two forms.

    mode          rule on a tie                    who uses it
    ties_away     toward larger magnitude          qtorch float_quantize (MXQuant as shipped)
    rne           toward even                      hardfloat round_near_even: lane accumulators, bf16, compiler codes
    truncate      (no tie: drop the bits)          PE product stage (MxFPMul)

Two entry points, same modes:

    round_bits(bits, keep, mode)   float32 bit pattern (int64 holding the uint32 pattern) rounded so that only
                                   `keep` fraction bits remain. `keep` may be a per-element tensor. A mantissa
                                   carry propagates into the exponent, as in hardware. Cannot round below the
                                   implicit leading 1, so it does not express subnormal grids.
    round_int(k, mode)             float tensor holding an integer-valued quantity, rounded to an integer.
                                   Exact for |k| < 2^23 in float32 (2^53 in float64). Used for scaled-integer
                                   (subnormal-capable) grids.
    bf16(x)                        float32 rounded to the bf16 grid, nearest-even, on the bit pattern (safe inside
                                   torch.compile, unlike a cast pair): matmul.MXGEMMINI's lane and tile rounding and
                                   every step of mxq.nn's bf16 vector ops.
"""
from typing import Union

import torch

from . import ties_away, rne, truncate

MODES = ("ties_away", "rne", "truncate")
_MOD = {"ties_away": ties_away, "rne": rne, "truncate": truncate}
_U32 = 0xFFFFFFFF   # keeps a float32 bit pattern inside 32 bits after int64 arithmetic

__all__ = ["MODES", "round_bits", "round_int", "bf16", "ties_away", "rne", "truncate"]


def _drop(bits: torch.Tensor, keep: Union[int, torch.Tensor]):
    """(number of dropped bits, mask of dropped bits, half ulp) for `keep` fraction bits, broadcast to `bits`."""
    if isinstance(keep, int):                                   # the common case: no tensor, no device sync
        if not 0 <= keep <= 22:
            raise ValueError("keep must be in [0, 22] (23 = identity is not a rounding)")
        drop = 23 - keep
        return drop, (1 << drop) - 1, 1 << (drop - 1)
    keep = torch.as_tensor(keep, dtype=torch.int64, device=bits.device)
    if bool((keep < 0).any()) or bool((keep > 22).any()):
        raise ValueError("keep must be in [0, 22] (23 = identity is not a rounding)")
    drop = 23 - keep
    one = torch.ones((), dtype=torch.int64, device=bits.device)
    mask = torch.bitwise_left_shift(one, drop) - 1
    half = torch.bitwise_left_shift(one, drop - 1)
    return drop, mask, half


def round_bits(bits: torch.Tensor, keep: Union[int, torch.Tensor], mode: str) -> torch.Tensor:
    if mode not in _MOD:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    drop, mask, half = _drop(bits, keep)
    return _MOD[mode].bits(bits, drop, mask, half) & _U32


def round_int(k: torch.Tensor, mode: str) -> torch.Tensor:
    if mode not in _MOD:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return _MOD[mode].integer(k)


def bf16(x: torch.Tensor) -> torch.Tensor:
    """Round to the bf16 grid, nearest-even, on the float32 bit pattern: (bits + 0x7FFF + lsb) & ~0xFFFF.
    Returns float32.

    matmul.MXGEMMINI's bf16 lane and tile rounding, and every step of mxq.nn's bf16 vector ops. It is the
    one lane rounding not done by float_em's scaled-integer grid. That grid runs in float64 for
    e = 8 because bf16's subnormal step, 2^-133, is itself a float32 subnormal that fused kernels flush; the
    bit form needs no float arithmetic at all, so it is exact on subnormals (bf16 shares float32's emin, so
    the uniform bit-add IS the grid), rounds the largest finite values into 0x7F800000 = Inf as IEEE does,
    keeps the sign of +-0, and passes NaN through unchanged. It is the same primitive the qtorch grid uses.

    NOT `x.to(bfloat16).to(float32)`: exact in eager, but torch.compile folds that cast pair away inside a
    fused kernel and the value comes back unrounded (2499980 of 3500168 corpus values). On schedule.HW_FINAL
    nothing that reaches a bf16 rounding is ever off the bf16 grid, so every gate passed with the cast; a
    ladder whose reduction ends in float32 showed it. `torch._inductor.config.emulate_precision_casts` does not
    prevent the fold. The gate's tier 1 now runs every stage compiled as well as eager for this reason.
    """
    x32 = x.to(torch.float32).contiguous()
    bits = x32.view(torch.int32).to(torch.int64) & _U32
    q = round_bits(bits, 7, "rne")
    q = torch.where(q >= 1 << 31, q - (1 << 32), q).to(torch.int32).view(torch.float32)
    return torch.where(torch.isnan(x32), x32, q)
