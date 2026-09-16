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
                                   Exact for |k| < 2^23. Used for scaled-integer (subnormal-capable) grids.
"""
from typing import Union

import torch

from . import ties_away, rne, truncate

MODES = ("ties_away", "rne", "truncate")
_MOD = {"ties_away": ties_away, "rne": rne, "truncate": truncate}
_U32 = 0xFFFFFFFF

__all__ = ["MODES", "round_bits", "round_int", "ties_away", "rne", "truncate"]


def _drop(bits: torch.Tensor, keep: Union[int, torch.Tensor]):
    """(mask of dropped bits, half ulp) for `keep` fraction bits, both broadcast to `bits`."""
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
