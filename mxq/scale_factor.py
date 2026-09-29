"""mxq.scale_factor — step 1 of block quantization: the per-block scale factor X = f(amax).

Both rules take the block's max |value| (any shape, computed by the caller) and return a
power-of-two scale of the same shape and dtype.

    mxquant(amax, floor)   X = 2^floor(log2 max(amax, floor))     block max lands in [1, 2)
    ocp(amax, emax)        X = 2^(floor(log2 amax) - emax)         block max lands in the format's top binade

mxquant is the rule used by MXQuant's linear-layer simulation and by the MX-Gemmini RTL and
spike (gemmini 0b2cc2c and later: log2_pmax = 0). `floor` is the smallest block max a scale is
computed from; it matters only for all-zero or tiny blocks, and MXQuant's two quantizers disagree:

    MXQUANT_FLOOR   1e-38   linear_e2e_wrap/eval_mx_linear_e2e.py::mx_block32_quantize (the simulation; default)
    HARDWARE_FLOOR  2^-23   end_to_end_linear/mx_block_quant.py::_po2 (torch.finfo(float32).eps) and the
                            MX-Gemmini requantizer (mx_fp_math.h: max(amax, FLT_EPSILON)) -- the hardware

ocp is the OCP Microscaling v1.0 rule as implemented in Microsoft's microxcaling `_quantize_mx`
(see mxq/microxcaling/blockwise.py), including its E8M0 range handling.
"""
import torch

from .microxcaling.formats import FP32_MIN_NORMAL

__all__ = ["mxquant", "ocp", "MXQUANT_FLOOR", "HARDWARE_FLOOR"]

#: floor used by MXQuant's `mx_block32_quantize` so a zero block gets a finite scale.
MXQUANT_FLOOR = 1e-38
#: floor used by MXQuant's end_to_end_linear quantizer and by the MX-Gemmini requantizer (FLT_EPSILON).
HARDWARE_FLOOR = 2.0 ** -23
_MXQUANT_FLOOR = MXQUANT_FLOOR


def mxquant(amax: torch.Tensor, floor: float = MXQUANT_FLOOR) -> torch.Tensor:
    """MXQuant scale: 2^floor(log2 max(amax, floor)), floored at `floor` (with the default, the verbatim
    numerics of linear_e2e_wrap/eval_mx_linear_e2e.py::mx_block32_quantize)."""
    sc = torch.pow(2.0, torch.floor(torch.log2(amax.clamp(min=floor))))
    return sc.clamp(min=floor)


def ocp(amax: torch.Tensor, emax: int, scale_bits: int = 8) -> torch.Tensor:
    """OCP scale: 2^(floor(log2 amax) - emax), with microxcaling's E8M0 range handling.

    Zero blocks use FP32_MIN_NORMAL for the log. Shared exponents above the E8M0 range
    become NaN (overflow); below it they clamp to -(2^(scale_bits-1) - 1).
    Numerics match mxq/microxcaling/blockwise.py::_quantize_mx exactly.
    """
    shared_exp = torch.floor(torch.log2(amax + FP32_MIN_NORMAL * (amax == 0).type(amax.dtype)))
    shared_exp = shared_exp - emax
    scale_emax = 2 ** (scale_bits - 1) - 1
    shared_exp = torch.where(shared_exp > scale_emax, torch.full_like(shared_exp, float("nan")), shared_exp)
    shared_exp = torch.where(shared_exp < -scale_emax, torch.full_like(shared_exp, -scale_emax), shared_exp)
    return 2 ** shared_exp
