"""mxq.scale_factor — step 1 of block quantization: the per-block scale factor X = f(amax).

Both rules take the block's max |value| (any shape, computed by the caller) and return a
power-of-two scale of the same shape and dtype.

    mxquant(amax)      X = 2^floor(log2 amax)             block max lands in [1, 2)
    ocp(amax, emax)    X = 2^(floor(log2 amax) - emax)    block max lands in the format's top binade

mxquant is the rule used by MXQuant's linear-layer simulation and by the MX-Gemmini RTL and
spike (gemmini 0b2cc2c and later: log2_pmax = 0). ocp is the OCP Microscaling v1.0 rule as implemented in Microsoft's
microxcaling `_quantize_mx` (see mxq/ocp/blockwise.py), including its E8M0 range handling.
"""
import torch

from .ocp.formats import FP32_MIN_NORMAL

__all__ = ["mxquant", "ocp"]

#: floor used by MXQuant's `mx_block32_quantize` so a zero block gets a finite scale.
_MXQUANT_FLOOR = 1e-38


def mxquant(amax: torch.Tensor) -> torch.Tensor:
    """MXQuant scale: 2^floor(log2 amax), floored at 1e-38 (verbatim numerics of
    linear_e2e_wrap/eval_mx_linear_e2e.py::mx_block32_quantize)."""
    sc = torch.pow(2.0, torch.floor(torch.log2(amax.clamp(min=_MXQUANT_FLOOR))))
    return sc.clamp(min=_MXQUANT_FLOOR)


def ocp(amax: torch.Tensor, emax: int, scale_bits: int = 8) -> torch.Tensor:
    """OCP scale: 2^(floor(log2 amax) - emax), with microxcaling's E8M0 range handling.

    Zero blocks use FP32_MIN_NORMAL for the log. Shared exponents above the E8M0 range
    become NaN (overflow); below it they clamp to -(2^(scale_bits-1) - 1).
    Numerics match mxq/ocp/blockwise.py::_quantize_mx exactly.
    """
    shared_exp = torch.floor(torch.log2(amax + FP32_MIN_NORMAL * (amax == 0).type(amax.dtype)))
    shared_exp = shared_exp - emax
    scale_emax = 2 ** (scale_bits - 1) - 1
    shared_exp = torch.where(shared_exp > scale_emax, torch.full_like(shared_exp, float("nan")), shared_exp)
    shared_exp = torch.where(shared_exp < -scale_emax, torch.full_like(shared_exp, -scale_emax), shared_exp)
    return 2 ** shared_exp
