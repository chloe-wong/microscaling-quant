"""mxq.scale_factor — step 1 of block quantization: the per-block scale factor X = f(amax).

Every rule takes the block's max |value| (any shape, computed by the caller) and returns a
power-of-two scale of the same shape and dtype.

    mxquant(amax, floor)         X = 2^floor(log2 max(amax, floor))     block max lands in [1, 2)
    ocp(amax, emax)              X = 2^(floor(log2 amax) - emax)         block max lands in the format's top binade
    ocp_below_top(amax, emax)    X = 2^(floor(log2 amax) - emax + 1)     block max lands one binade below the top:
                                                                         never clips, the top binade goes unused
    ocp_no_clip(amax, fmt)       ocp's X, doubled for the blocks whose max would clip (amax / X rounded to the
                                                                         format's mantissa exceeds max_norm)

mxquant is the rule used by MXQuant's linear-layer simulation and by the MX-Gemmini RTL and
spike (gemmini 0b2cc2c and later: log2_pmax = 0). `floor` is the smallest block max a scale is
computed from; it matters only for all-zero or tiny blocks, and MXQuant's two quantizers disagree:

    MXQUANT_FLOOR   1e-38   linear_e2e_wrap/eval_mx_linear_e2e.py::mx_block32_quantize (the simulation; default)
    HARDWARE_FLOOR  2^-23   end_to_end_linear/mx_block_quant.py::_po2 (torch.finfo(float32).eps) and the
                            MX-Gemmini requantizer (mx_fp_math.h: max(amax, FLT_EPSILON)) -- the hardware

ocp is the OCP Microscaling v1.0 rule as implemented in Microsoft's microxcaling `_quantize_mx`
(see mxq/microxcaling/blockwise.py), including its E8M0 range handling. Under ocp a block max in the top
binade is clipped to max_norm whenever its mantissa rounds above max_norm's (448 = 1.75 x 2^8 for E4M3: every
block max with mantissa above 1.8125 is clipped). ocp_below_top and ocp_no_clip are the two ways around that
clip: give up the top binade for every block, or only for the blocks that would clip (a comparator and an
exponent increment in a requantizer; the host encoder must apply the same test).
"""
import torch

from .microxcaling.formats import FP32_MIN_NORMAL

__all__ = ["mxquant", "ocp", "ocp_below_top", "ocp_no_clip", "MXQUANT_FLOOR", "HARDWARE_FLOOR"]

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


def ocp_below_top(amax: torch.Tensor, emax: int, scale_bits: int = 8) -> torch.Tensor:
    """OCP's scale times two: the block max lands in [2^(emax-1), 2^emax), the binade below the format's top,
    so no element is ever clipped. The format's top binade is never used. Same E8M0 range handling as `ocp`."""
    return ocp(amax, emax - 1, scale_bits)


def ocp_no_clip(amax: torch.Tensor, fmt, rounding_mode: str = "even", scale_bits: int = 8) -> torch.Tensor:
    """OCP's scale, doubled for exactly the blocks whose max would clip: those where amax / X, rounded to the
    format's mantissa the way the elements are (`element_quant.microsoft`, `rounding_mode`), exceeds max_norm.
    Every other block gets `ocp`'s scale and `ocp`'s codes. `fmt` is a format name or Format."""
    from .element_quant.formats import get
    from .microxcaling.elemwise import _quantize_elemwise
    from .microxcaling.formats import ElemFormat
    f = get(fmt)
    if f is None:
        raise ValueError(f"{fmt!r} is pass-through (no element format); it has no scale")
    X = ocp(amax, f.emax, scale_bits)
    rounded = _quantize_elemwise(amax / X, ElemFormat.from_str(f.ocp), round=rounding_mode,
                                 saturate_normals=False, allow_denorm=True)        # an overflow is +-Inf here
    return torch.where(torch.isinf(rounded), X * 2, X)
