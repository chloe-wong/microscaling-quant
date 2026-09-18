"""mxq.microxcaling — OCP Microscaling (MX) v1.0 element formats.

Implementation is Microsoft's reference (microxcaling), copied verbatim — see UPSTREAM.md.
This file is the only one here we wrote: thin wrappers with MX-conversion flags baked in.

    fp8_e4m3(x)   fp8_e5m2(x)   fp6_e3m2(x)   fp6_e2m3(x)   fp4_e2m1(x)   bf16(x)

All take a float tensor and return the dequantized float tensor on that format's grid
(fake quantization). `round` is "even" (RNE) by default; "nearest" is ties-away-from-zero.

Lower-level access: `quantize(x, fmt, ...)`, `params(fmt)`, `ElemFormat`, `RoundingMode`,
and the verbatim upstream modules `formats`, `elemwise`, `blockwise`.
"""
from .formats  import ElemFormat, RoundingMode, _get_format_params as params
from .elemwise import _quantize_elemwise, _quantize_bfloat
from .blockwise import _quantize_mx

__all__ = ["fp8_e4m3", "fp8_e5m2", "fp6_e3m2", "fp6_e2m3", "fp4_e2m1", "bf16",
           "quantize", "block_quantize", "params", "ElemFormat", "RoundingMode"]

#: formats with no Inf encoding: out-of-range values saturate to max_norm (OCP).
_SATURATING = {ElemFormat.fp8_e4m3, ElemFormat.fp6_e3m2, ElemFormat.fp6_e2m3, ElemFormat.fp4_e2m1}


def quantize(x, fmt, round="even"):
    """Quantize to any upstream ElemFormat with OCP MX conversion semantics."""
    fmt = ElemFormat.from_str(fmt) if isinstance(fmt, str) else fmt
    return _quantize_elemwise(x, fmt, round=round,
                              saturate_normals=(fmt in _SATURATING),
                              allow_denorm=True)


def fp8_e4m3(x, round="even"): return quantize(x, ElemFormat.fp8_e4m3, round)
def fp8_e5m2(x, round="even"): return quantize(x, ElemFormat.fp8_e5m2, round)
def fp6_e3m2(x, round="even"): return quantize(x, ElemFormat.fp6_e3m2, round)
def fp6_e2m3(x, round="even"): return quantize(x, ElemFormat.fp6_e2m3, round)
def fp4_e2m1(x, round="even"): return quantize(x, ElemFormat.fp4_e2m1, round)
def bf16(x, round="even"):     return _quantize_bfloat(x, 16, round=round)


def block_quantize(x, fmt, axis=-1, block_size=32, scale_bits=8, round="even"):
    """OCP MX block quantization: shared E8M0 exponent per block, then element quantize.

    Upstream `_quantize_mx`, dequantized output only: the oracle mxq.block.ocp is validated against.
    """
    fmt = ElemFormat.from_str(fmt) if isinstance(fmt, str) else fmt
    return _quantize_mx(x, scale_bits, fmt, axes=[axis], block_size=block_size, round=round)
