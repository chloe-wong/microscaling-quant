"""mxq.block.ocp — OCP Microscaling v1.0 block quantization, returning codes and scales.

    P, X = quantize(V, "MXFP8_E4M3", axis=0)        V_hat = P * expand(X)
    V_hat = dequantize(P, X, axis=0)

= scale_factor.ocp (shared exponent = floor(log2 amax) - emax, E8M0) followed by
  element_quant.microsoft (Microsoft's _quantize_elemwise, saturate, subnormals kept).

P * expand(X) is bit-identical to Microsoft's `_quantize_mx` (mxq.microxcaling.block_quantize) for the
same rounding mode. Unlike the reference, codes and scales are returned separately so a systolic
simulation can multiply codes and apply X_A * X_B once per block.
`rounding_mode` uses microxcaling's names: "even" (RNE), "nearest" (ties away), "floor".

`placement` is where the scale puts the block max (PLACEMENTS):
    top         the spec: in the format's top binade [2^emax, 2^(emax+1)); a max whose mantissa rounds above
                max_norm's is clipped to max_norm (scale_factor.ocp)
    below_top   one binade down, [2^(emax-1), 2^emax): never clips, the top binade unused (scale_factor.ocp_below_top)
    no_clip     top, except the blocks that would clip go one binade down (scale_factor.ocp_no_clip)
Only "top" is the OCP spec; the other two keep its element grid and change step 1 alone.
"""
from typing import Tuple, Union

import torch

from .. import scale_factor
from . import _driver
from ..element_quant import microsoft
from ..element_quant.formats import Format, get

__all__ = ["quantize", "dequantize", "PLACEMENTS"]

PLACEMENTS = ("top", "below_top", "no_clip")


def quantize(V: torch.Tensor, fmt: Union[str, Format], axis: int = 0, block_size: int = _driver.BLOCK,
             rounding_mode: str = "even", scale_bits: int = 8, placement: str = "top") -> Tuple[torch.Tensor, torch.Tensor]:
    """OCP block quantize V along `axis`. Returns (P codes, X power-of-two scales), float32."""
    f = get(fmt)
    if placement not in PLACEMENTS:
        raise ValueError(f"placement must be one of {PLACEMENTS}, got {placement!r}")
    if placement == "top":
        scale = lambda amax: scale_factor.ocp(amax, f.emax, scale_bits)                           # noqa: E731
    elif placement == "below_top":
        scale = lambda amax: scale_factor.ocp_below_top(amax, f.emax, scale_bits)                 # noqa: E731
    else:
        scale = lambda amax: scale_factor.ocp_no_clip(amax, f, rounding_mode, scale_bits)         # noqa: E731
    return _driver.compose(V, axis=axis, block_size=block_size,
                           scale=None if f is None else scale,                                                  # step 1
                           elem=None if f is None else lambda z: microsoft.quantize(z, f, rounding_mode=rounding_mode))   # step 2


def dequantize(P: torch.Tensor, X: torch.Tensor, axis: int = 0, block_size: int = _driver.BLOCK) -> torch.Tensor:
    """V_hat = P * expand(X): each scale repeated block_size times along `axis`, cut to P's length."""
    return _driver.dequantize(P, X, axis, block_size)
