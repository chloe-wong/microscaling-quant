"""mxq.block_mxgemmini — the block quantization of MX-Gemmini's operands.

    P, X = quantize(V, "MXFP8_E4M3", axis=0)        V_hat = P * expand(X)
    V_hat = dequantize(P, X, axis=0)

= scale_factor.mxquant (X = 2^floor(log2 amax), block max in [1, 2), no emax offset) followed by
  element_quant.float_em(grid="ocp", rounding_mode="ties_away") (OCP element grid: true subnormals, saturate at max_norm).

Same scale rule as block_mxquant; only the element grid differs (qtorch 0.2.0's fake subnormals there).
Validated against npu-exploration/rtl_exact: P and X equal the operands its hardware output was computed from.
"""
from typing import Tuple, Union

import torch

from . import _blocks, scale_factor
from .element_quant import float_em
from .element_quant.formats import Format, get

__all__ = ["quantize", "dequantize"]


def quantize(V: torch.Tensor, fmt: Union[str, Format], axis: int = 0,
             block_size: int = _blocks.BLOCK) -> Tuple[torch.Tensor, torch.Tensor]:
    """MX-Gemmini block quantize V along `axis`. Returns (P codes, X power-of-two scales), float32."""
    f = get(fmt)
    return _blocks.quantize(V, axis, block_size, passthrough=f is None,
                            scale=scale_factor.mxquant,                                                      # step 1
                            elem=lambda z: float_em.quantize(z, f.e, f.m, rounding_mode="ties_away", grid="ocp"))  # step 2


def dequantize(P: torch.Tensor, X: torch.Tensor, axis: int = 0, block_size: int = _blocks.BLOCK) -> torch.Tensor:
    """V_hat = P * expand(X): each scale repeated block_size times along `axis`, cut to P's length."""
    return _blocks.dequantize(P, X, axis, block_size)
