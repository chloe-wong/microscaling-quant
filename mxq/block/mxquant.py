"""mxq.block.mxquant — the block quantization MXQuant's systolic-array simulation used for its linear-layer
inputs (linear_e2e_wrap / complete_integration_e2e `mx_block32_quantize`). Every reported MXQuant perplexity
used these codes.

    P, X = quantize(V, "MXFP8_E4M3", axis=0)        V_hat = P * expand(X)
    V_hat = dequantize(P, X, axis=0)

= scale_factor.mxquant (X = 2^floor(log2 amax), block max in [1, 2), no emax offset) followed by
  element_quant.float_em (float(e, m) codes, qtorch 0.2.0 grid, bit-identical to qtorch.float_quantize "nearest").

P and X are bit-identical to `mx_block32_quantize(V, fmt, axis="col"|"row")` with axis=0|1.
Differences from OCP are listed in Notes/FP_Notes.md.
"""
from typing import Tuple, Union

import torch

from .. import scale_factor
from . import _driver
from ..element_quant import float_em
from ..element_quant.formats import Format, get

__all__ = ["quantize", "dequantize"]


def quantize(V: torch.Tensor, fmt: Union[str, Format], axis: int = 0,
             block_size: int = _driver.BLOCK) -> Tuple[torch.Tensor, torch.Tensor]:
    """MXQuant-simulation block quantize V along `axis`. Returns (P codes, X power-of-two scales), float32."""
    f = get(fmt)
    return _driver.quantize(V, axis, block_size, passthrough=f is None,
                            scale=scale_factor.mxquant,                       # step 1
                            elem=lambda z: float_em.quantize(z, f.e, f.m))    # step 2


def dequantize(P: torch.Tensor, X: torch.Tensor, axis: int = 0, block_size: int = _driver.BLOCK) -> torch.Tensor:
    """V_hat = P * expand(X): each scale repeated block_size times along `axis`, cut to P's length."""
    return _driver.dequantize(P, X, axis, block_size)
