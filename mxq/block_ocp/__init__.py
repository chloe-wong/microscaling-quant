"""mxq.block_ocp — OCP Microscaling v1.0 block quantization, returning codes and scales.

    P, X = quantize(V, "MXFP8_E4M3", axis=0)        V_hat = P * expand(X)
    V_hat = dequantize(P, X, axis=0)

= scale_factor.ocp (shared exponent = floor(log2 amax) - emax, E8M0) followed by
  element_quant.microsoft (Microsoft's _quantize_elemwise, saturate, subnormals kept).

P * expand(X) is bit-identical to Microsoft's `_quantize_mx` (mxq.ocp.block_quantize) for the
same `round`. Unlike the reference, codes and scales are returned separately so a systolic
simulation can multiply codes and apply X_A * X_B once per block.
"""
from typing import Tuple, Union

import torch

from .. import _blocks, scale_factor
from ..element_quant import microsoft
from ..element_quant.formats import Format, get

__all__ = ["quantize", "dequantize"]


def quantize(V: torch.Tensor, fmt: Union[str, Format], axis: int = 0, block_size: int = _blocks.BLOCK,
             round: str = "even", scale_bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """OCP block quantize V along `axis`. Returns (P codes, X power-of-two scales), float32."""
    f = get(fmt)
    return _blocks.quantize(V, axis, block_size, passthrough=f is None,
                            scale=lambda amax: scale_factor.ocp(amax, f.emax, scale_bits),   # step 1
                            elem=lambda z: microsoft.quantize(z, f, round=round))            # step 2


dequantize = _blocks.dequantize
