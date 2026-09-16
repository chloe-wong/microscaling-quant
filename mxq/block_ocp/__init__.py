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

from .. import _blocks
from ..scale_factor import ocp as _scale
from ..element_quant import microsoft as _elem
from ..element_quant.formats import Format, get

__all__ = ["quantize", "dequantize"]

BLOCK = 32


def quantize(V: torch.Tensor, fmt: Union[str, Format], axis: int = 0, block_size: int = BLOCK,
             round: str = "even", scale_bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """OCP block quantize V along `axis`. Returns (P codes, X power-of-two scales), float32."""
    V = V.to(torch.float32)
    f = get(fmt)
    b = _blocks.to_blocks(V, axis, block_size)
    if f is None:  # FP32 pass-through
        X = torch.ones(b.data.shape[:-1] + (1,), dtype=V.dtype, device=V.device)
        return _blocks.codes_from_blocks(b.data, b), _blocks.scales_from_blocks(X, b)
    amax = b.data.abs().amax(dim=-1, keepdim=True)
    X = _scale(amax, f.emax, scale_bits)
    P = _elem.quantize(b.data / X, f, round=round, saturate_normals=True, allow_denorm=True)
    return _blocks.codes_from_blocks(P, b), _blocks.scales_from_blocks(X, b)


def dequantize(P: torch.Tensor, X: torch.Tensor, axis: int = 0, block_size: int = BLOCK) -> torch.Tensor:
    return _blocks.dequantize(P, X, axis, block_size)
