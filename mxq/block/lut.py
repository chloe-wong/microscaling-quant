"""mxq.block.lut — any block quantizer followed by an mxq.lut table per group of codes.

    P, X = quantize(V, "MXFP6_E3M2", axis=0)                          V_hat = P * expand(X)   (same contract as block.<name>)
    P, X = quantize(V, "MXFP6_E3M2", base=block.mxquant, num_signposts=8)
    V_hat = dequantize(P, X, axis=0)

    V              tensor to quantize
    fmt            element format, passed to base.quantize (e.g. "MXFP6_E3M2"). The tables always snap to the
                   FP6 E3M2 codebook, so any other format warns: its codes are fine, its tables are E3M2
    axis           the axis blocks run along (0 for a K×N weight: blocks along K)
    block_size     values per block; one scale per block
    base           module whose quantize(V, fmt, axis, block_size) makes the codes: mxgemmini (default), mxquant, ...
    num_signposts  table entries per group (16 = 4-bit indices)
    iters          k-means iterations
    granularity    what shares one table:
                     mx       each block
                     channel  every block at the same index along V's first non-block axis
"""
import warnings
from types import ModuleType

import torch

from . import _driver, mxgemmini
from .. import lut
from ..element_quant.formats import get

__all__ = ["quantize", "dequantize"]


def quantize(V, fmt, axis=0, block_size=_driver.BLOCK, *,
             base: ModuleType = mxgemmini, num_signposts=16, iters=3, granularity='channel'):
    """base.quantize, then a num_signposts-entry table per group. Returns (P codes, X scales), float32."""
    f = get(fmt)
    if f is None or f.name != "MXFP6_E3M2":
        warnings.warn(f"block.lut: fmt={fmt!r}, but the LUT tables snap to the FP6 E3M2 codebook only: "
                      f"the codes are {fmt!r}, the table entries are E3M2 values", stacklevel=2)
    P, X = base.quantize(V, fmt=fmt, axis=axis, block_size=block_size)
    b = _driver.to_blocks(P, axis, block_size)                                  # (..., nblocks, block_size)

    if granularity == 'mx':
        rows = b.data.reshape(-1, block_size)                                   # one row per block
    elif granularity == 'channel':
        if b.data.ndim < 3:
            raise ValueError("granularity 'channel' needs V.ndim >= 2")
        moved = b.data.movedim(1, 0)                                            # one row per index along dim 1
        rows = moved.reshape(moved.shape[0], -1)
    else:
        raise ValueError(f"Unknown granularity: {granularity}. Must be 'mx' or 'channel'.")

    I, T = lut.fit(rows, num_signposts=num_signposts, iters=iters)
    P_lut = T.gather(1, I)

    if granularity == 'channel':
        P_lut = P_lut.reshape(moved.shape).movedim(0, 1)
    return _driver.codes_from_blocks(P_lut.reshape(b.data.shape), b), X


def dequantize(P: torch.Tensor, X: torch.Tensor, axis: int = 0, block_size: int = _driver.BLOCK) -> torch.Tensor:
    """V_hat = P * expand(X): each scale repeated block_size times along `axis`, cut to P's length."""
    return _driver.dequantize(P, X, axis, block_size)
