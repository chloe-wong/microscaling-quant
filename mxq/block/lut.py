"""mxq.block.lut — any block quantizer followed by a look-up table per block.

    P, X = quantize(V, "MXFP6_E3M2", axis=0)                          V_hat = P * expand(X)   (same contract as block.<name>)
    P, X = quantize(V, "MXFP6_E3M2", base=block.mxquant, num_signposts=8)
    V_hat = dequantize(P, X, axis=0)

    V              tensor to quantize
    fmt            element format, passed to base.quantize (e.g. "MXFP6_E3M2")
    axis           the axis blocks run along (0 for a K×N weight: blocks along K)
    block_size     values per block; one scale and one table per block. level2 requires 32
    base           module whose quantize(V, fmt, axis, block_size) makes the codes: mxgemmini (default), mxquant, ...
    num_signposts  table entries per block (16 = 4-bit indices)
    iters          k-means iterations

1. P, X = base.quantize(...): X = each block's scale, P = V / X rounded onto fmt's grid.
2. Each block of P picks num_signposts centers by k-means (_lut/level2.py), snapped to E3M2_CODEBOOK.
3. Each code in the block is replaced by its nearest center; P keeps V's shape.

X is base's, unchanged: V_hat = dequantize(P, X) = P * expand(X), as for every block quantizer.
Tables snap to E3M2 up to 14, so fmt must be MXFP6_E3M2 and base's codes within ±14 (not block.ocp).
Needs a GPU (level2's layer counter divides by the device count).

    python -m mxq.block.lut        one weight through FP6 and FP6 + LUT16/8/4
"""
from types import ModuleType
import torch

from . import _driver, mxgemmini
from ._lut.level2 import _quantize_level2
from ..element_quant.formats import Format, get

__all__ = ["quantize", "dequantize"]


def quantize(V, fmt, axis=0, block_size=_driver.BLOCK, *,
             base: ModuleType = mxgemmini, num_signposts=16, iters=3):
    """base.quantize, then a num_signposts-entry table per block. Returns (P codes, X scales), float32."""
    P, X = base.quantize(V, fmt=fmt, axis=axis, block_size=block_size)
    b = _driver.to_blocks(P, axis, block_size)                                  
    P_lut = _quantize_level2(b.data, num_signposts=num_signposts, iters=iters, group_size=block_size)
    return _driver.codes_from_blocks(P_lut, b), X                                


def dequantize(P: torch.Tensor, X: torch.Tensor, axis: int = 0, block_size: int = _driver.BLOCK) -> torch.Tensor:
    """V_hat = P * expand(X): each scale repeated block_size times along `axis`, cut to P's length."""
    return _driver.dequantize(P, X, axis, block_size)

if __name__ == "__main__":
    torch.manual_seed(0)
    W = torch.randn(4096, 1024, device="cuda")
    err = lambda P, X: ((dequantize(P, X) - W).pow(2).mean() / W.pow(2).mean()).item()

    print(f"FP6          rel. MSE {err(*mxgemmini.quantize(W, 'MXFP6_E3M2')):.3e}")
    for k in (16, 8, 4):
        print(f"FP6 + LUT{k:<2}  rel. MSE {err(*quantize(W, 'MXFP6_E3M2', num_signposts=k)):.3e}")
