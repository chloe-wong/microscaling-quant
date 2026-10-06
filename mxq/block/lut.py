"""mxq.block.lut — MX-Gemmini's LUT operand: block.mxgemmini, then one 16-entry table per 2^group columns.

    P, X = quantize(V, "MXFP6_E3M2", axis=0, block_size=32, rounding_mode="rne", scale_floor=2**-23,
                    group=1, max_iters=50)                                  V_hat = P * expand(X)
    V_hat = dequantize(P, X, axis=0)

Same contract as block.<name>: P has V's shape, X one scale per block along `axis`. V is 2-D; the tables
group along the other axis, 2^group rows/columns per table spanning all of `axis` (K), the last table taking
what is left over. For a Scheme operand (K×M or K×N, axis=0) that is 2^group rows of A or columns of B, as on
the chip (whose loader takes whole groups only; a partial one exists here for arbitrary token counts).

Every setting is required: the chip's are in the hardware and run recipes, and a default here would be a
second place for them. See mxq.lut for the rule.
"""
import torch

from . import _driver, mxgemmini
from .. import lut

__all__ = ["quantize", "dequantize"]


def quantize(V: torch.Tensor, fmt, axis: int, block_size: int, *, rounding_mode: str, scale_floor: float,
             group: int, max_iters: int):
    """block.mxgemmini.quantize, then each code replaced by its table entry. Returns (P, X), float32."""
    if V.ndim != 2 or axis not in (0, 1, -1, -2):
        raise ValueError(f"block.lut: V must be 2-D with axis 0 or 1, got shape {tuple(V.shape)}, axis {axis}")
    lut.formats._format(fmt)                                                     # refuse a non-LUT format first
    P, X = mxgemmini.quantize(V, fmt, axis=axis, block_size=block_size, rounding_mode=rounding_mode,
                              scale_floor=scale_floor)
    Pk = P if axis % 2 == 0 else P.t()                                           # K×n
    T = lut.tables(Pk, fmt, group=group, max_iters=max_iters)
    P_lut = lut.lookup(lut.pick(Pk, T, group=group), T, group=group)
    return (P_lut if axis % 2 == 0 else P_lut.t()).contiguous(), X


def dequantize(P: torch.Tensor, X: torch.Tensor, axis: int = 0, block_size: int = _driver.BLOCK) -> torch.Tensor:
    """V_hat = P * expand(X): each scale repeated block_size times along `axis`, cut to P's length."""
    return _driver.dequantize(P, X, axis, block_size)
