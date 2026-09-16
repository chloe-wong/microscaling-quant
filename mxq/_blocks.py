"""Shared plumbing for block quantizers: split a tensor into blocks along one axis, pad,
and put codes / scales back into the caller's layout.

Layout contract (same as MXQuant's mx_block32_quantize):
  quantize(V, axis) -> (P, X)   P: same shape as V, the codes
                                X: V.shape with V.shape[axis] replaced by ceil(V.shape[axis]/block_size)
  V_hat = P * expand(X)  where expand repeats each scale block_size times along axis.
"""
from typing import NamedTuple

import torch
import torch.nn.functional as F

__all__ = ["Blocks", "to_blocks", "codes_from_blocks", "scales_from_blocks", "expand_scales", "dequantize"]


class Blocks(NamedTuple):
    data: torch.Tensor   # (..., nblocks, block_size), block axis moved last, zero-padded
    axis: int
    n: int               # original length along axis
    block_size: int


def to_blocks(V: torch.Tensor, axis: int, block_size: int) -> Blocks:
    axis = axis % V.ndim
    Vt = V.movedim(axis, -1)
    n = Vt.shape[-1]
    pad = (-n) % block_size
    if pad:
        Vt = F.pad(Vt, (0, pad))
    nb = Vt.shape[-1] // block_size
    return Blocks(Vt.reshape(*Vt.shape[:-1], nb, block_size), axis, n, block_size)


def codes_from_blocks(P: torch.Tensor, b: Blocks) -> torch.Tensor:
    """(..., nblocks, block_size) -> original layout, padding removed."""
    Pt = P.reshape(*P.shape[:-2], -1)[..., : b.n]
    return Pt.movedim(-1, b.axis).contiguous()


def scales_from_blocks(X: torch.Tensor, b: Blocks) -> torch.Tensor:
    """(..., nblocks, 1) -> layout of V with axis length nblocks."""
    return X.squeeze(-1).movedim(-1, b.axis).contiguous()


def expand_scales(X: torch.Tensor, axis: int, n: int, block_size: int) -> torch.Tensor:
    axis = axis % X.ndim
    Xe = torch.repeat_interleave(X, block_size, dim=axis)
    return Xe.narrow(axis, 0, n)


def dequantize(P: torch.Tensor, X: torch.Tensor, axis: int, block_size: int = 32) -> torch.Tensor:
    axis = axis % P.ndim
    return P * expand_scales(X, axis, P.shape[axis], block_size)
