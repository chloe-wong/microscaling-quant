"""Shared plumbing for block quantizers: split a tensor into blocks along one axis, pad, run
the two steps, and put codes / scales back into the caller's layout.

Layout contract (same as MXQuant's mx_block32_quantize):
  quantize(V, axis) -> (P, X)   P: same shape as V, the codes
                                X: V.shape with V.shape[axis] replaced by ceil(V.shape[axis]/block_size)
  V_hat = P * expand(X)  where expand repeats each scale block_size times along axis.
"""
from typing import Callable, NamedTuple, Tuple

import torch
import torch.nn.functional as F

BLOCK = 32


class Blocks(NamedTuple):
    data: torch.Tensor   # (..., nblocks, block_size), block axis moved last, zero-padded
    axis: int
    n: int               # original length along axis


def to_blocks(V: torch.Tensor, axis: int, block_size: int) -> Blocks:
    axis = axis % V.ndim
    Vt = V.movedim(axis, -1)
    n = Vt.shape[-1]
    pad = (-n) % block_size
    if pad:
        Vt = F.pad(Vt, (0, pad))
    return Blocks(Vt.reshape(*Vt.shape[:-1], -1, block_size), axis, n)


def codes_from_blocks(P: torch.Tensor, b: Blocks) -> torch.Tensor:
    """(..., nblocks, block_size) -> original layout, padding removed."""
    return P.reshape(*P.shape[:-2], -1)[..., : b.n].movedim(-1, b.axis).contiguous()


def scales_from_blocks(X: torch.Tensor, b: Blocks) -> torch.Tensor:
    """(..., nblocks, 1) -> layout of V with axis length nblocks."""
    return X.squeeze(-1).movedim(-1, b.axis).contiguous()


def quantize(V: torch.Tensor, axis: int, block_size: int, passthrough: bool,
             scale: Callable[[torch.Tensor], torch.Tensor],
             elem: Callable[[torch.Tensor], torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Driver: block V, then X = scale(amax) (step 1) and P = elem(V / X) (step 2).
    passthrough (FP32): identity codes, unit scales. Computes in float32."""
    V = V.to(torch.float32)
    b = to_blocks(V, axis, block_size)
    if passthrough:
        P, X = b.data, torch.ones(b.data.shape[:-1] + (1,), dtype=torch.float32, device=V.device)
    else:
        X = scale(b.data.abs().amax(dim=-1, keepdim=True))
        P = elem(b.data / X)
    return codes_from_blocks(P, b), scales_from_blocks(X, b)


def dequantize(P: torch.Tensor, X: torch.Tensor, axis: int = 0, block_size: int = BLOCK) -> torch.Tensor:
    axis = axis % P.ndim
    Xe = torch.repeat_interleave(X, block_size, dim=axis).narrow(axis, 0, P.shape[axis])
    return P * Xe
