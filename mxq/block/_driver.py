"""The block quantization driver, `compose` (public as mxq.block.compose), and its plumbing: split a tensor into
blocks along one axis, pad, run the two steps, and put codes / scales back into the caller's layout.

Layout contract (same as MXQuant's mx_block32_quantize):
  quantize(V, axis) -> (P, X)   P: same shape as V, the codes
                                X: V.shape with V.shape[axis] replaced by ceil(V.shape[axis]/block_size)
  V_hat = P * expand(X)  where expand repeats each scale block_size times along axis.
"""
from typing import Callable, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F

BLOCK = 32


class Blocks(NamedTuple):
    data: torch.Tensor   # (..., nblocks, block_size), block axis moved last, zero-padded
    axis: int
    n: int               # original length along axis


def to_blocks(V: torch.Tensor, axis: int, block_size: int) -> Blocks:
    if not -V.ndim <= axis < V.ndim:
        raise IndexError(f"axis {axis} out of range for {V.ndim}-D tensor")
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


def compose(V: torch.Tensor, *, scale: Optional[Callable[[torch.Tensor], torch.Tensor]],
            elem: Optional[Callable[[torch.Tensor], torch.Tensor]],
            axis: int = 0, block_size: int = BLOCK) -> Tuple[torch.Tensor, torch.Tensor]:
    """Block quantization from its two steps. Returns (P codes, X scales), float32.

    scale(amax) -> X   step 1: amax is each block's max |value|, shape (..., nblocks, 1); X the same shape
    elem(z) -> P       step 2: z = block / X, shape (..., nblocks, block_size); P the same shape

    Both are required and either may be None: no scale step means X = 1, no element step means P = V / X.
    Both None is the FP32 pass-through. V is split into blocks of `block_size` along `axis` (the last block
    zero-padded) and computed in float32; P has V's shape, X has V's shape with `axis` of length
    ceil(len / block_size). block.mxquant, block.mxgemmini and block.ocp are this with a fixed pair."""
    V = V.to(torch.float32)
    b = to_blocks(V, axis, block_size)
    if scale is None:
        X = torch.ones(b.data.shape[:-1] + (1,), dtype=torch.float32, device=V.device)
        z = b.data
    else:
        X = scale(b.data.abs().amax(dim=-1, keepdim=True))
        z = b.data / X
    P = z.clone() if elem is None else elem(z)
    return codes_from_blocks(P, b), scales_from_blocks(X, b)


def dequantize(P: torch.Tensor, X: torch.Tensor, axis: int = 0, block_size: int = BLOCK) -> torch.Tensor:
    axis = axis % P.ndim
    Xe = torch.repeat_interleave(X, block_size, dim=axis).narrow(axis, 0, P.shape[axis])
    return P * Xe
