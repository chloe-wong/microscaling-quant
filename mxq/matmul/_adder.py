"""The adder tree, numerically: `size` products formed at once and folded by log2(size) levels of adjacent-pair
adds, level l rounding to its own float(e, m) from `schedule`; the root is rescaled and added to the output.
Same call as systolic; a `schedule` entry is a tree level, level 1 first. Timing is not modelled.

Every node is `arith.acc_add`, the add-and-round one systolic lane does (hardware-checked for MXGEMMINI). The
arrangement of the nodes into a tree is not checked against any hardware: no RTL for it exists. The pairing is
fixed, (0,1), (2,3), ..., because a floating-point tree's result depends on it. K tail: exact zeros.
"""
from typing import Sequence, Tuple

import torch

from ..block import BLOCK
from ._common import check_operands, check_schedule, scale_map
from ._arithmetic import Arithmetic

__all__ = ["adder_tree"]


def adder_tree(P_A: torch.Tensor, X_A: torch.Tensor, P_B: torch.Tensor, X_B: torch.Tensor,
               arith: Arithmetic, schedule: Sequence[Tuple[int, int]], size: int = 16,
               block_size: int = BLOCK) -> torch.Tensor:
    """Y = Aᵀ·B. A: K×M codes, X_A: ceil(K/block_size)×M scales; B: K×N, X_B: ceil(K/block_size)×N. Y: M×N float32.

    For each block of K: the block's scale map; for each reduction of `size` k inside it: all products at once by
    `arith.product`, then log2(size) levels of adjacent-pair additions by `arith.acc_add`, level l in
    `schedule[l]`'s format; the root is rescaled and added to the output by `arith.tile_add`.
    `size` must be a power of two dividing `block_size`.
    """
    K, M, N = check_operands(P_A, X_A, P_B, X_B, block_size)
    levels = size.bit_length() - 1
    if size < 2 or (1 << levels) != size:
        raise ValueError(f"size {size} must be a power of two >= 2")
    if block_size % size != 0:
        raise ValueError(f"size {size} must divide block size {block_size}")
    check_schedule(schedule, levels)
    C = torch.zeros((M, N), dtype=torch.float32, device=P_A.device)
    for g in range(0, K, block_size):
        g_end = min(g + block_size, K)
        scales = scale_map(X_A, X_B, g // block_size)
        for k0 in range(g, g_end, size):
            k1 = min(k0 + size, g_end)
            p = arith.product(P_A[k0:k1].unsqueeze(2), P_B[k0:k1].unsqueeze(1))        # (k1-k0)×M×N, all at once
            if k1 - k0 < size:
                p = torch.cat([p, torch.zeros((size - (k1 - k0), M, N), dtype=p.dtype, device=p.device)])
            for level in range(levels):
                e, m = schedule[level]
                p = arith.acc_add(p[0::2], p[1::2], e, m)                              # adjacent pairs
            C = arith.tile_add(C, p[0] * scales)
    return C.to(torch.float32)
