"""The inner-product tree, numerically: `fanin` products formed at once and folded by an adder tree of
log2(fanin) levels, each level rounding to its own float(e, m) from `schedule`; the root is rescaled and
accumulated into the output. Order of operations is a dot-product unit's; timing is not modelled.

Default fanin 16 groups the same 16 k's as systolic's default window, so systolic and ipt then differ only in
the order of additions inside a group (chain vs tree) and in which format each partial sum is rounded to.
"""
from typing import Sequence, Tuple

import torch

from .. import _blocks
from ._common import check_operands, check_schedule, scale_map
from .arithmetic import Arithmetic

__all__ = ["ipt"]


def ipt(P_A: torch.Tensor, X_A: torch.Tensor, P_B: torch.Tensor, X_B: torch.Tensor,
        arith: Arithmetic, schedule: Sequence[Tuple[int, int]], fanin: int = 16,
        block_size: int = _blocks.BLOCK) -> torch.Tensor:
    """Y = Aᵀ·B. A: K×M codes, X_A: ceil(K/block_size)×M scales; B: K×N, X_B: ceil(K/block_size)×N. Y: M×N float32.

    For each block of K: the block's scale map; for each group of `fanin` k's inside it: all products at once by
    `arith.product`, then log2(fanin) levels of adjacent-pair additions by `arith.acc_add`, level l in
    `schedule[l]`'s format; the root is rescaled and added to the output by `arith.tile_add`.
    `fanin` must be a power of two dividing `block_size`; a K tail is padded with exact zeros.
    """
    K, M, N = check_operands(P_A, X_A, P_B, X_B, block_size)
    levels = fanin.bit_length() - 1
    if fanin < 2 or (1 << levels) != fanin:
        raise ValueError(f"fanin {fanin} must be a power of two >= 2")
    if block_size % fanin != 0:
        raise ValueError(f"fanin {fanin} must divide block size {block_size}")
    check_schedule(schedule, levels)
    C = torch.zeros((M, N), dtype=torch.float32, device=P_A.device)

    for g in range(0, K, block_size):
        g_end = min(g + block_size, K)
        scales = scale_map(X_A, X_B, g // block_size)
        for k0 in range(g, g_end, fanin):
            k1 = min(k0 + fanin, g_end)
            p = arith.product(P_A[k0:k1].unsqueeze(2), P_B[k0:k1].unsqueeze(1))       # (k1-k0)×M×N, all at once
            if k1 - k0 < fanin:
                p = torch.cat([p, torch.zeros((fanin - (k1 - k0), M, N), dtype=p.dtype, device=p.device)])
            for level in range(levels):
                e, m = schedule[level]
                p = arith.acc_add(p[0::2], p[1::2], e, m)                             # adjacent pairs
            C = arith.tile_add(C, p[0] * scales)
    return C.to(torch.float32)
