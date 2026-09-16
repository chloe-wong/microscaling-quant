"""The 16-deep PE column, numerically: MXQuant `_simulate_atw` loop (eval_complete.py:547-579) with the three
rounding points taken from an Arithmetic. Order of operations is the hardware's; timing is not modelled."""
from typing import Sequence, Tuple

import torch

from .arithmetic import Arithmetic

__all__ = ["systolic"]


def systolic(P_A: torch.Tensor, X_A: torch.Tensor, P_B: torch.Tensor, X_B: torch.Tensor,
             arith: Arithmetic, schedule: Sequence[Tuple[int, int]], window: int = 16, block: int = 32) -> torch.Tensor:
    """Y = Aᵀ·B. A: K×M codes with X_A: ceil(K/block)×M scales; B: K×N with X_B: ceil(K/block)×N. Y: M×N float32.

    For each block of K: the block's scale map X_A[g] ⊗ X_B[g]; for each window of k inside it: a fresh lane sum,
    one product per k rounded by `arith.product`, accumulated by `arith.lane_add` in lane k % window's format from
    `schedule`; the finished window sum is rescaled and added to the output by `arith.tile_add`.
    """
    K, M = P_A.shape
    K_b, N = P_B.shape
    if K != K_b:
        raise ValueError(f"A is {K}×{M}, B is {K_b}×{N}: contraction lengths differ")
    if len(schedule) < window:
        raise ValueError(f"schedule has {len(schedule)} lanes, window is {window}")
    C = torch.zeros((M, N), dtype=torch.float32, device=P_A.device)

    for g in range(0, K, block):
        g_end = min(g + block, K)
        scale_map = X_A[g // block].unsqueeze(1) * X_B[g // block].unsqueeze(0)
        for k_base in range(g, g_end, window):
            S = torch.zeros((M, N), dtype=torch.float32, device=P_A.device)
            for k in range(k_base, min(k_base + window, g_end)):
                p = arith.product(P_A[k].unsqueeze(1), P_B[k].unsqueeze(0))
                e, m = schedule[k % window]
                S = arith.lane_add(S, p, e, m)
            C = arith.tile_add(C, S * scale_map)
    return C
