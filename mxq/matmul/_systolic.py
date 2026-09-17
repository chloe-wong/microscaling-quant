"""The systolic column, numerically: MXQuant's `MXLinearSim._simulate_atw` loop with the three rounding points
taken from an Arithmetic. Order of operations is the hardware's; timing is not modelled.

The mesh width never changes a value (each output element is its own K-sum); the column depth is `window`,
and the schedule has one float(e, m) per lane, i.e. exactly `window` entries. HW_FINAL is the tapeout's.
"""
from typing import List, Sequence, Tuple

import torch

from .. import _blocks
from ._common import check_operands, check_schedule, scale_map
from .arithmetic import Arithmetic

__all__ = ["systolic", "HW_FINAL"]

#: MX-Gemmini tapeout lanes (schedule_hw_final.csv, rtl_exact/acc_schedule.csv): 0-7 e4m4, 8-9 e4m5, 10-14 e4m6, 15 e8m7
HW_FINAL: List[Tuple[int, int]] = [(4, 4)] * 8 + [(4, 5)] * 2 + [(4, 6)] * 5 + [(8, 7)]


def systolic(P_A: torch.Tensor, X_A: torch.Tensor, P_B: torch.Tensor, X_B: torch.Tensor,
             arith: Arithmetic, schedule: Sequence[Tuple[int, int]], window: int = 16,
             block_size: int = _blocks.BLOCK) -> torch.Tensor:
    """Y = Aᵀ·B. A: K×M codes, X_A: ceil(K/block_size)×M scales; B: K×N, X_B: ceil(K/block_size)×N. Y: M×N float32.

    For each block of K: the block's scale map; for each window of k inside it: a fresh sum S, one product per k
    rounded by `arith.product`, accumulated by `arith.acc_add` in lane k % window's format from `schedule`;
    the finished window sum is rescaled and added to the output by `arith.tile_add`.
    `window` must divide `block_size`, otherwise a window would straddle two scale blocks.
    """
    K, M, N = check_operands(P_A, X_A, P_B, X_B, block_size)
    check_schedule(schedule, window)
    if block_size % window != 0:
        raise ValueError(f"window {window} must divide block size {block_size}")
    C = torch.zeros((M, N), dtype=torch.float32, device=P_A.device)

    for g in range(0, K, block_size):
        g_end = min(g + block_size, K)
        scales = scale_map(X_A, X_B, g // block_size)
        for k_base in range(g, g_end, window):
            S = torch.zeros((M, N), dtype=torch.float32, device=P_A.device)
            for k in range(k_base, min(k_base + window, g_end)):
                p = arith.product(P_A[k].unsqueeze(1), P_B[k].unsqueeze(0))
                e, m = schedule[k % window]
                S = arith.acc_add(S, p, e, m)
            C = arith.tile_add(C, S * scales)
    return C.to(torch.float32)
