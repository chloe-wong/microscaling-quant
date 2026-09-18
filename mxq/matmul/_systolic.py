"""The systolic column, numerically: MXQuant's `MXLinearSim._simulate_atw` loop with the three rounding points
taken from an Arithmetic. Order of operations is the hardware's; timing is not modelled.

The mesh width never changes a value (each output element is its own K-sum); the column depth is `window`,
and the schedule has one float(e, m) per lane, i.e. exactly `window` entries. schedule.HW_FINAL is the tapeout's.

K tail (K not a multiple of `window`): the last window's sum leaves the column after its last real product, as
in `_simulate_atw`. Hardware pads the tail with zero products that still pass through the remaining lanes;
with HW_FINAL every later lane holds every value of the lane before it, so the zeros change nothing and the two
agree. A schedule with a narrower lane after a wider one would differ; there mxq follows MXQuant.
"""
from typing import Sequence, Tuple

import torch

from ..block import BLOCK
from ._common import check_operands, check_schedule, scale_map
from ._arithmetic import Arithmetic

__all__ = ["systolic"]


def systolic(P_A: torch.Tensor, X_A: torch.Tensor, P_B: torch.Tensor, X_B: torch.Tensor,
             arith: Arithmetic, schedule: Sequence[Tuple[int, int]], window: int = 16,
             block_size: int = BLOCK) -> torch.Tensor:
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
