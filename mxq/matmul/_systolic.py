"""The systolic column, numerically: MXQuant's `MXLinearSim._simulate_atw` loop with the three rounding points
taken from an Arithmetic. Order of operations is the hardware's; timing is not modelled.

The mesh width never changes a value (each output element is its own K-sum); the column depth is `size`,
and the schedule has one float(e, m) per lane, i.e. exactly `size` entries. schedule.HW_FINAL is the tapeout's.

K tail (K not a multiple of `size`): the last reduction's sum leaves the column after its last real product, as
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
             arith: Arithmetic, schedule: Sequence[Tuple[int, int]], size: int = 16,
             block_size: int = BLOCK) -> torch.Tensor:
    """Y = Aᵀ·B. A: K×M codes, X_A: ceil(K/block_size)×M scales; B: K×N, X_B: ceil(K/block_size)×N. Y: M×N float32.

    For each block of K: the block's scale map; for each reduction of `size` k inside it: a fresh sum S, one product per k
    rounded by `arith.product`, accumulated by `arith.acc_add` in lane k % size's format from `schedule`;
    the finished sum is rescaled and added to the output by `arith.tile_add`.
    `size` must divide `block_size`, otherwise a reduction would straddle two scale blocks.
    """
    K, M, N = check_operands(P_A, X_A, P_B, X_B, block_size)
    check_schedule(schedule, size)
    if block_size % size != 0:
        raise ValueError(f"size {size} must divide block size {block_size}")
    C = torch.zeros((M, N), dtype=torch.float32, device=P_A.device)
    # An Arithmetic may offer to do a whole reduction at once (mxq.matmul.compiled does). Same operations in the
    # same order; it is one fused kernel instead of one per step. A short tail reduction is a different shape, so
    # it stays on the step-by-step path rather than forcing a rebuild per tail length.
    body = arith.fused_reduction(schedule, size) if getattr(arith, "fused_reduction", None) else None

    for g in range(0, K, block_size):
        g_end = min(g + block_size, K)
        scales = scale_map(X_A, X_B, g // block_size)
        for k_base in range(g, g_end, size):
            k_end = min(k_base + size, g_end)
            if body is not None and k_end - k_base == size:
                S = body(P_A[k_base:k_end], P_B[k_base:k_end])
            else:
                S = torch.zeros((M, N), dtype=torch.float32, device=P_A.device)
                for k in range(k_base, k_end):
                    p = arith.product(P_A[k].unsqueeze(1), P_B[k].unsqueeze(0))
                    e, m = schedule[k % size]
                    S = arith.acc_add(S, p, e, m)
            C = arith.tile_add(C, S * scales)
    return C.to(torch.float32)
