"""Checks and helpers shared by every reducer."""
import math
from typing import Sequence, Tuple

import torch


def check_operands(P_A, X_A, P_B, X_B, block_size: int) -> Tuple[int, int, int]:
    """A: K×M with X_A: ceil(K/block_size)×M; B: K×N with X_B: ceil(K/block_size)×N; one device. Returns (K, M, N)."""
    K, M = P_A.shape
    K_b, N = P_B.shape
    if K != K_b:
        raise ValueError(f"A is {K}×{M}, B is {K_b}×{N}: contraction lengths differ")
    nb = math.ceil(K / block_size)
    if tuple(X_A.shape) != (nb, M) or tuple(X_B.shape) != (nb, N):
        raise ValueError(f"scales must be {nb}×{M} and {nb}×{N} for block size {block_size}, got {tuple(X_A.shape)} and {tuple(X_B.shape)}")
    if len({t.device for t in (P_A, X_A, P_B, X_B)}) != 1:
        raise ValueError("codes and scales must be on one device")
    if any(t.dtype != torch.float32 for t in (P_A, X_A, P_B, X_B)):
        raise ValueError("codes and scales must be float32")
    return K, M, N


def check_schedule(schedule: Sequence[Tuple[int, int]], rows: int) -> None:
    """A schedule is fully defined when it has exactly one (e, m) row per accumulator position."""
    if len(schedule) != rows:
        raise ValueError(f"schedule has {len(schedule)} rows, reducer has {rows} accumulator positions")


def scale_map(X_A: torch.Tensor, X_B: torch.Tensor, g: int) -> torch.Tensor:
    """Outer product of block g's scales: the factor a finished block sum is multiplied by. Powers of two, exact."""
    return X_A[g].unsqueeze(1) * X_B[g].unsqueeze(0)
