"""The anchor tree, numerically: MxGen's MxAnchorAccTree (src/main/scala/configs/dot-product/MxAnchorAccTree.scala,
MxGen a27ce3c) as a reducer with systolic's call. Order of operations is the tree's; timing is not modelled.

A reduction of `size` products is `size / width` trees chained through the running sum c. One tree:

    S1  anchor = the largest label among the width products and c, plus headroom        l.79-83
        label of a product = e(a) + e(b), e(x) = floor(log2|x|) and at least `emin`        MxExp.AddBit
        label of c = floor(log2|c|) - 1 (its significand sits one bit higher)              l.52-57
        a zero never sets the anchor                                                       l.62-65
    S2  every term onto the grid 2^(anchor + 2 - bits[0]); what is below it is dropped     l.97-118
    S3  adjacent pairs added, products first and c last, an odd one passed up              l.67-77, l.121
        after level l the partial sums go onto the grid 2^(anchor + 2 - bits[l])          (RTL: all equal, a no-op)
    S4  the sum rounded once to schedule[t] by the Arithmetic's lane rounding              l.129-168

Defaults are the RTL's: headroom max(4, ceil(log2(width + 1)) + 1); every level (m + 1) + headroom +
ceil(log2(width + 1)) + 1 bits; drop "truncate" (a right shift of the magnitude, no sticky bit).
Two RTL behaviours belong to the Arithmetic, not the tree: MxGen's S4 saturates to max-normal and maps NaN
to 0, where MXGEMMINI's lane overflows to Inf and keeps NaN.
K tail: missing products are zeros, as hardware feeds them.
"""
import math
from typing import Optional, Sequence, Tuple

import torch

from ..block import BLOCK
from ._common import check_operands, check_schedule, scale_map
from ._arithmetic import Arithmetic

__all__ = ["anchor_tree"]

DROPS = ("truncate", "rne")


def _log2_floor(x: torch.Tensor) -> torch.Tensor:
    """floor(log2|x|) as float64 for x != 0; the value at 0 is never used."""
    return (torch.frexp(x.abs())[1] - 1).to(torch.float64)


def _keep(x: torch.Tensor, drop: str) -> torch.Tensor:
    """An exact float64 quotient made an integer: toward zero, or to nearest even."""
    return torch.trunc(x) if drop == "truncate" else torch.round(x)


def _tree(p, label, c, e, m, bits, drop, headroom, arith):
    """p, label: width x M x N products and labels; c: M x N running sum. Returns the new c, float32."""
    neg = torch.full(c.shape, -math.inf, dtype=torch.float64, device=c.device)
    top = torch.where(p != 0, label, neg).amax(0)
    top = torch.maximum(top, torch.where(c != 0, _log2_floor(c) - 1, neg))
    anchor = torch.where(torch.isfinite(top), top, torch.zeros_like(top)) + headroom
    x = torch.cat([p.to(torch.float64), c.to(torch.float64).unsqueeze(0)])
    x = _keep(x / torch.exp2(anchor + 2 - bits[0]), drop)                   # S2: exact, a power of two
    for level in range(1, len(bits)):                                          # S3
        n = x.shape[0]
        pairs = x[0:n - n % 2:2] + x[1:n - n % 2:2]
        x = torch.cat([pairs, x[n - 1:]]) if n % 2 else pairs                 # an odd one passed up
        if bits[level] != bits[level - 1]:
            x = _keep(x / 2.0 ** (bits[level - 1] - bits[level]), drop)
    s = (x[0] * torch.exp2(anchor + 2 - bits[-1])).to(torch.float32)         # exact: |x| < 2^23
    return arith.acc_add(torch.zeros_like(c), s, e, m)                         # S4: the lane rounding


def anchor_tree(P_A: torch.Tensor, X_A: torch.Tensor, P_B: torch.Tensor, X_B: torch.Tensor,
                arith: Arithmetic, schedule: Optional[Sequence[Tuple[int, int]]] = None, size: int = 16,
                block_size: int = BLOCK, *, width: Optional[int] = None, bits: Optional[Sequence[int]] = None,
                drop: str = "truncate", headroom: Optional[int] = None, emin: Optional[int] = None) -> torch.Tensor:
    """Y = Aᵀ·B, each reduction of `size` products summed by chained anchor trees of `width`. Same operands,
    output, `arith`, `size` and `block_size` as systolic; `schedule` has one (e, m) per tree (default bf16)."""
    K, M, N = check_operands(P_A, X_A, P_B, X_B, block_size)
    width = size if width is None else width
    if width < 1 or size % width or block_size % size:
        raise ValueError(f"width {width} must divide size {size}, and size must divide block size {block_size}")
    trees = size // width
    schedule = [(8, 7)] * trees if schedule is None else list(schedule)
    check_schedule(schedule, trees)
    if drop not in DROPS:
        raise ValueError(f"drop must be one of {DROPS}, got {drop!r}")
    lg = math.ceil(math.log2(width + 1))                                       # levels for width + 1 terms
    head = max(4, lg + 1) if headroom is None else headroom
    if head < lg + 1:
        raise ValueError(f"headroom {head} < {lg + 1}: the sum of {width + 1} terms could overflow")

    def level_bits(m):
        b = [(m + 1) + head + lg + 1] * (lg + 1) if bits is None else list(bits)
        if len(b) != lg + 1 or max(b) > 24 or any(x < y for x, y in zip(b, b[1:])):
            raise ValueError(f"bits needs {lg + 1} non-increasing entries, each <= 24; got {b}")
        return b

    per_tree = [(e, m, level_bits(m)) for e, m in schedule]
    la, lb = _log2_floor(P_A), _log2_floor(P_B)
    if emin is not None:
        la, lb = la.clamp(min=emin), lb.clamp(min=emin)
    dev = P_A.device
    C = torch.zeros((M, N), dtype=torch.float32, device=dev)
    for g in range(0, K, block_size):
        g_end = min(g + block_size, K)
        scales = scale_map(X_A, X_B, g // block_size)
        for k0 in range(g, g_end, size):
            c = torch.zeros((M, N), dtype=torch.float32, device=dev)
            for t, (e, m, b) in enumerate(per_tree):
                lo, hi = k0 + t * width, min(k0 + (t + 1) * width, g_end)
                p = torch.zeros((width, M, N), dtype=torch.float32, device=dev)
                label = torch.zeros((width, M, N), dtype=torch.float64, device=dev)
                if hi > lo:
                    p[:hi - lo] = arith.product(P_A[lo:hi].unsqueeze(2), P_B[lo:hi].unsqueeze(1))
                    label[:hi - lo] = la[lo:hi].unsqueeze(2) + lb[lo:hi].unsqueeze(1)
                c = _tree(p, label, c, e, m, b, drop, head, arith)
            C = arith.tile_add(C, c * scales)
    return C.to(torch.float32)
