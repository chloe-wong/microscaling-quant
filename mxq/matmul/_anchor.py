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

Speed: with a compiled Arithmetic (mxq.matmul.compiled) a whole reduction, every tree and every level, is
one compiled function (`Arithmetic.fuse`), as systolic's fused_reduction is. Same operations in the same order; the terms are a
list of M x N tensors rather than one stacked tensor, so nothing is written to memory between steps.
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
    """floor(log2|x|) as float64 for x != 0; the value at 0 is never used. Inf and NaN (a narrow lane that
    overflowed upstream) get -1, the value eager frexp gives them; compiled frexp gives another, which moved the
    anchor and turned an Inf into a NaN. Any finite value would do: an Inf or NaN term decides the sum alone."""
    x = torch.where(torch.isfinite(x), x, torch.full_like(x, 0.5))
    return (torch.frexp(x.abs())[1] - 1).to(torch.float64)


def _keep(x: torch.Tensor, drop: str) -> torch.Tensor:
    """An exact float64 quotient made an integer: toward zero, or to nearest even."""
    return torch.trunc(x) if drop == "truncate" else torch.round(x)


def _tree(p, label, c, e, m, bits, drop, headroom, arith):
    """p, label: lists of width M x N products and labels; c: M x N running sum. Returns the new c, float32."""
    neg = torch.full(c.shape, -math.inf, dtype=torch.float64, device=c.device)
    top = torch.where(c != 0, _log2_floor(c) - 1, neg)
    for pk, lk in zip(p, label):
        top = torch.maximum(top, torch.where(pk != 0, lk, neg))
    anchor = torch.where(torch.isfinite(top), top, torch.zeros_like(top)) + headroom
    grid = torch.exp2(anchor + 2 - bits[0])
    x = [_keep(t.to(torch.float64) / grid, drop) for t in p + [c]]            # S2: exact, a power of two
    for level in range(1, len(bits)):                                          # S3
        pairs = [x[i] + x[i + 1] for i in range(0, len(x) - 1, 2)]
        x = pairs + x[-1:] if len(x) % 2 else pairs                            # an odd one passed up
        if bits[level] != bits[level - 1]:
            x = [_keep(t / 2.0 ** (bits[level - 1] - bits[level]), drop) for t in x]
    s = (x[0] * torch.exp2(anchor + 2 - bits[-1])).to(torch.float32)         # exact: |x| < 2^23
    return arith.acc_add(torch.zeros_like(c), s, e, m)                         # S4: the lane rounding


def _reduction(A, B, la, lb, per_tree, width, drop, headroom, arith):
    """One reduction: A n x M and B n x N codes (n = size, fewer in a K tail), la and lb their labels.
    size / width trees chained through c; products past n are zeros. Returns c, M x N float32."""
    n = A.shape[0]
    c = torch.zeros((A.shape[1], B.shape[1]), dtype=torch.float32, device=A.device)
    for t, (e, m, b) in enumerate(per_tree):
        ks = range(t * width, min((t + 1) * width, n))
        p = [arith.product(A[k].unsqueeze(1), B[k].unsqueeze(0)) for k in ks]
        label = [la[k].unsqueeze(1) + lb[k].unsqueeze(0) for k in ks]
        zero = torch.zeros_like(c)
        p += [zero] * (width - len(p))
        label += [zero.to(torch.float64)] * (width - len(label))
        c = _tree(p, label, c, e, m, b, drop, headroom, arith)
    return c


def _fused(arith, per_tree, width, drop, headroom):
    """`_reduction` compiled for one configuration by `arith.fuse`, built once and kept with the Arithmetic;
    None for an Arithmetic that does not fuse."""
    if getattr(arith, "fuse", None) is None:
        return None
    key = ("anchor_tree", tuple((e, m, tuple(b)) for e, m, b in per_tree), width, drop, headroom)
    return arith.fuse(key, lambda A, B, la, lb: _reduction(A, B, la, lb, per_tree, width, drop, headroom, arith))


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
    body = _fused(arith, per_tree, width, drop, head)
    C = torch.zeros((M, N), dtype=torch.float32, device=P_A.device)
    for g in range(0, K, block_size):
        g_end = min(g + block_size, K)
        scales = scale_map(X_A, X_B, g // block_size)
        for k0 in range(g, g_end, size):
            k1 = min(k0 + size, g_end)
            args = (P_A[k0:k1], P_B[k0:k1], la[k0:k1], lb[k0:k1])
            if body is not None and k1 - k0 == size:
                c = body(*args)
            else:                                                              # uncompiled, or a short K tail
                c = _reduction(*args, per_tree, width, drop, head, arith)
            C = arith.tile_add(C, c * scales)
    return C.to(torch.float32)
