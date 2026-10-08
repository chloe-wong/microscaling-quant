"""The vector ops around the matmuls, at a chosen precision.

    precision None      as transformers computes them: RMSNorm and softmax in fp32 inside, bf16 out
    precision "bf16"    every step's result rounded to bf16 (nearest-even) right after the step

Only the rounding changes. The steps, their order and the formulas are transformers' own (and attend's softmax).
Each step computes in fp32 from bf16-valued inputs and rounds once:

  - for +, -, ×, ÷ that is exactly what a correctly rounding bf16 unit returns: fp32 keeps 24 significant
    bits, at least 2·8 + 2, so rounding to fp32 first and then to bf16 equals rounding once;
  - exp and rsqrt are the fp32 library functions, rounded once. A chip's table versions differ by a bf16 step or
    two: on every bf16 input, the MX-Gemmini VPU (gemmini-rocc-tests include/vpu_ref.h) gives the correctly
    rounded exp 99.3% of the time, 1/x 58.6% and 1/sqrt(x) 70.4%, never more than 2 steps off;
  - a long sum (a softmax row's sum of exps, RMSNorm's mean of squares) adds in fp32 and rounds once, as a wide
    accumulator does. The VPU's RSUM does the same.

    op        where it runs                           steps rounded under "bf16"
    softmax   mxq.nn.attend                           S·scale, + mask, S − max, exp, Σ, ÷   (max is exact)
    rmsnorm   the forward of every *RMSNorm module    x², mean, + eps, rsqrt, x·r, × weight

The other vector steps of a bf16 model (RoPE's q·cos + rotate(q)·sin, SiLU, × up, the residual adds) are bf16
tensor ops already, each rounded once, so their bits are the same under either setting and nothing here touches
them. RoPE's cos/sin table and the loss stay in fp32.

patch(model, rules, vector=...) applies a setting; `resolve` turns its three spellings into one dict.
"""
from functools import partial
from typing import Callable, Optional

import torch

from .. import rounding
from .._fp64_accum import fp64_accum
from ..block import mxgemmini
from ..scheme import Scheme

__all__ = ["OPS", "PRECISIONS", "EXACT", "resolve", "rounder", "softmax", "rmsnorm"]

#: the vector ops a setting names
OPS = ("softmax", "rmsnorm")
#: None = as transformers computes it; "bf16" = every step rounded to bf16
PRECISIONS = (None, "bf16")

_PASSTHROUGH = partial(mxgemmini.quantize, fmt="FP32", axis=0)
#: Q·Kᵀ and P·V with no quantization and no rounding inside (float64 sums, fp32 out). patch gives it to an
#: attention module that no core rule chose when softmax is set, so its softmax runs here and not inside sdpa's
#: fused kernel, where no step can be rounded. Against sdpa it differs only in the order of fp32 sums.
EXACT = Scheme("exact", a=_PASSTHROUGH, b=_PASSTHROUGH, reduce=fp64_accum)


def resolve(vector) -> dict:
    """None | "bf16" (every op) | {op: None or "bf16"} (an op left out is None) -> {op: precision} for every op."""
    if vector is None or isinstance(vector, str):
        vector = {op: vector for op in OPS}
    if (not isinstance(vector, dict) or set(vector) - set(OPS)
            or any(not (v is None or (isinstance(v, str) and v in PRECISIONS)) for v in vector.values())):
        raise ValueError(f"vector must be None, 'bf16' or {{op: None or 'bf16'}} with op in {OPS}, got {vector!r}")
    return {op: vector.get(op) for op in OPS}


def rounder(precision: Optional[str]) -> Callable[[torch.Tensor], torch.Tensor]:
    """The rounding after each step, keeping the tensor's dtype: none (None) or bf16 nearest-even."""
    if precision is not None and not (isinstance(precision, str) and precision in PRECISIONS):
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    if precision is None:
        return lambda x: x
    return lambda x: rounding.bf16(x).to(x.dtype)


def softmax(S: torch.Tensor, precision: Optional[str]) -> torch.Tensor:
    """Softmax over the last dim of S, which is already scaled and masked.
    None: torch.softmax, as attend always ran it. "bf16": S − max, exp, Σ and ÷, each rounded."""
    if precision is None:
        return torch.softmax(S, dim=-1)
    r = rounder(precision)
    e = r(torch.exp(r(S - S.amax(dim=-1, keepdim=True))))
    return r(e / r(e.sum(dim=-1, keepdim=True)))


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float, precision: Optional[str]) -> torch.Tensor:
    """transformers' RMSNorm (Llama, Mistral, Qwen2, Phi-3 and others) with a rounding after each step.
    With None it is their forward, operation for operation; patch checks that bit for bit against each module
    before it replaces one, and refuses a module that computes something else (Gemma's (1 + weight), say)."""
    r = rounder(precision)
    dtype = x.dtype
    h = x.to(torch.float32)
    v = r(r(h.pow(2)).mean(-1, keepdim=True))
    h = r(h * r(torch.rsqrt(r(v + eps))))
    return r(weight * h.to(dtype))
