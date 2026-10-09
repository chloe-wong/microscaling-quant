"""mxq.scheme — a container for one explicit chain: how a matmul's two operands become codes and how the codes
are multiplied. There are no presets; the caller names every piece.

    Scheme(name, a, b, reduce)
        a(V) -> (P, X)   how operand A becomes codes and scales (V is K×M, blocks along K)
        b(V) -> (P, X)   how operand B becomes codes and scales (V is K×N, blocks along K)
        reduce(P_A, X_A, P_B, X_B) -> Y   a dataflow with its Arithmetic and schedule already bound
        rows = 1          columns of A (token rows) that `a` must see in one call: a LUT operand
                          (block.lut, group G) shares a table across 2^G of them, so MXLinear splits its tokens
                          in multiples of `rows`
        block_size = 32   the block length along K that `a` and `b` use. Callers that slice codes and scales
                          (mxq.nn.attend) need it; `matmul` checks it against the scales a and b return
    Scheme.matmul(A, B) -> Y = Aᵀ·B  runs the three in order.

a and b are the same A and B the reducers use. For a Linear layer, A = xᵀ is the activation and B = Wᵀ is the
weight. For an attention product both are activations.

Example, the hardware chain:
    from functools import partial
    from mxq import Scheme, block, matmul, schedule
    q = partial(block.mxgemmini.quantize, fmt="MXFP8_E4M3", axis=0)
    hw = Scheme("hw_fp8", a=q, b=q,
                reduce=partial(matmul.systolic, arith=matmul.MXGEMMINI(), schedule=schedule.HW_FINAL))
"""
import math
from dataclasses import dataclass
from typing import Callable, Tuple

import torch

from .block import BLOCK

__all__ = ["Scheme"]

Tensor = torch.Tensor
Quantizer = Callable[[Tensor], Tuple[Tensor, Tensor]]
Reducer = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class Scheme:
    name: str
    a: Quantizer
    b: Quantizer
    reduce: Reducer
    rows: int = 1
    block_size: int = BLOCK

    def __post_init__(self):
        for field in ("rows", "block_size"):
            v = getattr(self, field)
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                raise ValueError(f"Scheme: {field} {v!r} must be a positive integer")

    def check_scales(self, X: Tensor, K: int, operand: str) -> None:
        """X must hold one row of scales per `block_size` of K, as the quantizer was meant to make them."""
        if X.shape[0] != math.ceil(K / self.block_size):
            raise ValueError(f"Scheme {self.name!r}: {operand} has {X.shape[0]} scale rows for K = {K}, "
                             f"block_size {self.block_size} needs {math.ceil(K / self.block_size)}; "
                             "the Scheme's block_size and its quantizer's disagree")

    def matmul(self, A: Tensor, B: Tensor) -> Tensor:
        """Y = Aᵀ·B for A: K×M and B: K×N, both float."""
        P_A, X_A = self.a(A)
        P_B, X_B = self.b(B)
        self.check_scales(X_A, A.shape[0], "a")
        self.check_scales(X_B, B.shape[0], "b")
        return self.reduce(P_A, X_A, P_B, X_B)
