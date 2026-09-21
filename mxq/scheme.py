"""mxq.scheme — a container for one explicit chain: how a matmul's two operands become codes and how the codes
are multiplied. There are no presets; the caller names every piece.

    Scheme(name, a, b, reduce)
        a(V) -> (P, X)   how operand A becomes codes and scales (V is K×M, blocks along K)
        b(V) -> (P, X)   how operand B becomes codes and scales (V is K×N, blocks along K)
        reduce(P_A, X_A, P_B, X_B) -> Y   a dataflow with its Arithmetic and schedule already bound
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
from dataclasses import dataclass
from typing import Callable, Tuple

import torch

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

    def matmul(self, A: Tensor, B: Tensor) -> Tensor:
        """Y = Aᵀ·B for A: K×M and B: K×N, both float."""
        P_A, X_A = self.a(A)
        P_B, X_B = self.b(B)
        return self.reduce(P_A, X_A, P_B, X_B)
