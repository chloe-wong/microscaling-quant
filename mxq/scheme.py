"""mxq.scheme — a container for one explicit chain: how a layer's two operands become codes and how the codes
are multiplied. There are no presets; the caller names every piece.

    Scheme(name, act, weight, reduce)
        act(V)    -> (P, X)   how activations become codes and scales (V is K×M, blocks along K)
        weight(V) -> (P, X)   how weights become codes and scales     (V is K×N)
        reduce(P_A, X_A, P_B, X_B) -> Y   a reducer with its Arithmetic and schedule already bound
    Scheme.matmul(A, B) -> Y = Aᵀ·B  runs the three in order.

Example, the hardware chain:
    from functools import partial
    from mxq import Scheme, block, matmul, schedule
    q = partial(block.mxgemmini.quantize, fmt="MXFP8_E4M3", axis=0)
    hw = Scheme("mxgemmini", act=q, weight=q,
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
    act: Quantizer
    weight: Quantizer
    reduce: Reducer

    def matmul(self, A: Tensor, B: Tensor) -> Tensor:
        """Y = Aᵀ·B for A: K×M activations and B: K×N weights, both float."""
        P_A, X_A = self.act(A)
        P_B, X_B = self.weight(B)
        return self.reduce(P_A, X_A, P_B, X_B)
