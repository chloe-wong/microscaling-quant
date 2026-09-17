"""mxq.scheme — one name for how a layer is quantized and multiplied.

    Scheme(name, act, weight, reduce)
        act(V)    -> (P, X)   how activations become codes and scales (V is K×M, blocks along K)
        weight(V) -> (P, X)   how weights become codes and scales     (V is K×N)
        reduce(P_A, X_A, P_B, X_B) -> Y   which reducer, with which Arithmetic and schedule, already bound
    Scheme.matmul(A, B) -> Y = Aᵀ·B  runs the three in order.

Factories, each a named composition of existing mxq functions and nothing else:
    MXQUANT(fmt, prod, schedule, window)   MXQuant's simulation: block_mxquant + systolic(MXQUANT)      all reported perplexities
    MXGEMMINI(fmt)                         the hardware:         block_mxgemmini + systolic(MXGEMMINI, HW_FINAL)
    OCP_FP64(fmt)                          OCP operands, no arithmetic rounding: block_ocp + fp64_accum
    PASSTHROUGH(prod, schedule, window)    FP32 operands (MXQuant --no-input-mx), MXQuant's product and lane rounding
An experiment names a scheme; it never calls a quantizer or a reducer itself.
"""
from dataclasses import dataclass
from functools import partial
from typing import Callable, Sequence, Tuple

import torch

from . import block_mxquant, block_mxgemmini, block_ocp
from . import matmul
from .matmul import systolic, fp64_accum, HW_FINAL

__all__ = ["Scheme", "MXQUANT", "MXGEMMINI", "OCP_FP64", "PASSTHROUGH"]

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


def _sched_name(schedule: Sequence[Tuple[int, int]]) -> str:
    if list(schedule) == HW_FINAL:
        return "hw_final"
    if len(set(schedule)) == 1:
        e, m = schedule[0]
        return f"e{e}m{m}x{len(schedule)}"
    return f"{len(schedule)}lanes"


def MXQUANT(fmt: str = "MXFP8_E4M3", prod: Tuple[int, int] = (4, 3),
            schedule: Sequence[Tuple[int, int]] = HW_FINAL, window: int = 16) -> Scheme:
    q = partial(block_mxquant.quantize, fmt=fmt, axis=0)
    return Scheme(name=f"mxquant/{fmt}/prod_e{prod[0]}m{prod[1]}/{_sched_name(schedule)}", act=q, weight=q,
                  reduce=partial(systolic, arith=matmul.MXQUANT(*prod), schedule=list(schedule), window=window))


def MXGEMMINI(fmt: str = "MXFP8_E4M3") -> Scheme:
    q = partial(block_mxgemmini.quantize, fmt=fmt, axis=0)
    return Scheme(name=f"mxgemmini/{fmt}", act=q, weight=q,
                  reduce=partial(systolic, arith=matmul.MXGEMMINI(), schedule=HW_FINAL, window=16))


def OCP_FP64(fmt: str = "MXFP8_E4M3") -> Scheme:
    q = partial(block_ocp.quantize, fmt=fmt, axis=0)
    return Scheme(name=f"ocp_fp64/{fmt}", act=q, weight=q, reduce=fp64_accum)


def PASSTHROUGH(prod: Tuple[int, int] = (4, 3), schedule: Sequence[Tuple[int, int]] = HW_FINAL, window: int = 16) -> Scheme:
    q = partial(block_mxquant.quantize, fmt="FP32", axis=0)
    return Scheme(name=f"passthrough/prod_e{prod[0]}m{prod[1]}/{_sched_name(schedule)}", act=q, weight=q,
                  reduce=partial(systolic, arith=matmul.MXQUANT(*prod), schedule=list(schedule), window=window))
