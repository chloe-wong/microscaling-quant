"""fp64_accum: the same codes and scales a reducer takes, with no rounding inside the multiply.

This is not an architecture, so it lives outside mxq.matmul, which holds only array dataflows. It is the
measuring stick that separates the two sources of error in a quantized matmul:

    what                                        how
    quantized operands, the array's arithmetic  matmul.systolic
    quantized operands, exact arithmetic        fp64_accum (this function)
    the original operands, exact arithmetic     A.double().t() @ B.double(), one line, no function needed

A reducer's answer minus fp64_accum's is the error the array's rounding adds. fp64_accum's answer minus the
original-operand product is the error the operand format adds.

Running a reducer with an Arithmetic that rounds nowhere gives the same numbers, but it loops over the
contraction in Python. This is one call into BLAS, which is what makes it usable over a whole model.
"""
import torch

from .block import BLOCK, dequantize

__all__ = ["fp64_accum"]


def fp64_accum(P_A: torch.Tensor, X_A: torch.Tensor, P_B: torch.Tensor, X_B: torch.Tensor,
               block_size: int = BLOCK) -> torch.Tensor:
    """Y = Aᵀ·B in float64, float32 out. Same operand layout as every reducer: A is K×M, B is K×N."""
    A = dequantize(P_A, X_A, axis=0, block_size=block_size).to(torch.float64)
    B = dequantize(P_B, X_B, axis=0, block_size=block_size).to(torch.float64)
    return (A.t() @ B).to(torch.float32)
