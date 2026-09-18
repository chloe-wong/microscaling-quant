"""mxq.matmul — reducers: Y = Aᵀ·B from codes and scales.

    Y = systolic(P_A, X_A, P_B, X_B, arith, schedule, window=16)     the PE column (MX-Gemmini, MXQuant's simulation)
    Y = ipt(P_A, X_A, P_B, X_B, arith, schedule, fanin=16)          the inner-product tree (dot-product unit)
    Y = fp64_accum(P_A, X_A, P_B, X_B)                              same inputs, no rounding anywhere

A: K×M, B: K×N, X: one row of scales per 32-block of K, Y: M×N (A = xᵀ, B = Wᵀ, as in MXQuant).
Codes, scales and Y are all float32: the narrow formats' values are exact in it, and every reducer returns it.
Every reducer takes the same operands and the same two objects: an Arithmetic (how a product is rounded, how a
sum is accumulated, how a finished block is added into the output) and a schedule (one float(e, m) per
accumulator position; the reducer says what a position is: a lane for systolic, a tree level for ipt).
MXQUANT(prod_e, prod_m) reproduces MXQuant's `_simulate_atw`; MXGEMMINI() the hardware.
"""
from .arithmetic import Arithmetic, MXQUANT, MXGEMMINI
from ._systolic import systolic, HW_FINAL
from ._ipt import ipt
from ._fp64_accum import fp64_accum

__all__ = ["Arithmetic", "MXQUANT", "MXGEMMINI", "systolic", "HW_FINAL", "ipt", "fp64_accum"]
