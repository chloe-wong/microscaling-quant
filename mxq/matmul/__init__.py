"""mxq.matmul — reducers: Y = Aᵀ·B from codes and scales.

    Y = systolic(P_A, X_A, P_B, X_B, arith, schedule)     the 16-deep PE column (MX-Gemmini, MXQuant's simulation)
    Y = fp64_accum(P_A, X_A, P_B, X_B)                    same inputs, no rounding anywhere

A: K×M, B: K×N, X: one row of scales per 32-block of K, Y: M×N (A = xᵀ, B = Wᵀ, as in MXQuant).
`arith` is an Arithmetic: how a product is rounded, how a lane sum is rounded, how a finished block is
added into the output. MXQUANT(prod_e, prod_m) reproduces MXQuant's `_simulate_atw`; MXGEMMINI() the hardware.
"""
from .arithmetic import Arithmetic, MXQUANT, MXGEMMINI
from .systolic import systolic
from .fp64_accum import fp64_accum

__all__ = ["Arithmetic", "MXQUANT", "MXGEMMINI", "systolic", "fp64_accum"]
