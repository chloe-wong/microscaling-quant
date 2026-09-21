"""mxq.matmul — the array dataflows: Y = Aᵀ·B from codes and scales, in the order the hardware sums them.

    Y = systolic(P_A, X_A, P_B, X_B, arith, schedule, window=16)     the PE column (MX-Gemmini, MXQuant's simulation)
    Y = ipt(P_A, X_A, P_B, X_B, arith, schedule, fanin=16)          the inner-product tree (dot-product unit)

A: K×M, B: K×N, X: one row of scales per 32-block of K, Y: M×N (A = xᵀ, B = Wᵀ, as in MXQuant).
Codes, scales and Y are all float32: the narrow formats' values are exact in it, and both dataflows return it.
Both take the same operands and the same two objects: an Arithmetic (how a product is rounded, how a sum is
accumulated, how a finished block is added into the output; the datapaths are defined stage by stage in
_arithmetic.py) and a schedule (one float(e, m) per accumulator position; the dataflow says what a position
is: a lane for systolic, a tree level for ipt).
MXQUANT(prod_e, prod_m) is MXQuant's `_simulate_atw` datapath; MXGEMMINI() is the hardware's. They are
Arithmetics, i.e. the `arith` argument, not matmuls themselves.

For the same codes with no rounding inside the multiply, see mxq.fp64_accum: it is a measuring stick, not an
architecture, so it is not here.
"""
from ._arithmetic import Arithmetic, MXQUANT, MXGEMMINI
from ._systolic import systolic
from ._ipt import ipt
from ..schedule import HW_FINAL

__all__ = ["Arithmetic", "MXQUANT", "MXGEMMINI", "systolic", "ipt", "HW_FINAL"]
