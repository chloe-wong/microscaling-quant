"""The three rounding points inside a reducer, as one object. THIS FILE DEFINES THE DATAPATHS.

    product(a, b)          multiply two codes, round to the product format
    acc_add(S, p, e, m)    add product p into running sum S held in an accumulator of format float(e, m)
    tile_add(C, tile)      add a finished, rescaled block sum into the output C

The reducer (matmul.systolic) decides the ORDER of these calls; an Arithmetic decides the ROUNDING
at each. Two Arithmetics are defined here, each built only from named mxq calls (float_em, arith); no rounding
rule is written in this file. Stage by stage:

    stage                 MXQUANT(prod_e, prod_m)                          MXGEMMINI(prod_e=4, prod_m=3)
    product(a, b)         fp32 a*b, then float_em ties_away on the         arith.truncate_significand to prod_m fraction
                          qtorch grid to float(prod_e, prod_m)             bits (flushed below 2^-16), then arith.saturate_product
                                                                           (448 for e4m3)
    acc_add(S, p, e, m)   fp32 S+p, then float_em ties_away on the         S and p each rounded rne on the ieee grid to
                          qtorch grid to float(e, m)                       float(e, m); arith.exact_add: exact sum, one rne
                                                                           rounding to float(e, m)
    tile_add(C, tile)     fp32 C+tile, no rounding                         C and tile each rounded rne to bf16 (8, 7);
                                                                           arith.exact_add to bf16
    matches               MXQuant MXLinearSim._simulate_atw                npu-exploration rtl_exact Y_hw (65536/65536) and
                          (complete_integration_e2e), bit-identical         the gemmini golden fp8_matmul_model, bit-identical
    validated for         6 operand formats, 14,820 configs                MXFP8_E4M3 operands only

The MXQUANT lane add must be a plain fp32 add: MXQuant rounds the fp32 sum, not the exact sum. The two differ
on rare inputs, so MXQUANT does not use arith.exact_add.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import torch

from .. import arith, rounding
from ..element_quant import float_em

__all__ = ["Arithmetic", "MXQUANT", "MXGEMMINI", "compiled"]

Tensor = torch.Tensor


@dataclass(frozen=True)
class Arithmetic:
    name: str
    product: Callable[[Tensor, Tensor], Tensor]
    acc_add: Callable[[Tensor, Tensor, int, int], Tensor]
    tile_add: Callable[[Tensor, Tensor], Tensor]
    #: optional: given a schedule and a window depth, return one function that does a whole window of
    #: products and accumulations. Same operations in the same order as calling product and acc_add step by
    #: step; it exists only so the three stages can be fused into one GPU kernel. A reducer uses it when it
    #: has a full window and falls back to the stages otherwise. `compiled` below sets it; a plain
    #: Arithmetic leaves it None and nothing changes.
    fused_window: Optional[Callable[[Sequence[Tuple[int, int]], int], Callable]] = None


def MXQUANT(prod_e: int, prod_m: int) -> Arithmetic:
    """MXQuant `_simulate_atw` (complete_integration_e2e/eval_complete.py): qtorch float_quantize "nearest" on the
    fp32 product; fp32 add then qtorch float_quantize "nearest" of the SUM to the lane; fp32 cross-block add."""
    q = lambda x, e, m: float_em.quantize(x, e, m, rounding_mode="ties_away", grid="qtorch")
    return Arithmetic(
        name=f"mxquant(prod=e{prod_e}m{prod_m})",
        product=lambda a, b: q(a * b, prod_e, prod_m),
        acc_add=lambda S, p, e, m: q(S + p, e, m),
        tile_add=lambda C, tile: C + tile,
    )


_U32 = 0xFFFFFFFF


def _bf16_rne(x: torch.Tensor) -> torch.Tensor:
    """Round float32 to the bf16 grid, nearest-even, on the bit pattern: (bits + 0x7FFF + lsb) & ~0xFFFF.

    This is the one lane rounding not done by float_em's scaled-integer grid. That grid runs in float64 for
    e = 8 because bf16's subnormal step, 2^-133, is itself a float32 subnormal that fused kernels flush; the
    bit form needs no float arithmetic at all, so it is exact on subnormals (bf16 shares float32's emin, so
    the uniform bit-add IS the grid), rounds the largest finite values into 0x7F800000 = Inf as IEEE does,
    keeps the sign of +-0, and passes NaN through unchanged. It is the same primitive the qtorch grid uses.

    NOT `x.to(bfloat16).to(float32)`: exact in eager, but torch.compile folds that cast pair away inside a
    fused kernel and the value comes back unrounded (2499980 of 3500168 corpus values). On schedule.HW_FINAL
    nothing that reaches a bf16 rounding is ever off the bf16 grid, so every gate passed with the cast; a
    ladder whose window ends in float32 showed it. `torch._inductor.config.emulate_precision_casts` does not
    prevent the fold. The gate's tier 1 now runs every stage compiled as well as eager for this reason.
    """
    x32 = x.to(torch.float32).contiguous()
    bits = x32.view(torch.int32).to(torch.int64) & _U32
    q = rounding.round_bits(bits, 7, "rne")
    q = torch.where(q >= 1 << 31, q - (1 << 32), q).to(torch.int32).view(torch.float32)
    return torch.where(torch.isnan(x32), x32, q)


def MXGEMMINI(prod_e: int = 4, prod_m: int = 3, prod_floor: Optional[int] = -16) -> Arithmetic:
    """MX-Gemmini PE column (npu-exploration/rtl_exact, gemmini golden fp8_matmul_model.py):
    product significand truncated to prod_m bits, flushed below 2^prod_floor, then PE saturation (`mx_product_quantize_trunc`);
    both addends rounded RNE to the lane's float(e, m), added exactly, rounded once (`fp_add_exact(fp_quantize_rne, fp_quantize_rne)`);
    cross-block: both rounded to bf16, added exactly, rounded to bf16 (`bf16_accum_add(C, q_bf16_rne(tile))`).
    The bf16 roundings are done on the bit pattern, `_bf16_rne` above: 2.98x on an all-bf16 ladder, 1.23x on
    schedule.HW_FINAL against the float64 grid, bit-identical, and safe under torch.compile.
    Validated against hardware for MXFP8_E4M3 operands with the default prod (4, 3) and schedule.HW_FINAL only;
    other operand formats or prod widths run the same stages unchecked."""
    lane = lambda x, e, m: (_bf16_rne(x) if (e, m) == (8, 7)
                            else float_em.quantize(x, e, m, rounding_mode="rne", grid="ieee"))
    flush = (lambda x: x) if prod_floor is None else (lambda x: arith.flush_product(x, prod_floor))
    return Arithmetic(
        name=f"mxgemmini(prod=e{prod_e}m{prod_m})",
        product=lambda a, b: arith.saturate_product(flush(arith.truncate_significand(a * b, prod_m)), prod_e, prod_m),
        acc_add=lambda S, p, e, m: arith.exact_add(lane(S, e, m), lane(p, e, m), e, m),
        tile_add=lambda C, tile: arith.exact_add(lane(C, 8, 7), lane(tile, 8, 7), 8, 7),
    )


def compiled(arith: Arithmetic) -> Arithmetic:
    """The same Arithmetic with its three functions passed through torch.compile, which fuses each chain of
    elementwise kernels into a few. Same IEEE operations per element in the same order, so the results are
    bit-identical; this is checked, not assumed (tests compare it with the uncompiled one bit for bit, including
    subnormal, Inf, NaN and saturating inputs). Needs a GPU with Triton; the first call of each shape compiles."""
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 512)   # one entry per shape, lane and window
    c = lambda f: torch.compile(f, dynamic=False)
    cache: dict = {}

    def fused_window(schedule, window: int):
        """One compiled function for a whole window, built once per (schedule, window) and reused.

        A step of the loop launches its own kernels, about nine of them, and a single layer at 2048 tokens
        runs tens of thousands of steps, so the loop is bound by launch overhead rather than by arithmetic.
        The window depth is a constant, so the compiler unrolls the whole window into one graph and fuses
        across it. Nothing about the order or the rounding changes."""
        key = (tuple(tuple(t) for t in schedule), window)
        if key not in cache:
            sched = list(key[0])

            def body(A, B):
                """A: window x M codes, B: window x N codes -> that window's sum, M x N float32."""
                S = torch.zeros((A.shape[1], B.shape[1]), dtype=torch.float32, device=A.device)
                for i in range(window):
                    e, m = sched[i]
                    S = arith.acc_add(S, arith.product(A[i].unsqueeze(1), B[i].unsqueeze(0)), e, m)
                return S

            cache[key] = torch.compile(body, dynamic=False)
        return cache[key]

    return Arithmetic(name=f"compiled({arith.name})", product=c(arith.product), acc_add=c(arith.acc_add),
                      tile_add=c(arith.tile_add), fused_window=fused_window)
