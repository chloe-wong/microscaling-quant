"""The three rounding points inside a reducer, as one object. THIS FILE DEFINES THE DATAPATHS.

    product(a, b)          multiply two codes, round to the product format
    acc_add(S, p, e, m)    add product p into running sum S held in an accumulator of format float(e, m)
    tile_add(C, tile)      add a finished, rescaled block sum into the output C

The reducer (matmul.systolic) decides the ORDER of these calls; an Arithmetic decides the ROUNDING
at each. Two Arithmetics are defined here, each built only from named mxq calls (float_em, arith); no rounding
rule is written in this file. Stage by stage:

    stage                 MXQUANT(prod_e, prod_m)                          MXGEMMINI(prod_e=4, prod_m=3)
    product(a, b)         fp32 a*b, then float_em ties_away on the         arith.truncate_significand to prod_m fraction
                          qtorch grid to float(prod_e, prod_m)             bits (no exponent clamp), then arith.saturate_product
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
from typing import Callable

import torch

from .. import arith
from ..element_quant import float_em

__all__ = ["Arithmetic", "MXQUANT", "MXGEMMINI", "compiled"]

Tensor = torch.Tensor


@dataclass(frozen=True)
class Arithmetic:
    name: str
    product: Callable[[Tensor, Tensor], Tensor]
    acc_add: Callable[[Tensor, Tensor, int, int], Tensor]
    tile_add: Callable[[Tensor, Tensor], Tensor]


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


def MXGEMMINI(prod_e: int = 4, prod_m: int = 3) -> Arithmetic:
    """MX-Gemmini PE column (npu-exploration/rtl_exact, gemmini golden fp8_matmul_model.py):
    product significand truncated to prod_m bits, no exponent clamp, then PE saturation (`mx_product_quantize_trunc`);
    both addends rounded RNE to the lane's float(e, m), added exactly, rounded once (`fp_add_exact(fp_quantize_rne, fp_quantize_rne)`);
    cross-block: both rounded to bf16, added exactly, rounded to bf16 (`bf16_accum_add(C, q_bf16_rne(tile))`).
    Validated against hardware for MXFP8_E4M3 operands with the default prod (4, 3) and schedule.HW_FINAL only;
    other operand formats or prod widths run the same stages unchecked."""
    lane = lambda x, e, m: float_em.quantize(x, e, m, rounding_mode="rne", grid="ieee")
    return Arithmetic(
        name=f"mxgemmini(prod=e{prod_e}m{prod_m})",
        product=lambda a, b: arith.saturate_product(arith.truncate_significand(a * b, prod_m), prod_e, prod_m),
        acc_add=lambda S, p, e, m: arith.exact_add(lane(S, e, m), lane(p, e, m), e, m),
        tile_add=lambda C, tile: arith.exact_add(lane(C, 8, 7), lane(tile, 8, 7), 8, 7),
    )


def compiled(arith: Arithmetic) -> Arithmetic:
    """The same Arithmetic with its three functions passed through torch.compile, which fuses each chain of
    elementwise kernels into a few. Same IEEE operations per element in the same order, so the results are
    bit-identical; this is checked, not assumed (tests compare it with the uncompiled one bit for bit, including
    subnormal, Inf, NaN and saturating inputs). Needs a GPU with Triton; the first call of each shape compiles."""
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 256)   # one entry per shape and lane
    c = lambda f: torch.compile(f, dynamic=False)
    return Arithmetic(name=f"compiled({arith.name})", product=c(arith.product), acc_add=c(arith.acc_add), tile_add=c(arith.tile_add))
