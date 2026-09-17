"""The three rounding points inside a reducer, as one object.

    product(a, b)          multiply two codes, round to the product format
    acc_add(S, p, e, m)    add product p into running sum S held in an accumulator of format float(e, m)
    tile_add(C, tile)      add a finished, rescaled 32-block sum into the output C

Each preset is built only from named mxq calls (float_em, rounding, arith); no rounding is written here.
"""
from dataclasses import dataclass
from typing import Callable

import torch

from .. import arith, rounding
from ..element_quant import float_em

__all__ = ["Arithmetic", "MXQUANT", "MXGEMMINI"]

Tensor = torch.Tensor


@dataclass(frozen=True)
class Arithmetic:
    name: str
    product: Callable[[Tensor, Tensor], Tensor]
    acc_add: Callable[[Tensor, Tensor, int, int], Tensor]
    tile_add: Callable[[Tensor, Tensor], Tensor]


def MXQUANT(prod_e: int, prod_m: int) -> Arithmetic:
    """MXQuant `_simulate_atw` (complete_integration_e2e/eval_complete.py): qtorch float_quantize "nearest" on the
    fp32 product; fp32 add then qtorch float_quantize "nearest" of the SUM to the lane; fp32 cross-block add.
    The lane add must be a plain fp32 add: MXQuant rounds the fp32 sum, not the exact sum."""
    q = lambda x, e, m: float_em.quantize(x, e, m, round="ties_away", grid="qtorch")
    return Arithmetic(
        name=f"mxquant(prod=e{prod_e}m{prod_m})",
        product=lambda a, b: q(a * b, prod_e, prod_m),
        acc_add=lambda S, p, e, m: q(S + p, e, m),
        tile_add=lambda C, tile: C + tile,
    )


def MXGEMMINI(prod_e: int = 4, prod_m: int = 3) -> Arithmetic:
    """MX-Gemmini PE column (npu-exploration/rtl_exact, gemmini golden fp8_matmul_model.py):
    product mantissa truncated to prod_m bits with no exponent clamp (float32 subnormals kept), then PE saturation (`mx_product_quantize_trunc`);
    both addends rounded RNE to the lane's float(e, m), then added exactly and rounded once (`fp_add_exact(fp_quantize_rne, fp_quantize_rne)`);
    cross-block: tile rounded to bf16, accumulated in bf16 (`bf16_accum_add(C, q_bf16_rne(tile))`)."""
    lane = lambda x, e, m: float_em.quantize(x, e, m, round="rne", grid="ieee")

    def product(a, b):
        x = (a * b).to(torch.float32)
        ax = x.abs()
        nz = torch.isfinite(x) & (ax != 0)
        mant, ex = torch.frexp(torch.where(nz, ax, torch.ones_like(ax)))         # ax = mant * 2^ex, mant in [0.5, 1)
        k = rounding.round_int(mant * (1 << (prod_m + 1)), "truncate")            # significand with prod_m fraction bits, exact
        t = torch.ldexp(k / (1 << prod_m), ex - 1)                                # back to the value; no exponent clamp
        t = torch.where(nz, torch.copysign(t, x), x)                              # zeros, Inf, NaN pass through
        return arith.saturate_product(t, prod_e, prod_m)

    return Arithmetic(
        name=f"mxgemmini(prod=e{prod_e}m{prod_m})",
        product=product,
        acc_add=lambda S, p, e, m: arith.exact_add(lane(S, e, m), lane(p, e, m), e, m),
        tile_add=lambda C, tile: arith.exact_add(lane(C, 8, 7), lane(tile, 8, 7), 8, 7),
    )
