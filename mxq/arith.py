"""Arithmetic helpers that pair with float_em: exact add then round, and product saturation.
Bit-identical to the gemmini golden (`fp8_matmul_model.py`) functions named in each docstring."""
import torch

from .element_quant import float_em

__all__ = ["exact_add", "saturate_product"]


def exact_add(a: torch.Tensor, b: torch.Tensor, e: int, m: int, round: str = "rne", grid: str = "ieee") -> torch.Tensor:
    """a + b computed exactly, then quantized once to float(e, m). Golden `fp_add_exact` / `bf16_accum_add`.

    The sum is formed in float64. For operands already on a grid with <= 24-bit significands this is exact
    whenever their exponents differ by <= 29 bits; when they differ by more, the smaller operand is far below
    half an ulp of the result and cannot change the rounding, so the outcome is still the exact-add result.
    """
    a64, b64 = a.detach().to(torch.float64), b.detach().to(torch.float64)
    q = float_em.quantize(a64 + b64, e, m, round=round, grid=grid)
    # golden short-circuits a zero addend and returns the OTHER operand as is (so 0 + -0 -> -0, not +0)
    return torch.where(a64 == 0, b.to(torch.float32), torch.where(b64 == 0, a.to(torch.float32), q))


def saturate_product(x: torch.Tensor, e: int, m: int) -> torch.Tensor:
    """MxPEOutToRaw saturation after the product truncation. Golden `mx_product_saturate`.

    hardfloat treats biased exponent 2^e - 1 as special, so the PE saturates to mantissa 2^m - 2 at unbiased
    exponent bias + 1. For e4m3 that is 448, the format max. Values below max_normal pass through unchanged.
    """
    x = x.to(torch.float32)
    bias = float_em.bias(e)
    is_mx_fp8 = (e == 4 and m == 3)
    emax = bias + 1 if is_mx_fp8 else bias
    scale = float(2 ** m)
    max_mant = (2 ** m - 2) if is_mx_fp8 else (2 ** m - 1)
    max_normal = (2.0 ** emax) * (1.0 + max_mant / scale)
    sat_val = (1.0 + (2 ** m - 2) / scale) * (2.0 ** (bias + 1))
    ax = x.abs()
    return torch.sign(x) * torch.where(ax > max_normal, torch.full_like(ax, sat_val), ax)
