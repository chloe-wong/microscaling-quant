"""mxq.block.lut — MX-Gemmini's LUT operand: block.mxgemmini, then one 16-entry table per 2^group columns.

    P, X = quantize(V, "MXFP6_E3M2", axis=0, block_size=32, rounding_mode="rne", scale_floor=2**-23,
                    group=1, max_iters=50)                                  V_hat = P * expand(X)
    V_hat = dequantize(P, X, axis=0)

Same contract as block.<name>: P has V's shape, X one scale per block along `axis`. V is 2-D; the tables
group along the other axis, 2^group rows/columns per table spanning all of `axis` (K), the last table taking
what is left over. For a Scheme operand (K×M or K×N, axis=0) that is 2^group rows of A or columns of B, as on
the chip (whose loader takes whole groups only; a partial one exists here for arbitrary token counts).

Every setting is required: the chip's are in the hardware and run recipes, and a default here would be a
second place for them. See mxq.lut for the rule.

The one exception is opt-in, and its defaults ARE the chip today: fit= and pick= select the proposed raw-value
finder (mxq.lut.raw), where tables are fitted on, and indices picked from, the scaled values V / X instead of their
codes. Both "codes" (default): unchanged, bit for bit. Both "raw": the proposed design. Measured on TinyLlama
(MXFP6_E3M2, G = 1, MLP and lm_head, bf16 matmuls), every table fitted on its own operand as here: perplexity
9.62 (codes/codes) -> 8.18 (raw/raw); bf16 7.20. Mixed settings are not design points: fit="raw" with
pick="codes" measured 11.28, fit="codes" with pick="raw" 9.23. pick="raw" on an operand the chip requantizes needs
the new finder in hardware (mxq.lut.finder_raw models it); on an operand the host sends it is software only.

    P, X = quantize(V, "MXFP6_E3M2", axis=0, block_size=32, rounding_mode="rne", scale_floor=2**-23,
                    group=1, max_iters=50, fit="raw", pick="raw")
"""
import torch

from . import _driver, mxgemmini
from .. import lut

__all__ = ["quantize", "dequantize"]


FITS = PICKS = ("codes", "raw")


def quantize(V: torch.Tensor, fmt, axis: int, block_size: int, *, rounding_mode: str, scale_floor: float,
             group: int, max_iters: int, fit: str = "codes", pick: str = "codes"):
    """block.mxgemmini.quantize, then each code replaced by its table entry. Returns (P, X), float32.
    fit / pick: "codes" (default, the chip) or "raw" (the proposed raw-value finder; see the module doc)."""
    if fit not in FITS or pick not in PICKS:
        raise ValueError(f"block.lut: fit {fit!r} and pick {pick!r} must each be one of {', '.join(FITS)}")
    if V.ndim != 2 or axis not in (0, 1, -1, -2):
        raise ValueError(f"block.lut: V must be 2-D with axis 0 or 1, got shape {tuple(V.shape)}, axis {axis}")
    lut.formats._format(fmt)                                                     # refuse a non-LUT format first
    P, X = mxgemmini.quantize(V, fmt, axis=axis, block_size=block_size, rounding_mode=rounding_mode,
                              scale_floor=scale_floor)
    Pk = P if axis % 2 == 0 else P.t()                                           # K×n
    if "raw" in (fit, pick):                                                     # R = V / X, exact (powers of two)
        Vk, Xk = (V, X) if axis % 2 == 0 else (V.t(), X.t())
        R = Vk.float() / Xk.repeat_interleave(block_size, dim=0)[:Vk.shape[0]]
    T = lut.tables_raw(R, fmt, group=group, max_iters=max_iters) if fit == "raw" else \
        lut.tables(Pk, fmt, group=group, max_iters=max_iters)
    I = lut.pick_raw(R, T, group=group) if pick == "raw" else lut.pick(Pk, T, group=group)
    P_lut = lut.lookup(I, T, group=group)
    return (P_lut if axis % 2 == 0 else P_lut.t()).contiguous(), X


def dequantize(P: torch.Tensor, X: torch.Tensor, axis: int = 0, block_size: int = _driver.BLOCK) -> torch.Tensor:
    """V_hat = P * expand(X): each scale repeated block_size times along `axis`, cut to P's length."""
    return _driver.dequantize(P, X, axis, block_size)
