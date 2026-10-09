"""mxq.block.mxgemmini — the block quantization of MX-Gemmini's operands.

    P, X = quantize(V, "MXFP8_E4M3", axis=0)        V_hat = P * expand(X)
    V_hat = dequantize(P, X, axis=0)

= scale_factor.mxquant (X = 2^floor(log2 amax), block max in [1, 2), no emax offset) followed by
  element_quant.float_em(grid="ocp") (OCP element grid: true subnormals, saturate at max_norm).

Same scale rule as block.mxquant; only the element grid differs (qtorch 0.2.0's fake subnormals there).

Two knobs, because the hardware moved after the fixture this module is validated against was frozen:

    rounding_mode   "ties_away" (default): the rtl_exact fixture of 2026-09-06, about 3% of whose
                    elements are exact ties, so the default cannot change without a new fixture.
                    "rne": the RTL and spike since 2026-09-10, and npu-exploration's current reference.
    scale_floor     scale_factor.MXQUANT_FLOOR = 1e-38 (default): MXQuant's simulation.
                    scale_factor.HARDWARE_FLOOR = 2^-23: MXQuant's end_to_end_linear quantizer and the
                    MX-Gemmini requantizer (FLT_EPSILON). Only all-zero or tiny blocks see the difference.

Validated against npu-exploration/rtl_exact with the defaults: P and X equal the operands its hardware
output was computed from. With rounding_mode="rne", scale_floor=HARDWARE_FLOOR: P and X equal MXQuant
end_to_end_linear `quantize_mx_block32(round_mode="even")` bit for bit on every format
(npu-exploration tests/selftest_block.py).
"""
from typing import Optional, Tuple, Union

import torch

from .. import scale_factor
from . import _driver
from ..element_quant import float_em
from ..element_quant.formats import Format, get

__all__ = ["quantize", "dequantize"]


def quantize(V: torch.Tensor, fmt: Union[str, Format], axis: int = 0, block_size: int = _driver.BLOCK,
             rounding_mode: str = "ties_away",
             scale_floor: float = scale_factor.MXQUANT_FLOOR,
             via: Optional[Tuple[int, int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """MX-Gemmini block quantize V along `axis`. Returns (P codes, X power-of-two scales), float32.

    `via=(e, m)`: the scaled value is first rounded to float(e, m) on the IEEE grid (subnormals, no saturation)
    with the same rounding mode, then to the format. MX-Gemmini's fp4 REQUANTIZER rounds the bf16 accumulator
    bf16 -> E3M1 -> E2M1 (BF16ScaleRoundToTiny), so `fmt="MXFP4", rounding_mode="rne", via=(3, 1)` is that
    requantizer; the single rounding differs from it on about one value in seven. Measured against
    npu-exploration rtl_exact/mxmesh/fp4.py::matrix_mx_requantize: identical on every finite block whose max
    is at least 2^-126 (E8M0's smallest scale); below that the hardware's scale underflows to zero and the
    block reads as zero, which this does not reproduce."""
    f = get(fmt)

    def elem(z: torch.Tensor) -> torch.Tensor:
        if via is not None:
            z = float_em.quantize(z, via[0], via[1], rounding_mode=rounding_mode, grid="ieee")
        return float_em.quantize(z, f.e, f.m, rounding_mode=rounding_mode, grid="ocp")

    return _driver.compose(V, axis=axis, block_size=block_size,
                           scale=None if f is None else lambda amax: scale_factor.mxquant(amax, scale_floor),   # step 1
                           elem=None if f is None else elem)                                                    # step 2


def dequantize(P: torch.Tensor, X: torch.Tensor, axis: int = 0, block_size: int = _driver.BLOCK) -> torch.Tensor:
    """V_hat = P * expand(X): each scale repeated block_size times along `axis`, cut to P's length."""
    return _driver.dequantize(P, X, axis, block_size)
