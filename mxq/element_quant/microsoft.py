"""Element quantization with Microsoft's microxcaling `_quantize_elemwise` (verbatim code in
mxq/ocp/elemwise.py). This is the element step of the OCP reference block algorithm, with the flags
`_quantize_mx` passes: saturate out-of-range normals to max_norm, keep subnormals.
`rounding_mode` uses microxcaling's names: "nearest" (ties away), "even" (RNE) or "floor".
Reference only; block_ocp is its one caller. FP32 (pass-through) has no OCP element format and is rejected.
"""
from typing import Union

import torch

from ..ocp.formats import ElemFormat
from ..ocp.elemwise import _quantize_elemwise
from .formats import Format, get

__all__ = ["quantize"]


def quantize(z: torch.Tensor, fmt: Union[str, Format], rounding_mode: str = "even") -> torch.Tensor:
    f = get(fmt)
    if f is None:
        raise ValueError(f"{fmt!r} is pass-through (no element format); nothing to quantize")
    return _quantize_elemwise(z, ElemFormat.from_str(f.ocp), round=rounding_mode,
                              saturate_normals=True, allow_denorm=True)
