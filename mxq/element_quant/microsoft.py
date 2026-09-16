"""Element quantization with Microsoft's microxcaling `_quantize_elemwise` (verbatim code in
mxq/ocp/elemwise.py). This is the element step of the OCP reference block algorithm.

Defaults are the flags `_quantize_mx` passes: saturate out-of-range normals to max_norm,
keep subnormals. `round` is "nearest" (ties away), "even" (RNE) or "floor".
"""
from typing import Union

import torch

from ..ocp.formats import ElemFormat
from ..ocp.elemwise import _quantize_elemwise
from .formats import Format, get

__all__ = ["quantize", "quantizer"]


def quantize(z: torch.Tensor, fmt: Union[str, Format], round: str = "even",
             saturate_normals: bool = True, allow_denorm: bool = True) -> torch.Tensor:
    f = get(fmt)
    if f is None:
        return z
    return _quantize_elemwise(z, ElemFormat.from_str(f.ocp), round=round,
                              saturate_normals=saturate_normals, allow_denorm=allow_denorm)


def quantizer(fmt: Union[str, Format], round: str = "even", **kw):
    return lambda z: quantize(z, fmt, round, **kw)
