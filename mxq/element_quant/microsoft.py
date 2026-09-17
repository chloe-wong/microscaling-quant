"""Element quantization with Microsoft's microxcaling `_quantize_elemwise` (verbatim code in
mxq/ocp/elemwise.py). This is the element step of the OCP reference block algorithm, with the flags
`_quantize_mx` passes: saturate out-of-range normals to max_norm, keep subnormals.
`round` is "nearest" (ties away), "even" (RNE) or "floor". Reference only; block_ocp is its one caller.
"""
from typing import Union

import torch

from ..ocp.formats import ElemFormat
from ..ocp.elemwise import _quantize_elemwise
from .formats import Format, get

__all__ = ["quantize"]


def quantize(z: torch.Tensor, fmt: Union[str, Format], round: str = "even") -> torch.Tensor:
    return _quantize_elemwise(z, ElemFormat.from_str(get(fmt).ocp), round=round,
                              saturate_normals=True, allow_denorm=True)
