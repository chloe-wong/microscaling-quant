"""mxq.block — block quantization: step 1 (scale_factor) and step 2 (element_quant) composed, three ways.

    P, X = block.<name>.quantize(V, fmt, axis=0, block_size=BLOCK)    P: V's shape, the codes; X: ceil(len/BLOCK) along axis, the scales
    V_hat = block.<name>.dequantize(P, X, axis=0)                      = P * expand(X)

    mxquant      scale_factor.mxquant + float_em grid=qtorch      MXQuant's simulation codes (all reported perplexities)
    mxgemmini    scale_factor.mxquant + float_em grid=ocp         MX-Gemmini's operand codes (== rtl_exact operands)
    ocp          scale_factor.ocp     + element_quant.microsoft   OCP MX v1.0 as Microsoft's microxcaling computes it

_driver.py holds the shared plumbing: split along an axis into blocks, pad, run the two steps, reassemble.
"""
from ._driver import BLOCK
from . import mxquant, mxgemmini, ocp

__all__ = ["BLOCK", "mxquant", "mxgemmini", "ocp"]
