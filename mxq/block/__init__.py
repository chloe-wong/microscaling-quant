"""mxq.block — block quantization: step 1 (scale_factor) and step 2 (element_quant) composed.

    P, X = block.compose(V, scale=f, elem=g, axis=0, block_size=BLOCK)   any pair: X = f(block amax), P = g(block / X)
    P, X = block.<name>.quantize(V, fmt, axis=0, block_size=BLOCK)    P: V's shape, the codes; X: ceil(len/BLOCK) along axis, the scales
    V_hat = block.<name>.dequantize(P, X, axis=0)                      = P * expand(X)

The four named compositions (compose with a fixed pair, or on top of one):

    mxquant      scale_factor.mxquant + float_em grid=qtorch      MXQuant's simulation codes (all reported perplexities)
    mxgemmini    scale_factor.mxquant + float_em grid=ocp         MX-Gemmini's operand codes (== rtl_exact operands)
    ocp          scale_factor.ocp     + element_quant.microsoft   OCP MX v1.0 as Microsoft's microxcaling computes it
    lut          mxgemmini + mxq.lut (2-D, all settings required)  MX-Gemmini's LUT operand: codes replaced by table entries

`dequantize` is also exported from the package itself: putting the scales back is layout only, identical for
every composition, so each composition's dequantize is the same function under its own name.
_driver.py holds compose and its plumbing: split along an axis into blocks, pad, run the two steps, reassemble.
"""
from ._driver import BLOCK, compose, dequantize
from . import mxquant, mxgemmini, ocp, lut

__all__ = ["BLOCK", "compose", "dequantize", "mxquant", "mxgemmini", "ocp", "lut"]
