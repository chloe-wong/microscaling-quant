"""mxq — granular microscaling (MX) quantization ops.

Block quantization is two steps: a per-block scale factor, then element quantization of the
scaled values. The steps are separate packages so they can be validated and mixed; two
compositions are provided.

    scale_factor      step 1: mxquant(amax) | ocp(amax, emax)
    element_quant     step 2: float_em (== qtorch float_quantize) | microsoft (microxcaling)
    block_ocp         OCP MX v1.0            = scale_factor.ocp     + element_quant.microsoft
    block_mxquant     MXQuant simulation     = scale_factor.mxquant + element_quant.float_em (qtorch grid)
    ocp               Microsoft microxcaling, verbatim: the oracle block_ocp is validated against
    rounding          ties_away | rne | truncate, on float32 bit patterns or integers; shared by every quantizer

Every block quantizer has the same interface:  P, X = quantize(V, fmt, axis)   V_hat = P * expand(X)
"""
from . import ocp, rounding, scale_factor, element_quant, block_ocp, block_mxquant

__all__ = ["ocp", "rounding", "scale_factor", "element_quant", "block_ocp", "block_mxquant"]
