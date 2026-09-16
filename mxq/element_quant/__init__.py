"""mxq.element_quant — step 2 of block quantization: snap scaled values onto a format's codes.

    formats     the format table (name -> e, m, emax, max_norm)
    float_em    float(e, m), bit-identical to qtorch.float_quantize   (MXQuant's choice)
    microsoft   Microsoft microxcaling _quantize_elemwise             (OCP reference's choice)

float_em is also the quantizer for products and accumulators, which are float(e, m) too.
"""
from . import formats, float_em, microsoft
from .formats import FORMATS, Format, get

__all__ = ["formats", "float_em", "microsoft", "FORMATS", "Format", "get"]
