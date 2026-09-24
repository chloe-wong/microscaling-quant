"""mxq — granular microscaling (MX) quantization ops.

Block quantization is two steps: a per-block scale factor, then element quantization of the
scaled values. The steps are separate modules so they can be validated and mixed; `block` holds
the three compositions. A matmul on the codes is a reducer with an Arithmetic and a schedule.

    scale_factor      step 1: mxquant(amax) | ocp(amax, emax)
    element_quant     step 2: float_em with grid qtorch (== qtorch 0.2.0) | ieee (lane accumulators) | ocp (OCP element formats)
                              microsoft: microxcaling verbatim, reference only
    block             step 1 + step 2, one interface P, X = quantize(V, fmt, axis):
                        block.mxquant     MXQuant simulation     = scale_factor.mxquant + float_em grid=qtorch
                        block.mxgemmini   MX-Gemmini operands    = scale_factor.mxquant + float_em grid=ocp
                        block.ocp         OCP MX v1.0            = scale_factor.ocp     + element_quant.microsoft
    microxcaling      Microsoft's microxcaling package, verbatim: the oracle block.ocp is validated against
    rounding          ties_away | rne | truncate, on float32 bit patterns or integers; shared by every quantizer
    arith             exact_add, truncate_significand, saturate_product: what a PE does between quantizations
    matmul            the array dataflows, Y = Aᵀ·B from codes and scales: systolic;
                      Arithmetic = the three rounding points, with MXQUANT and MXGEMMINI as the two documented datapaths
    fp64_accum        the same codes with no rounding inside the multiply: the error floor, not an architecture
    schedule          one float(e, m) per accumulator position: load(csv, rows), fixed(e, m, rows), HW_FINAL
    scheme            Scheme(name, a, b, reduce): a container for one explicit chain (one matmul); no presets
    nn                putting Schemes into a model: MXLinear (one nn.Linear through one Scheme), patch (rules per layer or type)
"""
__version__ = "0.1.0"

from . import microxcaling, rounding, arith, schedule, scale_factor, element_quant, block, matmul, scheme, nn
from .block import BLOCK
from ._fp64_accum import fp64_accum
from .element_quant.formats import Format
from .scheme import Scheme

__all__ = ["microxcaling", "rounding", "arith", "schedule", "scale_factor", "element_quant", "block", "matmul", "scheme", "nn",
           "BLOCK", "Format", "Scheme", "fp64_accum"]
