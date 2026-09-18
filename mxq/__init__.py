"""mxq — granular microscaling (MX) quantization ops.

Block quantization is two steps: a per-block scale factor, then element quantization of the
scaled values. The steps are separate modules so they can be validated and mixed; three
compositions are provided. A matmul on the codes is a reducer with an Arithmetic and a schedule.

    scale_factor      step 1: mxquant(amax) | ocp(amax, emax)
    element_quant     step 2: float_em with grid qtorch (== qtorch 0.2.0) | ieee (lane accumulators) | ocp (OCP element formats)
                              microsoft: microxcaling verbatim, reference only
    block_ocp         OCP MX v1.0            = scale_factor.ocp     + element_quant.microsoft
    block_mxquant     MXQuant simulation     = scale_factor.mxquant + float_em grid=qtorch
    block_mxgemmini   MX-Gemmini operands    = scale_factor.mxquant + float_em grid=ocp
    ocp               Microsoft microxcaling, verbatim: the oracle block_ocp is validated against
    rounding          ties_away | rne | truncate, on float32 bit patterns or integers; shared by every quantizer
    arith             exact_add, truncate_significand, saturate_product: what a PE does between quantizations
    matmul            reducers Y = Aᵀ·B from codes and scales: systolic, ipt, fp64_accum;
                      Arithmetic = the three rounding points, with MXQUANT and MXGEMMINI as the two documented datapaths
    schedule          one float(e, m) per accumulator position: load(csv, rows), fixed(e, m, rows), HW_FINAL
    scheme            Scheme(name, act, weight, reduce): a container for one explicit chain; no presets

Every block quantizer has the same interface:  P, X = quantize(V, fmt, axis)   V_hat = P * expand(X)
"""
from . import ocp, rounding, arith, schedule, scale_factor, element_quant, block_ocp, block_mxquant, block_mxgemmini, matmul, scheme
from ._blocks import BLOCK
from .element_quant.formats import Format
from .scheme import Scheme

__all__ = ["ocp", "rounding", "arith", "schedule", "scale_factor", "element_quant", "block_ocp", "block_mxquant",
           "block_mxgemmini", "matmul", "scheme", "BLOCK", "Format", "Scheme"]
