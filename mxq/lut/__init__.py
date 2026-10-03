"""mxq.lut — MX-Gemmini's look-up tables (LUTs), bit for bit.

A LUT format does not put element codes on the wire. Each element travels as a 4-bit index into a 16-entry table
of element values, one table per 2^G columns of a K×n operand (the chip's G, `lutUpdateRegularity`: rows of A,
columns of B, rows of a requantized C), each table spanning all of K. The operand is MX block-quantized first
(block.mxgemmini), so every value is already a code of its format; the table is a second, coarser step.

    T = tables(P, fmt, group=G, max_iters=50)     P: K×n block codes; T: (n >> G) × 16, rows ascending
    I = pick(P, T, group=G)                        host pick (operands): nearest entry, ties to the lower index
    I = finder(codes, T, fmt, group=G)             the chip's finder (requantized outputs), from element codes
    P_lut = lookup(I, T, group=G)
    P, X = block.lut.quantize(V, fmt, axis=0, ...) all of the above behind the block quantizer contract

    formats   decode / encode element codes, values(fmt) = what a table may hold, finder
    kmeans    tables, pick, lookup

Formats: FORMATS (MXFP6_E3M2, MXFP6_E2M3, MXFP8_E5M2, MXFP8_E4M3 for the quad PE). The rule is npu-exploration's
compiler/codebook.py, whose kernels are bit-exact on spike; the layout (mxq/lut, mxq/block/lut.py) is from
the luts branch (PR #1).
"""
from .formats import FORMATS, SIZE, decode, encode, finder, values
from .kmeans import lookup, pick, tables

__all__ = ["FORMATS", "SIZE", "decode", "encode", "finder", "values", "tables", "pick", "lookup"]
