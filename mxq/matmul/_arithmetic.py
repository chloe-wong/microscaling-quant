"""The three rounding points inside a reducer, as one object. THIS FILE DEFINES THE DATAPATHS.

    product(a, b)          multiply two codes, round to the product format
    acc_add(S, p, e, m)    add product p into running sum S held in an accumulator of format float(e, m)
    tile_add(C, tile)      add a finished, rescaled block sum into the output C

The reducer (matmul.systolic) decides the ORDER of these calls; an Arithmetic decides the ROUNDING
at each. Two Arithmetics are defined here, each built only from named mxq calls (float_em, arith); no rounding
rule is written in this file. Stage by stage:

    stage                 MXQUANT(prod_e, prod_m)                          MXGEMMINI(prod_e=4, prod_m=3)
    product(a, b)         fp32 a*b, then float_em ties_away on the         arith.truncate_significand to prod_m fraction
                          qtorch grid to float(prod_e, prod_m)             bits (flushed below 2^-16), then arith.saturate_product
                                                                           (448 for e4m3)
    acc_add(S, p, e, m)   fp32 S+p, then float_em ties_away on the         S and p each rounded rne on the ieee grid to
                          qtorch grid to float(e, m)                       float(e, m); arith.exact_add: exact sum, one rne
                                                                           rounding to float(e, m)
    tile_add(C, tile)     fp32 C+tile, no rounding                         C and tile each rounded rne to bf16 (8, 7);
                                                                           arith.exact_add to bf16
    matches               MXQuant MXLinearSim._simulate_atw                npu-exploration rtl_exact Y_hw (65536/65536) and
                          (complete_integration_e2e), bit-identical         the gemmini golden fp8_matmul_model, bit-identical
    validated for         6 operand formats, 14,820 configs                MXFP8_E4M3 operands only

The MXQUANT lane add must be a plain fp32 add: MXQuant rounds the fp32 sum, not the exact sum. The two differ
on rare inputs, so MXQUANT does not use arith.exact_add.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import torch

from .. import arith, rounding
from ..element_quant import float_em

__all__ = ["Arithmetic", "MXQUANT", "MXGEMMINI", "compiled"]

Tensor = torch.Tensor


@dataclass(frozen=True)
class Arithmetic:
    name: str
    product: Callable[[Tensor, Tensor], Tensor]
    acc_add: Callable[[Tensor, Tensor, int, int], Tensor]
    tile_add: Callable[[Tensor, Tensor], Tensor]
    #: optional: given a schedule and a reduction size, return one function that does a whole reduction of
    #: products and accumulations. Same operations in the same order as calling product and acc_add step by
    #: step; it exists only so the three stages can be fused into one GPU kernel. A reducer uses it when it
    #: has a full reduction and falls back to the stages otherwise. `compiled` below sets it; a plain
    #: Arithmetic leaves it None and nothing changes.
    fused_reduction: Optional[Callable[[Sequence[Tuple[int, int]], int], Callable]] = None
    #: optional, the same for a whole block of K: given a schedule, a reduction size and the block size, one
    #: function block(A, B, scales, C) -> C that runs every reduction of the block and adds each finished sum,
    #: times the block's scale map, into C. The same product, acc_add, multiply and tile_add calls in the same
    #: order; one call where a reducer would make two per reduction. `compiled` sets it; see there for why.
    fused_block: Optional[Callable[[Sequence[Tuple[int, int]], int, int], Callable]] = None


def MXQUANT(prod_e: int, prod_m: int) -> Arithmetic:
    """MXQuant `_simulate_atw` (complete_integration_e2e/eval_complete.py): qtorch float_quantize "nearest" on the
    fp32 product; fp32 add then qtorch float_quantize "nearest" of the SUM to the lane; fp32 cross-block add."""
    q = lambda x, e, m: float_em.quantize(x, e, m, rounding_mode="ties_away", grid="qtorch")
    return Arithmetic(
        name=f"mxquant(prod=e{prod_e}m{prod_m})",
        product=lambda a, b: q(a * b, prod_e, prod_m),
        acc_add=lambda S, p, e, m: q(S + p, e, m),
        tile_add=lambda C, tile: C + tile,
    )


#: the bf16 lane and tile rounding, on the bit pattern; why not a cast pair: see rounding.bf16
_bf16_rne = rounding.bf16


def MXGEMMINI(prod_e: int = 4, prod_m: int = 3, prod_floor: Optional[int] = -16) -> Arithmetic:
    """MX-Gemmini PE column (npu-exploration/rtl_exact, gemmini golden fp8_matmul_model.py):
    product significand truncated to prod_m bits, flushed below 2^prod_floor, then PE saturation (`mx_product_quantize_trunc`);
    both addends rounded RNE to the lane's float(e, m), added exactly, rounded once (`fp_add_exact(fp_quantize_rne, fp_quantize_rne)`);
    cross-block: both rounded to bf16, added exactly, rounded to bf16 (`bf16_accum_add(C, q_bf16_rne(tile))`).
    The bf16 roundings are done on the bit pattern, `rounding.bf16`: 2.98x on an all-bf16 ladder, 1.23x on
    schedule.HW_FINAL against the float64 grid, bit-identical, and safe under torch.compile.
    Validated against hardware for MXFP8_E4M3 operands with the default prod (4, 3) and schedule.HW_FINAL only;
    other operand formats or prod widths run the same stages unchecked."""
    lane = lambda x, e, m: (_bf16_rne(x) if (e, m) == (8, 7)
                            else float_em.quantize(x, e, m, rounding_mode="rne", grid="ieee"))
    flush = (lambda x: x) if prod_floor is None else (lambda x: arith.flush_product(x, prod_floor))
    return Arithmetic(
        name=f"mxgemmini(prod=e{prod_e}m{prod_m})",
        product=lambda a, b: arith.saturate_product(flush(arith.truncate_significand(a * b, prod_m)), prod_e, prod_m),
        acc_add=lambda S, p, e, m: arith.exact_add(lane(S, e, m), lane(p, e, m), e, m),
        tile_add=lambda C, tile: arith.exact_add(lane(C, 8, 7), lane(tile, 8, 7), 8, 7),
    )


@torch.library.custom_op("mxq::scale", mutates_args=())
def _scale(S: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """S * scales as a kernel of its own, which torch.compile cannot fuse into what follows.

    Inside one compiled graph a multiply feeding an add is contracted into a fused multiply-add, which rounds
    once where eager rounds twice (measured on the L40S: compiled `C + S * x` gave 1.4e-14 where eager gives 0,
    and a bitcast or a float64 detour between them did not prevent it). MXQUANT's tile_add is exactly that add,
    so inside `compiled`'s block the rescaling is this opaque op: the same eager multiply the reducer ran
    between two compiled calls before, in the same place."""
    return S * scales


@_scale.register_fake
def _(S, scales):
    return torch.empty_like(S)


def compiled(arith: Arithmetic) -> Arithmetic:
    """The same Arithmetic with its three functions passed through torch.compile, which fuses each chain of
    elementwise kernels into a few. Same IEEE operations per element in the same order, so the results are
    bit-identical; this is checked, not assumed (tests compare it with the uncompiled one bit for bit, including
    subnormal, Inf, NaN and saturating inputs). Needs a GPU with Triton; the first call of each shape compiles
    (about a minute for a block on a busy host, cached on disk by inductor across processes)."""
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 512)   # one entry per shape, lane and size
    c = lambda f: torch.compile(f, dynamic=False)
    cache: dict = {}

    def reduction(sched, A, B, k0):
        """Products k0 .. k0 + len(sched) of A (K x M) and B (K x N), accumulated lane by lane: M x N float32."""
        S = torch.zeros((A.shape[1], B.shape[1]), dtype=torch.float32, device=A.device)
        for i, (e, m) in enumerate(sched):
            S = arith.acc_add(S, arith.product(A[k0 + i].unsqueeze(1), B[k0 + i].unsqueeze(0)), e, m)
        return S

    def fused_reduction(schedule, size: int):
        """One compiled function for a whole reduction, built once per (schedule, size) and reused.

        A step of the loop launches its own kernels, about nine of them, and a single layer at 2048 tokens
        runs tens of thousands of steps, so the loop is bound by launch overhead rather than by arithmetic.
        The size is a constant, so the compiler unrolls the whole reduction into one graph and fuses
        across it. Nothing about the order or the rounding changes."""
        key = (tuple(tuple(t) for t in schedule), size)
        if key not in cache:
            sched = list(key[0])
            cache[key] = torch.compile(lambda A, B: reduction(sched, A, B, 0), dynamic=False)
        return cache[key]

    def fused_block(schedule, size: int, block_size: int):
        """One compiled function for a whole block of K: each of its reductions, rescaled and added into C.

        A compiled function costs 0.3-0.5 ms of host time per call on this host (dynamo guards, the AOT wrapper,
        allocations), whatever its size, and a reduction made two such calls and an eager multiply. The PE
        column's contraction over 2048 keys is 128 reductions, so a P·V call spent 60 ms on the host for 33 ms
        of GPU work, and one of 8 query rows (decoding) the same 60 ms for 3 ms. One call per block puts the host
        under the GPU at the layer shapes (measured: 128 reductions of 16384 x 64, 62 -> 38 ms; of 8 x 64, 62 ->
        34 ms; bit-identical). The graph is the reductions' graphs back to back with `_scale` and tile_add between
        them, so every value is rounded where it was before; a bigger graph was not better (two blocks per call:
        the same speed, twice the compile time)."""
        key = (tuple(tuple(t) for t in schedule), size, block_size)
        if key not in cache:
            sched = list(key[0])

            def block(A, B, scales, C):
                """A: block_size x M codes, B: block_size x N, scales: M x N (the block's scale map), C: M x N -> C."""
                for k0 in range(0, block_size, size):
                    C = arith.tile_add(C, _scale(reduction(sched, A, B, k0), scales))
                return C

            cache[key] = torch.compile(block, dynamic=False)
        return cache[key]

    return Arithmetic(name=f"compiled({arith.name})", product=c(arith.product), acc_add=c(arith.acc_add),
                      tile_add=c(arith.tile_add), fused_reduction=fused_reduction, fused_block=fused_block)
