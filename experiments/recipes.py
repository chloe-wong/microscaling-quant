"""The MX-Gemmini configuration the experiments run, and which layers it applies to.

One recipe: `hw_mxfp8_tapeout`. It is MX-Gemmini as the RTL computes it, and every value below is the
hardware's. More will be added as each is pinned down the same way.

    python experiments/llm_ppl.py --rules hw_mxfp8_tapeout --gpus 0,1,2,3
    python experiments/llm_ppl.py --rules hw_mxfp8_tapeout --dry-run    # which layer gets it, then stop
    python experiments/llm_ppl.py --rules none                          # the model as loaded, for comparison

The configuration, value by value, and where each value comes from:

    block scale       X = 2^floor(log2 max(amax, floor)); blocks of 32 along axis 0; no emax offset
                      Gemmini RTL and spike (log2_pmax = 0)
    scale floor       2^-23, i.e. FLT_EPSILON
                      Gemmini mx_fp_math.h: max(amax, FLT_EPSILON)
    element format    MXFP8_E4M3 on the OCP grid: true subnormals, saturate at max_norm 448
                      Gemmini
    element rounding  round-to-nearest-even
                      Gemmini mx_fp_math.h, since 2026-09-10
    product           significand truncated to 3 bits, no exponent clamp, then saturated at 448
                      Gemmini PE column (mx_product_quantize_trunc)
    accumulate        both addends rounded RNE to the lane's float(e, m), added exactly, rounded once
                      Gemmini PE column (fp_add_exact)
    cross-tile        both rounded to bf16 RNE, added exactly, rounded to bf16
                      Gemmini PE column (bf16_accum_add)
    accumulator ladder  16 lanes: 8x e4m4, 2x e4m5, 5x e4m6, 1x bf16
                      the tapeout's choice
    window            16 deep, inside blocks of 32
                      Gemmini
    layers            every nn.Linear except the attention projections; lm_head IS quantized
                      MXQuant, complete_integration_e2e/eval_complete.py

How far this is proven. The MX-Gemmini datapath here -- product, accumulate, cross-tile, ladder, window -- is
bit-identical to a real hardware capture: all 65536 elements of npu-exploration
`rtl_exact/fixture_llama_mlp.npz::Y_hw`. That capture was frozen 2026-09-06, four days before the RTL moved
its element rounding to RNE, so the capture itself was taken with ties-away operands. The difference has been
traced to that one convention and nothing else: forcing npu-exploration's golden back to ties-away reproduces
the same 65536 of 65536. So the datapath is proven against hardware; the operand rounding above follows the
current RTL but is not yet covered by a capture. A fresh capture would close that.

Two knobs on the operand quantizer are written out rather than left to their defaults, so a change to mxq's
defaults cannot silently move what this claims.
"""
from functools import partial

from torch import nn

from mxq import Scheme, block, matmul, scale_factor, schedule
from mxq.nn import is_attention

__all__ = ["OPERANDS", "ARITHMETIC", "LADDER", "HW_MXFP8_TAPEOUT", "mlp_and_head", "RULES"]

#: MX-Gemmini's operand codes: block scale, then the MXFP8_E4M3 element grid, rounded as the RTL rounds.
OPERANDS = partial(block.mxgemmini.quantize, fmt="MXFP8_E4M3", axis=0,
                   rounding_mode="rne", scale_floor=scale_factor.HARDWARE_FLOOR)

#: The three rounding points of MX-Gemmini's PE column. `compiled` fuses their GPU kernels; it is
#: bit-identical, and it is the reason a run takes minutes rather than hours.
ARITHMETIC = matmul.compiled(matmul.MXGEMMINI())

#: The accumulator ladder the tapeout chose: 8x e4m4, 2x e4m5, 5x e4m6, 1x bf16.
LADDER = schedule.HW_FINAL

HW_MXFP8_TAPEOUT = Scheme("hw_mxfp8_tapeout", a=OPERANDS, b=OPERANDS,
                          reduce=partial(matmul.systolic, arith=ARITHMETIC, schedule=LADDER))


def mlp_and_head(scheme):
    """Every nn.Linear except the attention projections, which are left alone whole (a module holding
    q_proj and k_proj is an attention module). lm_head is quantized. MXQuant's layer set."""
    return [(is_attention, None), (nn.Linear, scheme)]


#: A rule list says which nn.Linear gets which Scheme: (name with * wildcards | type | function, Scheme|None).
#: First match wins; None leaves the layer as it is.
RULES = {
    "none": None,                                       # no patch at all: the model as loaded, in bf16
    "hw_mxfp8_tapeout": mlp_and_head(HW_MXFP8_TAPEOUT),
}

try:                                                    # machine-local extras, not part of this repo
    from experiments._local import RULES as _extra
except ImportError:
    pass
else:
    RULES.update(_extra)
