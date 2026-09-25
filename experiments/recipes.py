"""The named Schemes the experiments use, and the rule lists that say which layer gets which.

THE CANONICAL FLOW IS `hw_mxfp8_tapeout`. It is MX-Gemmini as the RTL computes it today, and every choice in
it is the hardware's, not a simulation's. Anything else here exists to be compared against it and says so.

    Scheme      how one matmul is done: quantizer for A, quantizer for B, dataflow + Arithmetic + ladder
    rule list   which nn.Linear gets which Scheme: (layer name with * wildcards | layer type | function,
                Scheme | None); first matching rule wins, and None leaves the layer as it is

`hw_mxfp8_tapeout`, choice by choice, and where each choice comes from:

    block scale       X = 2^floor(log2 max(amax, floor)), block 32 along axis 0, no emax offset
                      -- Gemmini RTL and spike (log2_pmax = 0)
    scale floor       HARDWARE_FLOOR = 2^-23, i.e. FLT_EPSILON
                      -- Gemmini mx_fp_math.h: max(amax, FLT_EPSILON)
    element format    MXFP8_E4M3 on the OCP grid: true subnormals, saturate at max_norm 448
                      -- Gemmini
    element rounding  RNE -- Gemmini mx_fp_math.h, since 2026-09-10
    product           significand truncated to 3 bits, no exponent clamp, saturated at 448
                      -- Gemmini PE column (mx_product_quantize_trunc)
    accumulate        both addends rounded RNE to the lane's float(e, m), added exactly, rounded once
                      -- Gemmini PE column (fp_add_exact)
    cross-tile        both rounded to bf16 RNE, added exactly, rounded to bf16
                      -- Gemmini PE column (bf16_accum_add)
    ladder            16 lanes: 8x e4m4, 2x e4m5, 5x e4m6, 1x bf16 -- the tapeout's choice
    window            16 deep, inside blocks of 32 -- Gemmini
    layers            every nn.Linear except the attention projections; lm_head IS quantized
                      -- MXQuant's layer set (complete_integration_e2e/eval_complete.py)

What is proven and what is not. The only bit-exact hardware capture is
npu-exploration `rtl_exact/fixture_llama_mlp.npz::Y_hw`, frozen 2026-09-06. It predates the RTL's move to RNE
on 2026-09-10, so it anchors `hw_mxfp8_fixture`, not the canonical flow: that recipe reproduces all 65536
elements of it. `hw_mxfp8_tapeout` differs from it ONLY in operand rounding and the scale floor, and the
divergence has been traced to exactly that -- forcing npu-exploration's golden back to ties-away reproduces
the same 65536/65536. A fresh capture from the current Gemmini would anchor the canonical flow directly; until
then its hardware claim rests on that traced equivalence, not on a capture.

Use:  from experiments import recipes;  patch(model, recipes.RULES["hw_mxfp8_tapeout"])
"""
from functools import partial

from torch import nn

from mxq import Scheme, block, matmul, scale_factor, schedule
from mxq.nn import is_attention

__all__ = ["OPERANDS", "ARITHMETIC", "LADDER", "HW_MXFP8_TAPEOUT", "HW_MXFP8_FIXTURE",
           "mlp_and_head", "RULES"]

#: THE operand quantizer: MX-Gemmini's codes as the RTL computes them today. Both knobs are spelled out so a
#: change to mxq's defaults cannot silently move what this claims. Import this rather than rebuilding it --
#: a second copy is how two recipes quietly stop describing the same hardware.
OPERANDS = partial(block.mxgemmini.quantize, fmt="MXFP8_E4M3", axis=0,
                   rounding_mode="rne", scale_floor=scale_factor.HARDWARE_FLOOR)

#: THE arithmetic: the three rounding points of MX-Gemmini's PE column. `compiled` fuses its GPU kernels and
#: is bit-identical; it is the reason a run takes minutes rather than hours.
ARITHMETIC = matmul.compiled(matmul.MXGEMMINI())

#: THE accumulator ladder the tapeout chose: 8x e4m4, 2x e4m5, 5x e4m6, 1x bf16.
LADDER = schedule.HW_FINAL


def mlp_and_head(scheme):
    """MXQuant's layer set: every nn.Linear except the attention projections, which are left alone whole
    (a module holding q_proj and k_proj is an attention module). lm_head is quantized."""
    return [(is_attention, None), (nn.Linear, scheme)]


#: THE flow. MX-Gemmini as the RTL computes it today.
HW_MXFP8_TAPEOUT = Scheme("hw_mxfp8_tapeout", a=OPERANDS, b=OPERANDS,
                          reduce=partial(matmul.systolic, arith=ARITHMETIC, schedule=LADDER))

#: The same flow with the operand rounding and scale floor the RTL used BEFORE 2026-09-10. Not current
#: hardware. It exists because it is the only thing bit-identical to a real capture -- all 65536 elements of
#: rtl_exact's Y_hw -- so it is the regression anchor that proves the datapath itself is right.
_FIXTURE_OPERANDS = partial(block.mxgemmini.quantize, fmt="MXFP8_E4M3", axis=0,
                            rounding_mode="ties_away", scale_floor=scale_factor.MXQUANT_FLOOR)
HW_MXFP8_FIXTURE = Scheme("hw_mxfp8_fixture", a=_FIXTURE_OPERANDS, b=_FIXTURE_OPERANDS,
                          reduce=partial(matmul.systolic, arith=ARITHMETIC, schedule=LADDER))

RULES = {
    "none": None,                                       # no patch at all: the model as loaded, in bf16
    "hw_mxfp8_tapeout": mlp_and_head(HW_MXFP8_TAPEOUT),
    "hw_mxfp8_fixture": mlp_and_head(HW_MXFP8_FIXTURE),
}

try:                                                    # machine-local extras, not part of this repo
    from experiments._local import RULES as _extra
except ImportError:
    pass
else:
    RULES.update(_extra)
