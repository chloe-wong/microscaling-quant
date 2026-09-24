"""The named Schemes and rule lists the experiments use. Every chain is written out in full here, so a name in
a results table can always be traced back to the exact quantizers, arithmetic and ladder behind it.

    Scheme      how one matmul is done: quantizer for A, quantizer for B, dataflow + Arithmetic + ladder
    rule list   which nn.Linear gets which Scheme: (layer name with * wildcards | layer type | function,
                Scheme | None); first matching rule wins, and None leaves the layer as it is

Both Schemes are the MX-Gemmini datapath: its operand codes, its arithmetic, its 16-deep accumulator window.
They differ only in the accumulator ladder, which is what the tapeout had to choose:

    hw_fp8_baseline   every lane bf16                      what a ladder is measured against
    hw_fp8_tapeout    8x e4m4, 2x e4m5, 5x e4m6, 1x bf16   the tapeout ladder

What is gated, and what is not. `hw_fp8_tapeout` is bit-identical to the saved hardware output in
npu-exploration, all 65536 elements of `rtl_exact/fixture_llama_mlp.npz::Y_hw`, which spike reproduces element
for element. That fixture was frozen on 2026-09-06, so the gate covers `rounding_mode="ties_away"` with
`scale_floor=MXQUANT_FLOOR`. Both are written out below rather than left to defaults, so changing the defaults
cannot silently change what this claims. The RTL has rounded operands RNE since 2026-09-10; passing
`rounding_mode="rne"` and `scale_floor=scale_factor.HARDWARE_FLOOR` follows it, and is NOT covered by the
frozen fixture.

`hw_fp8_baseline` is the same datapath on a uniform ladder. No hardware output exists for it, so it is a
reference point for the ladder, not a claim about hardware.

Use:  from experiments import recipes;  patch(model, recipes.RULES["hw_fp8_tapeout"])
"""
from functools import partial

from torch import nn

from mxq import Scheme, block, matmul, scale_factor, schedule
from mxq.nn import is_attention

__all__ = ["HW_FP8_BASELINE", "HW_FP8_TAPEOUT", "RULES"]

#: MX-Gemmini's operand codes. Both knobs are spelled out: these are the ones the frozen fixture covers.
_hw_fp8 = partial(block.mxgemmini.quantize, fmt="MXFP8_E4M3", axis=0,
                  rounding_mode="ties_away", scale_floor=scale_factor.MXQUANT_FLOOR)

#: matmul.compiled fuses the arithmetic's GPU kernels. Bit-identical, and the reason a run takes minutes.
_hw_arith = matmul.compiled(matmul.MXGEMMINI())


def _hw(name, lanes):
    return Scheme(name, a=_hw_fp8, b=_hw_fp8, reduce=partial(matmul.systolic, arith=_hw_arith, schedule=lanes))


#: The MX-Gemmini datapath with every accumulator lane in bf16.
HW_FP8_BASELINE = _hw("hw_fp8_baseline", schedule.fixed(8, 7))

#: The MX-Gemmini datapath on the tapeout ladder. Equals rtl_exact's Y_hw, 65536 of 65536 elements.
HW_FP8_TAPEOUT = _hw("hw_fp8_tapeout", schedule.HW_FINAL)


def _mlp_and_head(scheme):
    """MXQuant's layer set: every nn.Linear except the attention projections, which are left alone whole
    (a module holding q_proj and k_proj is an attention module). lm_head is quantized."""
    return [(is_attention, None), (nn.Linear, scheme)]


RULES = {
    "none": None,                                       # no patch at all: the model as loaded, in bf16
    "hw_fp8_baseline": _mlp_and_head(HW_FP8_BASELINE),
    "hw_fp8_tapeout": _mlp_and_head(HW_FP8_TAPEOUT),
}

try:                                                    # machine-local extras, not part of this repo
    from experiments._local import RULES as _extra
except ImportError:
    pass
else:
    RULES.update(_extra)
