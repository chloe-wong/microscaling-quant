# MX operand capture integration

`mxq.nn.operand_capture` supplies one reusable BF16 to MX operand profile:
block size 32, E8M0 scales, a `2^-23` block maximum floor, round to nearest
even, and E4M3, E3M2, or E2M1 elements. FP4 uses an E3M1 intermediate.
It produces unsigned element codes, E8M0 exponent bytes, and an exportable
quantize/dequantize (Q/DQ) graph. FP6 accepts a reviewed 16-code LUT for
each operand. This profile was extracted from the MX Gemmini operand adapter;
the target contract still decides whether it is legal for a given RTL revision.

There are two different TorchAO configurations in this package:

| Configuration | Purpose |
| --- | --- |
| `mxq.nn.torchao.MXQConfig` | Full `MXLinear` arithmetic simulation, with a selectable reduction. |
| `mxq.nn.operand_capture.MXOperandFakeQuantConfig` | Operand Q/DQ for static `nn.Linear` weights and dynamic activations during compiler capture. |

The capture configuration uses PyTorch BF16 for the contraction. Its output
is a capture diagnostic, not an accelerator arithmetic golden. It does not
pack tiles, schedule a mesh, model reductions or transfers, or certify a
full model.

## Target integration

The target package validates the chosen RTL revision and software contract
before calling any capture operation. It owns operator and shape legality,
the authored per-site precision policy, FP6 LUT review, output chains,
packing, lowering, and manifest verification. The shared library does not
guess a precision from accuracy or select all eligible sites.

For modules, use the exact-site TorchAO helper:

```python
from mxq.nn.operand_capture import MXOperandFakeQuantConfig, quantize_selected_linear_modules_

quantize_selected_linear_modules_(
    model, {"encoder.project": MXOperandFakeQuantConfig(format="mxfp8")}
)
```

For functional `matmul`, `mm`, `bmm`, `linear`, and `addmm` sites in an
exported graph, the target supplies its policy lookup and shape gate:

```python
from mxq.nn.operand_capture import CaptureDecision, quantize_functional_contractions_

census = quantize_functional_contractions_(
    exported.module(),
    select=lambda site_id: authored_policy[site_id],  # CaptureDecision or legacy format string
    contract=compiled_contract,
    shape_reason=target_shape_reason,
    codebooks_for=reviewed_fp6_codebooks,
)
```

The pass returns every visible functional contraction with its disposition:
`quantized`, `host`, `preserved`, or `skipped` with a reason. `preserved` means
the tensor remains in its source dtype for an explicitly named accelerator
execution route; it requires a model2MLIR `m2m.quantization_manifest.v2` receipt.
`CaptureDecision("refuse", reason=...)` stops capture rather than silently
selecting another format. A fused SDPA site must be
exposed with `expose_sdpa_contractions` before running the pass. The target
adapter adds module sites, verifies exact graph identity and policy coverage,
and emits the model2MLIR quantization manifest with contract and policy
digests. The target package registers the `m2m.quantization_adapters` entry
point; `mxq` does not register a target adapter.

For MX-Gemmini, the out-of-tree adapter verifies its standalone RTL and contract,
then chooses MX FP8, FP6, FP4, or host at each site. For Radiance, an adapter
can preserve selected SIMT float sites using
`CaptureDecision("preserve", execution_route="simt_float")` and apply this MX
profile only to sites assigned to the contained MX PE. Its RTL configuration
differs from the standalone MX-Gemmini configuration, so the adapter must
verify numerical equivalence against that configuration before selecting MX.
For Atlas, the selected software spec describes FP8 E4M3 operands but leaves
the model operand scale encoding and block size unknown. Its adapter must
refuse hardware FP8 quantization until those rules and the resulting encoding
are reviewed; the E8M0-named result pack ports do not settle operand scaling.
These are capture routes, not claims of executable lowering or full-model
numerical qualification.
