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

For modules, call TorchAO with an exact site filter:

```python
from torchao.quantization import quantize_
from mxq.nn.operand_capture import MXOperandFakeQuantConfig

quantize_(model, MXOperandFakeQuantConfig(format="mxfp8"),
          filter_fn=lambda module, fqn: fqn == "encoder.project")
```

For functional `matmul`, `mm`, `bmm`, `linear`, and `addmm` sites in an
exported graph, the target supplies its policy lookup and shape gate:

```python
from mxq.nn.operand_capture import quantize_functional_contractions_

census = quantize_functional_contractions_(
    exported.module(),
    select=lambda site_id: authored_policy[site_id],
    contract=compiled_contract,
    shape_reason=target_shape_reason,
    codebooks_for=reviewed_fp6_codebooks,
)
```

The pass returns every visible functional contraction with its disposition:
`quantized`, `host`, or `skipped` with a reason. A fused SDPA site must be
exposed with `expose_sdpa_contractions` before running the pass. The target
adapter adds module sites, verifies exact graph identity and policy coverage,
and emits the model2MLIR quantization manifest with contract and policy
digests. The target package registers the `m2m.quantization_adapters` entry
point; `mxq` does not register a target adapter.

This separation lets another target use the same operand profile only after
its own reviewed contract approves the numerical semantics and shape rules.
Atlas currently lacks reviewed block size and scale encoding, so it cannot
select this profile by default. A Radiance MX PE can use it after its own
contract and validation establish compatibility.
