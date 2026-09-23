# microscaling-quant (`mxq`)

Granular microscaling (MX) quantization ops, cleaned up from MXQuant. Pure PyTorch, no compiled extensions.

## Setup

Requires Python >= 3.10 and torch.

```bash
git clone git@github.com:chloe-wong/microscaling-quant.git
cd microscaling-quant
pip install -e .            # inside the conda env you run experiments in
python -c "import mxq"
```

## Use

Block quantization = (1) a per-block scale factor, then (2) element quantization of the scaled values.
Every block quantizer has one interface: a tensor in, codes `P` and power-of-two scales `X` out.

```python
import torch
from mxq import block

V = torch.randn(4096, 512)                      # e.g. A = xᵀ (K×M), blocks of 32 along K
P, X = block.mxquant.quantize(V, "MXFP8_E4M3", axis=0)     # MXQuant simulation (qtorch 0.2.0 element grid)
P, X = block.mxgemmini.quantize(V, "MXFP8_E4M3", axis=0)   # MX-Gemmini operand codes (OCP element grid)
P, X = block.ocp.quantize(V, "MXFP8_E4M3", axis=0)         # OCP MX v1.0 (Microsoft reference)
V_hat = block.mxquant.dequantize(P, X, axis=0)             # == P * expand(X)
```

`P` has `V`'s shape. `X` has `V`'s shape with the block axis of length ceil(len/32) (last block zero-padded).
Formats: `MXFP8_E4M3`, `MXFP8_E5M2`, `MXFP6_E3M2`, `MXFP6_E2M3`, `MXFP4`, `FP32` (pass-through).

A matmul on the codes is a reducer (the order of the additions) plus an Arithmetic (the rounding at each step)
plus a schedule (the accumulator format at each position). Every piece is named explicitly; there are no presets:

```python
from mxq import block, fp64_accum, matmul, schedule
P_A, X_A = block.mxgemmini.quantize(x.t(), "MXFP8_E4M3", axis=0)      # A = xᵀ, K×M
P_B, X_B = block.mxgemmini.quantize(W.t(), "MXFP8_E4M3", axis=0)      # B = Wᵀ, K×N
Y = matmul.systolic(P_A, X_A, P_B, X_B, matmul.MXGEMMINI(), schedule.HW_FINAL)    # M×N, bit-identical to MX-Gemmini
Y = matmul.systolic(P_A, X_A, P_B, X_B, matmul.MXQUANT(4, 3), schedule.HW_FINAL)  # MXQuant's simulation
Y = fp64_accum(P_A, X_A, P_B, X_B)                                                 # same codes, no rounding inside the multiply
```

`matmul.MXQUANT` and `matmul.MXGEMMINI` are Arithmetics, not matmuls: each bundles the three rounding functions a
reducer calls (product, accumulate, add a finished block). Their datapaths are written out stage by stage in
`mxq/matmul/_arithmetic.py`; the table below is the summary.

Into a model: a rule list says which Scheme each `nn.Linear` runs through, by layer name (`*` is a wildcard) or
by layer type. First matching rule wins; `None` leaves the layer as it is.

```python
from functools import partial
from torch import nn
from mxq import Scheme, block, matmul, schedule
from mxq.nn import patch

q = partial(block.mxgemmini.quantize, fmt="MXFP8_E4M3", axis=0)
chain = Scheme("mesh_fp8", a=q, b=q, reduce=partial(matmul.systolic, arith=matmul.MXGEMMINI(), schedule=schedule.HW_FINAL))

rules = [("*.self_attn.*", None),        # attention projections stay as they are
         ("lm_head",       None),
         (nn.Linear,       chain)]       # every other Linear
patch(model, rules, dry_run=True)        # print which layer gets what; change nothing
handle = patch(model, rules)             # replace the chosen layers with mxq.nn.MXLinear
handle.revert()                          # put the originals back
```

Product / accumulator quantization to any float(e, m):

```python
from mxq.element_quant import float_em
S = float_em.quantize(S, e=6, m=9)              # bit-identical to qtorch.float_quantize(..., "nearest")
S = float_em.quantize(S, 8, 7, rounding_mode="rne", grid="ieee")   # bf16, as the mesh lanes round
```

## The two Arithmetic datapaths

| stage | `MXQUANT(prod_e, prod_m)` | `MXGEMMINI()` |
|---|---|---|
| product | fp32 multiply, then `float_em` ties-away on the qtorch grid to float(prod_e, prod_m) | `arith.truncate_significand` to prod_m fraction bits (no exponent clamp), then `arith.saturate_product` (448 for e4m3) |
| accumulate into lane float(e, m) | fp32 add, then `float_em` ties-away on the qtorch grid | both addends `float_em` RNE on the ieee grid, then `arith.exact_add` (exact sum, one RNE rounding) |
| add finished block into output | fp32 add, no rounding | both rounded RNE to bf16, then `arith.exact_add` to bf16 |
| matches | MXQuant `MXLinearSim._simulate_atw`, bit-identical | `rtl_exact` hardware output `Y_hw` and the gemmini golden model, bit-identical |
| validated for | 6 operand formats | MXFP8_E4M3 operands, prod (4, 3), `HW_FINAL` lanes |

## Structure

```
mxq/
  scale_factor.py    step 1   mxquant(amax) = 2^floor(log2 amax)        ocp(amax, emax) = 2^(floor(log2 amax) - emax)
  element_quant/     step 2   float_em(x, e, m, rounding_mode=, grid=)  grid: qtorch (MXQuant) | ieee (accumulators) | ocp (MX operands)
                              microsoft: microxcaling _quantize_elemwise, verbatim, reference only
                              formats.py: one table of e, m, emax, max_norm per format
  block/             step 1 + step 2 composed; one interface: P, X = quantize(V, fmt, axis), V_hat = dequantize(P, X, axis)
    mxquant.py       scale_factor.mxquant + float_em grid=qtorch  -> MXQuant's simulation (all reported perplexities)
    mxgemmini.py     scale_factor.mxquant + float_em grid=ocp     -> MX-Gemmini operand codes
    ocp.py           scale_factor.ocp     + element_quant.microsoft  -> OCP MX v1.0, validated against mxq.microxcaling
    _driver.py       split along an axis into 32-blocks, pad, run the two steps, reassemble; BLOCK = 32
  microxcaling/      Microsoft's microxcaling package, verbatim (MIT). Oracle only; see microxcaling/UPSTREAM.md
  rounding/          ties_away | rne | truncate, on float32 bit patterns (round_bits) or integers (round_int)
  arith.py           exact_add, truncate_significand, saturate_product: what a PE does between quantizations
  matmul/            the array dataflows: Y = Aᵀ·B from codes and scales, summed in the hardware's order
    _arithmetic.py   Arithmetic(product, acc_add, tile_add); MXQUANT(prod_e, prod_m) and MXGEMMINI(): the datapaths, stage by stage
    _systolic.py     the PE column (window deep, 16 for the tapeout): per-k product, per-lane accumulate, per-block rescale and accumulate
    _common.py       operand shape, dtype and device checks, schedule length check, per-block scale map
  _fp64_accum.py     fp64_accum: the same codes with no rounding inside the multiply, the error floor; not an architecture
  schedule.py        one float(e, m) per accumulator position: load(csv, rows), fixed(e, m, rows), HW_FINAL; exactly rows entries or ValueError
  scheme.py          Scheme(name, a, b, reduce): one explicit chain for one matmul, .matmul(A, B); no presets
  nn/                putting Schemes into a model
    _linear.py       MXLinear: one nn.Linear through one Scheme; weight codes cached, token rows chunked (bit-identical)
    _patch.py        patch(model, rules): a Scheme per layer name or layer type, first match wins; dry_run, revert
Notes/FP_Notes.md    MXQuant vs OCP: scale factor and element quantization differences, measured
```

`block.mxquant` and `block.mxgemmini` share the scale rule (block max in [1, 2)) and differ only in the element
grid: qtorch 0.2.0's (no true subnormals, top exponent reserved) vs the OCP element formats' (subnormals kept).
`block.ocp` differs in both steps: block max in the format's top binade (448 for E4M3), OCP element grid.
Details and measurements: `Notes/FP_Notes.md`.

Name history: on `main` before this branch, `block_mxgemmini` was the name of what is now `block.mxquant`
(qtorch grid). The current `block.mxgemmini` produces the hardware's operand codes (OCP grid). Code written
against the old name must switch to `block.mxquant` to keep its numbers. `mxq.ocp` (Microsoft's code) is now
`mxq.microxcaling`; `ocp` in mxq always means the OCP spec.

## Validation

Tests are local (not in the repo) and differential: each module is checked bit-for-bit against the code it
replaces, on CPU and CUDA.

| module | oracle |
|---|---|
| `element_quant.float_em` | grid qtorch: `qtorch.quant.float_quantize`, 15 (e, m) pairs, 1M samples each; grid ieee: gemmini golden `fp_quantize_rne`; grid ocp: microxcaling `_quantize_elemwise` |
| `block.mxquant` | MXQuant `mx_block32_quantize` (two copies), codes and scales |
| `block.mxgemmini` | MXQuant `quantize_mx_block32` (round nearest); operands of npu-exploration `rtl_exact` saved hardware test case |
| `matmul.systolic` + `MXQUANT` | MXQuant `MXLinearSim._simulate_atw`, bit-identical, 3 schedules × 3 product formats |
| `matmul.systolic` + `MXGEMMINI` | hardware output `Y_hw` of the `rtl_exact` test case (TinyLlama MLP), 65536/65536 identical |
| `Scheme` | `.matmul` equals the explicit quantizer + reducer calls |
| `nn.MXLinear` | MXQuant `MXLinearSim.forward`, bit-identical (bf16 inputs, bias, three lengths, two ladders); every chunk size equals unchunked |
| `nn.patch` | the `rtl_exact` MLP built from `nn.Linear` layers and patched by type: `Y_hw` 65536/65536; rule order, unused-rule and bad-Scheme errors, revert, tied weights |
| `block.ocp` | Microsoft `_quantize_mx`; codes checked to be in the format's code set, scales E8M0 |
| `microxcaling/` | upstream microxcaling clone, AST-verbatim and numeric |
| `rounding` | qtorch (ties away), torch bf16 and gemmini golden `_rne_e8` (RNE), golden `mx_product_quantize_trunc` (truncate) |
| `scale_factor` | MXQuant `mx_block32_quantize` scales; Microsoft `_quantize_mx` shared exponents |
| `element_quant.formats` | microxcaling `ElemFormat` table |
| `arith` | gemmini golden `fp_add_exact`, `bf16_accum_add`, `mx_product_quantize_trunc`, `mx_product_saturate` |
| `schedule` | MXQuant `load_schedule` on both CSV layouts and 400 real files |

Running them needs `qtorch`, an MXQuant checkout (`MXQUANT_ROOT`) and an upstream microxcaling clone
(`MICROXCALING_UPSTREAM`); defaults point at the firesim2 paths.

```bash
python -m pytest -q tests
```
