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
from mxq import block_mxquant, block_mxgemmini, block_ocp

V = torch.randn(4096, 512)                      # e.g. A = xᵀ (K×M), blocks of 32 along K
P, X = block_mxquant.quantize(V, "MXFP8_E4M3", axis=0)     # MXQuant simulation (qtorch 0.2.0 element grid)
P, X = block_mxgemmini.quantize(V, "MXFP8_E4M3", axis=0)   # MX-Gemmini operand codes (OCP element grid)
P, X = block_ocp.quantize(V, "MXFP8_E4M3", axis=0)         # OCP MX v1.0 (Microsoft reference)
V_hat = block_mxquant.dequantize(P, X, axis=0)             # == P * expand(X)
```

`P` has `V`'s shape. `X` has `V`'s shape with the block axis of length ceil(len/32) (last block zero-padded).
Formats: `MXFP8_E4M3`, `MXFP8_E5M2`, `MXFP6_E3M2`, `MXFP6_E2M3`, `MXFP4`, `FP32` (pass-through).

A matmul on the codes is a reducer (the order of the additions) plus an Arithmetic (the rounding at each step)
plus a schedule (the accumulator format at each position). Every piece is named explicitly; there are no presets:

```python
from mxq import block_mxgemmini, matmul, schedule
P_A, X_A = block_mxgemmini.quantize(x.t(), "MXFP8_E4M3", axis=0)      # A = xᵀ, K×M
P_B, X_B = block_mxgemmini.quantize(W.t(), "MXFP8_E4M3", axis=0)      # B = Wᵀ, K×N
Y = matmul.systolic(P_A, X_A, P_B, X_B, matmul.MXGEMMINI(), schedule.HW_FINAL)    # M×N, bit-identical to MX-Gemmini
Y = matmul.systolic(P_A, X_A, P_B, X_B, matmul.MXQUANT(4, 3), schedule.HW_FINAL)  # MXQuant's simulation
Y = matmul.ipt(P_A, X_A, P_B, X_B, matmul.MXGEMMINI(), [(4, 4), (4, 4), (4, 4), (8, 7)])   # adder tree, one format per level
Y = matmul.fp64_accum(P_A, X_A, P_B, X_B)                                          # same codes, no rounding anywhere
```

`matmul.MXQUANT` and `matmul.MXGEMMINI` are Arithmetics, not matmuls: each bundles the three rounding functions a
reducer calls (product, accumulate, add a finished block). Their datapaths are written out stage by stage in
`mxq/matmul/_arithmetic.py`; the table below is the summary.

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
  block_mxquant.py   scale_factor.mxquant + float_em grid=qtorch  -> MXQuant's simulation (all reported perplexities)
  block_mxgemmini.py scale_factor.mxquant + float_em grid=ocp     -> MX-Gemmini operand codes
  block_ocp.py       scale_factor.ocp     + element_quant.microsoft  -> OCP MX v1.0, validated against mxq.ocp
  ocp/               Microsoft microxcaling code, verbatim (MIT). Oracle only; see ocp/UPSTREAM.md
  rounding/          ties_away | rne | truncate, on float32 bit patterns (round_bits) or integers (round_int)
  arith.py           exact_add, truncate_significand, saturate_product: what a PE does between quantizations
  matmul/            Y = Aᵀ·B from codes and scales
    _arithmetic.py   Arithmetic(product, acc_add, tile_add); MXQUANT(prod_e, prod_m) and MXGEMMINI(): the datapaths, stage by stage
    _systolic.py     the PE column (window deep, 16 for the tapeout): per-k product, per-lane accumulate, per-block rescale and accumulate
    _ipt.py          the inner-product tree: fanin (16) products at once, log2(fanin) adder levels each with its own format, per-block rescale and accumulate
    _fp64_accum.py   same inputs, no rounding anywhere: the reducer's error floor
    _common.py       operand shape, dtype and device checks, schedule length check, per-block scale map
  schedule.py        one float(e, m) per accumulator position: load(csv, rows), fixed(e, m, rows), HW_FINAL; exactly rows entries or ValueError
  scheme.py          Scheme(name, act, weight, reduce): a container for one explicit chain, .matmul(A, B); no presets
  _blocks.py         split along an axis into 32-blocks, pad, reassemble; BLOCK = 32
Notes/FP_Notes.md    MXQuant vs OCP: scale factor and element quantization differences, measured
```

`block_mxquant` and `block_mxgemmini` share the scale rule (block max in [1, 2)) and differ only in the element
grid: qtorch 0.2.0's (no true subnormals, top exponent reserved) vs the OCP element formats' (subnormals kept).
`block_ocp` differs in both steps: block max in the format's top binade (448 for E4M3), OCP element grid.
Details and measurements: `Notes/FP_Notes.md`.

Name history: on `main` before this branch, `block_mxgemmini` was the name of what is now `block_mxquant`
(qtorch grid). The current `block_mxgemmini` produces the hardware's operand codes (OCP grid). Code written
against the old name must switch to `block_mxquant` to keep its numbers.

## Validation

Tests are local (not in the repo) and differential: each module is checked bit-for-bit against the code it
replaces, on CPU and CUDA.

| module | oracle |
|---|---|
| `element_quant.float_em` | grid qtorch: `qtorch.quant.float_quantize`, 15 (e, m) pairs, 1M samples each; grid ieee: gemmini golden `fp_quantize_rne`; grid ocp: microxcaling `_quantize_elemwise` |
| `block_mxquant` | MXQuant `mx_block32_quantize` (two copies), codes and scales |
| `block_mxgemmini` | MXQuant `quantize_mx_block32` (round nearest); operands of npu-exploration `rtl_exact` saved hardware test case |
| `matmul.systolic` + `MXQUANT` | MXQuant `MXLinearSim._simulate_atw`, bit-identical, 3 schedules × 3 product formats |
| `matmul.systolic` + `MXGEMMINI` | hardware output `Y_hw` of the `rtl_exact` test case (TinyLlama MLP), 65536/65536 identical |
| `matmul.ipt` | scalar per-element tree in plain Python, bit-identical, fanin 2 to 32 with K tails; `fp64_accum` without rounding |
| `Scheme` | `.matmul` equals the explicit quantizer + reducer calls |
| `block_ocp` | Microsoft `_quantize_mx`; codes checked to be in the format's code set, scales E8M0 |
| `ocp/` | upstream microxcaling clone, AST-verbatim and numeric |
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
