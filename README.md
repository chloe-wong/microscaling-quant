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

Product / accumulator quantization to any float(e, m):

```python
from mxq.element_quant import float_em
S = float_em.quantize(S, e=6, m=9)              # bit-identical to qtorch.float_quantize(..., "nearest")
```

## Structure

```
mxq/
  scale_factor/      step 1   mxquant(amax) = 2^floor(log2 amax)        ocp(amax, emax) = 2^(floor(log2 amax) - emax)
  element_quant/     step 2   float_em(x, e, m, round=, grid=)  grid: qtorch (MXQuant) | ieee (accumulators) | ocp (MX operands)
                              microsoft: microxcaling _quantize_elemwise, verbatim, reference only
                              formats.py: one table of e, m, emax, max_norm per format
  block_mxquant/     scale_factor.mxquant + float_em grid=qtorch  -> MXQuant's simulation (all reported perplexities)
  block_mxgemmini/   scale_factor.mxquant + float_em grid=ocp     -> MX-Gemmini operand codes
  block_ocp/         scale_factor.ocp     + element_quant.microsoft  -> OCP MX v1.0, validated against mxq.ocp
  ocp/               Microsoft microxcaling code, verbatim (MIT). Oracle only; see ocp/UPSTREAM.md
  rounding/          ties_away | rne | truncate, on float32 bit patterns (round_bits) or integers (round_int)
  schedule.py        per-lane accumulator formats: load(csv), fixed(e, m), HW_FINAL (the tapeout schedule)
  arith.py           exact_add (exact sum, one rounding), saturate_product: the operations between quantizations
  _blocks.py         split along an axis into 32-blocks, pad, reassemble
Notes/FP_Notes.md    MXQuant vs OCP: scale factor and element quantization differences, measured
```

`block_mxquant` and `block_mxgemmini` share the scale rule (block max in [1, 2)) and differ only in the element
grid: qtorch 0.2.0's (no true subnormals, top exponent reserved) vs the OCP element formats' (subnormals kept).
`block_ocp` differs in both steps: block max in the format's top binade (448 for E4M3), OCP element grid.
Details and measurements: `Notes/FP_Notes.md`.

## Validation

Tests are local (not in the repo) and differential: each module is checked bit-for-bit against the code it
replaces, on CPU and CUDA.

| module | oracle |
|---|---|
| `element_quant.float_em` | grid qtorch: `qtorch.quant.float_quantize`, 15 (e, m) pairs, 1M samples each; grid ieee: gemmini golden `fp_quantize_rne`; grid ocp: microxcaling `_quantize_elemwise` |
| `block_mxquant` | MXQuant `mx_block32_quantize` (two copies), codes and scales |
| `block_mxgemmini` | MXQuant `quantize_mx_block32` (round nearest); operands of npu-exploration `rtl_exact` saved hardware test case |
| `block_ocp` | Microsoft `_quantize_mx`; codes checked to be in the format's code set, scales E8M0 |
| `ocp/` | upstream microxcaling clone, AST-verbatim and numeric |

Running them needs `qtorch`, an MXQuant checkout (`MXQUANT_ROOT`) and an upstream microxcaling clone
(`MICROXCALING_UPSTREAM`); defaults point at the firesim2 paths.

```bash
python -m pytest -q tests
```
