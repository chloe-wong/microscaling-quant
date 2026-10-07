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

MX-Gemmini's LUT formats (`MXFP6_E3M2`, `MXFP6_E2M3`, `MXFP8_E5M2`, `MXFP8_E4M3` on the quad PE) send each
element as a 4-bit index into a 16-entry table of element values, one table per 2^G columns of a K×n operand
(rows of A, columns of B). `block.lut` is that operand, bit-identical to the chip's rule (npu-exploration
`compiler/codebook.py`, whose kernels are bit-exact on spike). Every setting is required:

```python
P, X = block.lut.quantize(V, "MXFP6_E3M2", axis=0, block_size=32, rounding_mode="rne", scale_floor=2**-23,
                          group=1, max_iters=50)               # V_hat = P * expand(X); P holds table entries
from mxq import lut                                            # the steps, on K×n block codes P
T = lut.tables(P, "MXFP6_E3M2", group=1, max_iters=50)         # (n >> G) × 16 entries
I = lut.pick(P, T, group=1)                                    # 4-bit indices (host pick: nearest, ties low)
I = lut.finder(codes, T, "MXFP6_E3M2", group=1)                # the chip's finder, for requantized outputs
```

In a Scheme, pass `rows=2**G` so MXLinear keeps each group of tokens in one call; `MXQConfig(lut={"group": G,
"max_iters": n})` does this for you.

A matmul on the codes is a reducer (the order of the additions) plus an Arithmetic (the rounding at each step)
plus a schedule (the accumulator format at each position). Every piece is named explicitly; there are no presets:

```python
from mxq import block, fp64_accum, matmul, schedule
P_A, X_A = block.mxgemmini.quantize(x.t(), "MXFP8_E4M3", axis=0)      # A = xᵀ, K×M
P_B, X_B = block.mxgemmini.quantize(W.t(), "MXFP8_E4M3", axis=0)      # B = Wᵀ, K×N
Y = matmul.systolic(P_A, X_A, P_B, X_B, matmul.MXGEMMINI(), schedule.HW_FINAL)    # M×N, bit-identical to MX-Gemmini
Y = matmul.systolic(P_A, X_A, P_B, X_B, matmul.MXQUANT(4, 3), schedule.HW_FINAL)  # MXQuant's simulation
Y = matmul.anchor_tree(P_A, X_A, P_B, X_B, matmul.MXGEMMINI())                    # MxGen's anchor tree, bf16 out
Y = matmul.adder_tree(P_A, X_A, P_B, X_B, matmul.MXGEMMINI(), [(4, 4), (4, 5), (4, 6), (8, 7)])   # one format per level
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

The attention core (S = Q·Kᵀ, softmax, O = P·V) is not a Linear: HF computes it in one function between the
projections. A rule whose value is a pair of Schemes `(qk, pv)` chooses attention modules (those holding q_proj and
k_proj) and runs their core through `mxq.nn.attend`: Q·Kᵀ through `qk`, scale, mask and softmax in fp32, P·V through
`pv`, each operand quantized once, blocks along its contraction. The model switches to the `"mxq"` attention
implementation (transformers' AttentionInterface); attention modules no rule chose run sdpa as before, and with no
such rule nothing about attention changes. The projections stay Linear rules:

```python
patch(model, [("model.layers.*.self_attn", (fp8, fp8)),   # the core: Q·Kᵀ, P·V
              (is_attention, None),                       # q/k/v/o_proj: bf16 (or a Scheme)
              (nn.Linear, fp8)])                          # MLP + lm_head
```

From TorchAO or Hugging Face (`pip install -e ".[torchao]"`): the same MXLinear behind `torchao.quantize_`, for tools
that only accept a TorchAO config (Model2MLIR, `transformers.TorchAoConfig`, lm-eval). The config is plain fields
(defaults: the MX-Gemmini tapeout), so it can go into a checkpoint's config.json; `patch` stays the primary API.

```python
from torchao.core.config import config_to_dict
from torchao.quantization import quantize_
from transformers import AutoModelForCausalLM, TorchAoConfig
from mxq.nn.torchao import MXQConfig

cfg = MXQConfig(fmt="MXFP8_E4M3")                 # rounding, scale floor, product, ladder, size: see the class
quantize_(model, cfg)                             # every nn.Linear, changed in place
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=TorchAoConfig(cfg))   # HF skips lm_head
cfg2 = MXQConfig.from_dict(config_to_dict(cfg))   # torchao's own config_from_dict cannot find classes outside torchao
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
| product | fp32 multiply, then `float_em` ties-away on the qtorch grid to float(prod_e, prod_m) | `arith.truncate_significand` to prod_m fraction bits (flushed below 2^-16), then `arith.saturate_product` (448 for e4m3) |
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
    lut.py           mxgemmini + mxq.lut                         -> MX-Gemmini LUT operand (2-D; group, max_iters required)
    _driver.py       split along an axis into 32-blocks, pad, run the two steps, reassemble; BLOCK = 32
  lut/               MX-Gemmini's look-up tables (layout from the luts branch, PR #1)
    formats.py       decode / encode element codes; values(fmt): what the finder tells apart; finder: the chip's index
    kmeans.py        tables (weighted k-means on distinct codes, snapped, padded), pick (host nearest), lookup
  microxcaling/      Microsoft's microxcaling package, verbatim (MIT). Oracle only; see microxcaling/UPSTREAM.md
  rounding/          ties_away | rne | truncate, on float32 bit patterns (round_bits) or integers (round_int)
  arith.py           exact_add, truncate_significand, saturate_product: what a PE does between quantizations
  matmul/            the array dataflows: Y = Aᵀ·B from codes and scales, summed in the hardware's order
    _arithmetic.py   Arithmetic(product, acc_add, tile_add); MXQUANT(prod_e, prod_m) and MXGEMMINI(): the datapaths, stage by stage
    _systolic.py     the PE column (`size` deep, 16 for the tapeout): per-k product, per-lane accumulate, per-block rescale and accumulate
    _anchor.py       MxGen's anchor tree: align to one anchor, integer adds, one rounding per tree; `width`, per-level `bits`
    _adder.py        an adder tree: log2(size) levels of pairwise acc_add, one float(e, m) per level; the tree is not RTL-checked
    _common.py       operand shape, dtype and device checks, schedule length check, per-block scale map
  _fp64_accum.py     fp64_accum: the same codes with no rounding inside the multiply, the error floor; not an architecture
  schedule.py        one float(e, m) per accumulator position: load(csv, rows), fixed(e, m, rows), HW_FINAL; exactly rows entries or ValueError
  scheme.py          Scheme(name, a, b, reduce, rows=1): one explicit chain for one matmul, .matmul(A, B); no presets
  nn/                putting Schemes into a model
    _linear.py       MXLinear: one nn.Linear through one Scheme; weight codes cached, token rows chunked (bit-identical)
    _patch.py        patch(model, rules): a Scheme per layer name or layer type, first match wins; (qk, pv) for an attention core; dry_run, revert
    _attention.py    attend(q, k, v, mask, scale, qk, pv): the attention core through two Schemes; the "mxq" attention implementation
    torchao.py       MXQConfig: the same MXLinear behind torchao.quantize_ (optional, needs torchao)
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
| `matmul.anchor_tree` | MxGen `MxDotProduct` in chiseltest (E4M3, 4 cores, product and accumulator Custom(8,8)): 8000 calls and 200 chained 16-product windows identical |
| `matmul.adder_tree` | each node is the `MXGEMMINI` lane add above; the arrangement into a tree has no RTL to check against |
| `Scheme` | `.matmul` equals the explicit quantizer + reducer calls |
| `lut`, `block.lut` | npu-exploration `compiler/codebook.py` (the rule its LUT kernels are bit-exact on spike with): tables, picks, finder, values, decode, 4 formats × G 0/1/2, random and TinyLlama operands, CPU and CUDA (`tests/selftest_codebook_mxq.py` there); MXLinear with a LUT Scheme: every chunk size equals unchunked |
| `nn.MXLinear` | MXQuant `MXLinearSim.forward`, bit-identical (bf16 inputs, bias, three lengths, two ladders); every chunk size equals unchunked |
| `nn.patch` | the `rtl_exact` MLP built from `nn.Linear` layers and patched by type: `Y_hw` 65536/65536; rule order, unused-rule and bad-Scheme errors, revert, tied weights |
| `nn.attend` | the one-head-per-call implementation it replaced (kept verbatim in the tests), bit-identical over 14 Schemes (systolic with the hardware, MXQuant and bf16 ladders, anchor and adder trees, fp64_accum, FP32 passthrough, MXFP4/6, LUTs of G 0/1/2 in four formats, the tapeout chain compiled) x GQA 1:1, 4:1, 8:1, batch 2 with padding, no mask, decode-like Tq 1 and 5, Tq not a multiple of rows, D 64/80/128, unchunked and chunked, CPU and CUDA, and the TinyLlama layer shape; masks of -1e4, -inf and a stray finite value; compiled Arithmetic equals eager on random ladders; FP32 codes with fp64_accum match float64 attention; a Llama with no core rule is unchanged, revert restores sdpa logits exactly; transformers 4.57 and 5.17 |
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
