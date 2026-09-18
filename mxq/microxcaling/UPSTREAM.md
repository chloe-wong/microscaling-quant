# Provenance

This directory was `mxq/ocp/` until 2026-09-18; renamed so that `ocp` in mxq always means the spec and
`microxcaling` means Microsoft's code.

Source: https://github.com/microsoft/microxcaling
Commit: 7bc41952de394f5cc5e782baf132e7c7542eb4e4 (2024-08-19)
License: MIT (see LICENSE, copied verbatim)

## Copied verbatim

| here | upstream | what |
|---|---|---|
| `formats.py` | `mx/formats.py` | whole file: `RoundingMode`, `ElemFormat`, `_get_min_norm`, `_get_max_norm`, `_get_format_params`, `FP32_EXPONENT_BIAS`, `FP32_MIN_NORMAL` |
| `elemwise.py` | `mx/elemwise_ops.py` | `_safe_lshift`, `_safe_rshift`, `_round_mantissa`, `_quantize_elemwise_core`, `_quantize_elemwise`, `_quantize_bfloat` |
| `blockwise.py` | `mx/mx_ops.py` | `_shared_exponents`, `_reshape_to_blocks`, `_undo_reshape_to_blocks`, `_quantize_mx` |

Function bodies are byte-for-byte upstream. `tests/test_ocp.py::test_verbatim` enforces
this against a clone of the upstream commit. The only lines we wrote in these three files
are the module docstring header and the import block.

## Not copied

- `quantize_mx_op`, `quantize_elemwise_op`, `_quantize_fp` — read an `mx_specs` config dict
  via `mx/specs.py`. Config plumbing, not algorithm.
- `mx/custom_extensions.py`, `mx/cpp/` — compiled CUDA/C++ kernels. The `custom_cuda`
  branches remain in the copied code but `custom_cuda=True` will raise ImportError.
  Default is False and nothing here sets it.
- All nn-layer drop-ins: `linear.py`, `matmul.py`, `bmm.py`, `convolution.py`, norms,
  activations, `mx_mapping.py`, `quantize.py`.
- Nothing from MXQuant's copy. MXQuant's `microxcaling/` carries eight `level2_*.py`
  files and a modified `elemwise_ops.py` (LUT hook inside `_quantize_elemwise_core`).
  None of that is here. Verified 2026-09-11: with the hook off, MXQuant's copy is
  bit-identical to this upstream on 1M samples x 4 formats x 2 rounding modes.

## Known upstream bug, kept as-is

`elemwise.py`, sparse branch at the end of `_quantize_elemwise_core`: assigns
`output = torch.sparse_coo_tensor(..., output, ...)` where `output` is undefined
(should be `out`), then returns `out`. Raises NameError for any sparse input.
Never reached: every caller passes dense tensors. Left verbatim so that the
"is this Microsoft's code" answer stays "yes". Fix belongs upstream.

## Our wrappers

`__init__.py` is the only file in this directory we authored. It exposes one function
per OCP element format with the MX-conversion flags baked in:

- `saturate_normals=True` for e4m3, e3m2, e2m3, e2m1 — these formats have no Inf;
  overflow clamps to max_norm. Upstream's own `_quantize_mx` hardcodes this.
- `saturate_normals=False` for e5m2 — the one MX element format with IEEE Inf/NaN.
- `allow_denorm=True` everywhere — OCP subnormals are representable.
- default `round="even"` (RNE, spec-recommended). MXQuant used `"nearest"`
  (ties away from zero); differential tests pass it explicitly.
