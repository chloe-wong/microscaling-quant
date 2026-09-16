# FP notes: input quantization, MXQuant vs OCP

Both take a matrix, split it into blocks of 32 along one axis, and run two steps per block:

    1. scale factor         X = f(amax)               amax = max |block|
    2. element quantization P = q(block / X)           P are the codes, V_hat = P * X

The two differ in the formula for step 1 and the function used for step 2.

| | MXQuant linear inputs (`mx_block32_quantize`) | OCP MX v1.0 (Microsoft `_quantize_mx`) |
|---|---|---|
| **step 1: scale factor** | `2 ^ floor(log2 amax)` | `2 ^ (floor(log2 amax) - emax)`; shared exponent clamped below at -127, NaN above 127 (E8M0 range) |
| emax | not used | e4m3: 8, e5m2: 15, e3m2: 4, e2m3: 2, e2m1: 2 |
| block max lands in | [1, 2) | top binade of the format; saturates to max_norm (448 / 57344 / 28 / 7.5 / 6) |
| codes actually used | only those <= 2 (fp4: 0, 0.5, 1, 1.5, 2) | whole format |
| headroom above block max | 8 binades (e4m3) | none |
| zero block | scale clamped to 1e-38 | shared exponent clamped |
| **step 2: element quantization** | qtorch `float_quantize(z, exp=e, man=m, rounding="nearest")` | Microsoft `_quantize_elemwise(z, fmt, round, saturate_normals=True, allow_denorm=True)` |
| exponent bias | IEEE: 2^(e-1) - 1, top exponent reserved | OCP: top exponent used for normals (e4m3 max 448, no Inf) |
| max value, e4m3 | 240 (never reached: block max < 2) | 448 |
| subnormals | one binade below min-normal keeps mantissa bits, then flush to 0 | true fixed-step subnormals down to 2^(emin - m) |
| rounding | nearest, ties away from zero | "even" (RNE) default; "nearest" and "floor" available |
| **output** | codes `P` and scales `X` separately | dequantized `P * X` only |
| **where in mxq** | `mxq.block_mxgemmini.quantize(V, fmt, axis)` | `mxq.block_ocp.quantize(V, fmt, axis)`; reference impl in `mxq.ocp.block_quantize` |

## Measured differences (firesim2, 2026-09-15)

Same power-of-two scales, values in [-2, 2], 1M Gaussian samples. Fraction of codes where
qtorch `float_quantize` != Microsoft `_quantize_elemwise`:

| format | differ | differ within normal range |
|---|---|---|
| fp8_e4m3 | 1.1 % | 0 % |
| fp6_e3m2 | 13.9 % | 0 % |
| fp6_e2m3 | 55.8 % | 0 % |
| fp4_e2m1 | 22.7 % | 0 % |

All differences are below the format's minimum normal, i.e. subnormal handling. The narrow formats
are hit hardest because their minimum normal is 1.0 and the [1, 2) scale placement puts most
elements below it.

## Relation to the RTL

The old RTL requantizer used the OCP scale factor (block max at 448). The mesh accumulates at
exponent width 4, so chaining one tile's output into the next overflowed to NaN. The requantizer
rework (gemmini 0b2cc2c) switched the RTL to the MXQuant scale factor (block max in [1, 2)).
`block_mxgemmini` is therefore the one that matches hardware today.
