"""What a LUT may hold and how the chip's finder reads it: element codes, their values, and the finder's
fixed-point view of them, for the four MX-Gemmini LUT element formats.

    decode(codes, fmt)        element codes -> float32 values (mx_fp_math.h's decoders)
    values(fmt)               the values a table may hold: finite, and distinct to the finder
    encode(T, fmt)            exact table values -> element codes; raises on a value the format lacks
    finder(codes, T, fmt, group=G)   the chip's index for each element code: nearest in fixed point, ties low

The finder does not compare floats. It turns the element code and every table entry into a fixed-point
magnitude and takes the smallest masked |difference| (`*NearestFinder.scala`, mirrored in
`mx_fp_math.h::*_to_fixed_point`). The masks make large magnitudes wrap onto small ones (E3M2: 4.0 and 20.0
both read 64, 16.0 reads 0), so a table holding such a value is wrong, not merely coarse. `values` is
therefore every value the finder can tell apart, keeping the smaller magnitude of any colliding pair: the one
whose fixed point is faithful.

Transcribed from npu-exploration compiler/codebook.py (`_fixed_point`, `_DIFF_MASK`, `codebook_values`,
`finder_indices`), which is verified against a C oracle built from mx_fp_math.h (4096/4096) and against spike.
"""
from functools import lru_cache
from typing import Union

import torch

from ..element_quant.formats import Format, get

__all__ = ["SIZE", "FORMATS", "decode", "values", "encode", "finder"]

#: entries per table: the index on the wire is a nibble
SIZE = 16

#: the element formats MX-Gemmini sends through a LUT (E4M3 is the quad PE's)
FORMATS = ("MXFP6_E3M2", "MXFP6_E2M3", "MXFP8_E5M2", "MXFP8_E4M3")


def _fx_e3m2(v: int) -> int:                       # 1s|3e|2m, sigW 3, 9-bit shift, 8-bit magnitude
    sign, exp, mant = (v >> 5) & 1, (v >> 2) & 0x7, v & 0x3
    if exp == 0 and mant == 0:
        return 0
    s_exp = -2 if exp == 0 else exp - 3
    sig = (0 if exp == 0 else 4) | mant
    mag = ((sig << ((s_exp + 2) & 0b111)) & 0x1FF) & 0xFF
    return -mag if sign else mag


def _fx_e2m3(v: int) -> int:                       # exact: value * 8, no mask
    sign, exp, mant = (v >> 5) & 1, (v >> 3) & 0x3, v & 0x7
    mag = mant if exp == 0 else ((8 + mant) << (exp - 1))
    return -mag if sign else mag


def _fx_e5m2(v: int) -> int:                       # sigW 3, shiftW 5, fixedW 32
    sign, exp, mant = (v >> 7) & 1, (v >> 2) & 0x1F, v & 0x3
    if exp == 0 and mant == 0:
        return 0
    sig = (0 if exp == 0 else 4) | mant
    s_exp = (1 - 15) if exp == 0 else exp - 15
    mag = (sig << ((s_exp + 14) & 0x1F)) & 0xFFFFFFFF
    return -mag if sign else mag


def _fx_e4m3(v: int) -> int:                       # sigW 4, shiftW 4, fixedW 18
    sign, exp, mant = (v >> 7) & 1, (v >> 3) & 0xF, v & 0x7
    if exp == 0 and mant == 0:
        return 0
    sig = (0 if exp == 0 else 8) | mant
    s_exp = (1 - 7) if exp == 0 else exp - 7
    mag = (sig << ((s_exp + 6) & 0xF)) & 0x3FFFF
    return -mag if sign else mag


#: (e, m) -> (code -> fixed point, the finder's |difference| mask; None: E2M3's fixed point is exact)
_FINDER = {(3, 2): (_fx_e3m2, 0x1FF), (2, 3): (_fx_e2m3, None),
           (5, 2): (_fx_e5m2, 0x1FFFFFFFF), (4, 3): (_fx_e4m3, 0x7FFFF)}


def _format(fmt: Union[str, Format]) -> Format:
    f = get(fmt)
    if f is None or f.name not in FORMATS:
        raise ValueError(f"mxq.lut: {fmt!r} is not a LUT format; MX-Gemmini sends {', '.join(FORMATS)} through LUTs")
    return f


@lru_cache(maxsize=None)
def _tables(name: str):
    """(all code values float32, all code fixed points int64, finder-safe values float32 sorted)."""
    f = _format(name)
    e, m = f.e, f.m
    n = 1 << (1 + e + m)
    bias = (1 << (e - 1)) - 1
    vals = []
    for c in range(n):
        s, ex, man = (c >> (e + m)) & 1, (c >> m) & ((1 << e) - 1), c & ((1 << m) - 1)
        if (e, m) == (5, 2) and ex == 0x1F:                      # E5M2 alone keeps IEEE Inf/NaN (mx_fp_math.h:416)
            v = float("nan") if man else float("inf")
        elif ex == 0:
            v = man * 2.0 ** (1 - bias - m)
        else:
            v = (1.0 + man / (1 << m)) * 2.0 ** (ex - bias)
        vals.append(-v if s else v)
    fx = [_FINDER[(e, m)][0](c) for c in range(n)]

    seen = {}                                                    # fixed point -> value; the smaller magnitude wins
    for c in range(n):
        v = vals[c]
        if v != v or v in (float("inf"), float("-inf")):
            continue
        if fx[c] not in seen or abs(v) < abs(seen[fx[c]]):
            seen[fx[c]] = v
    safe = sorted(set(seen.values()))                            # -0.0 == 0.0: one zero
    return (torch.tensor(vals, dtype=torch.float32), torch.tensor(fx, dtype=torch.int64),
            torch.tensor(safe, dtype=torch.float32))


def decode(codes: torch.Tensor, fmt: Union[str, Format]) -> torch.Tensor:
    """Element codes (integers) -> their float32 values."""
    table = _tables(_format(fmt).name)[0].to(codes.device)
    return table[codes.long()]


def values(fmt: Union[str, Format]) -> torch.Tensor:
    """The values a table may hold, sorted ascending: finite, and no two with the same fixed point."""
    return _tables(_format(fmt).name)[2].clone()


def encode(T: torch.Tensor, fmt: Union[str, Format]) -> torch.Tensor:
    """Exact values -> element codes (the lowest code of each value). Raises on a value the format lacks."""
    f = _format(fmt)
    table = _tables(f.name)[0]
    lookup = {}
    for c, bits in enumerate(table.view(torch.int32).tolist()):
        lookup.setdefault(bits, c)
    flat = T.detach().to("cpu", torch.float32).reshape(-1).view(torch.int32).tolist()
    missing = [b for b in flat if b not in lookup]
    if missing:
        bad = torch.tensor(missing[:1], dtype=torch.int32).view(torch.float32).item()
        raise ValueError(f"mxq.lut: {bad!r} is not an exact {f.name} value, so it cannot be a table entry")
    return torch.tensor([lookup[b] for b in flat], dtype=torch.int64, device=T.device).reshape(T.shape)


def finder(codes: torch.Tensor, T: torch.Tensor, fmt: Union[str, Format], *, group: int) -> torch.Tensor:
    """The chip's index for each element code of a K×n tensor, against its column group's table.

    codes  K×n element codes (integers): what the requantizer rounded each output to
    T      (n >> group) × SIZE table values, column j reads table j >> group
    Nearest in the finder's fixed point, |difference| masked, ties to the lower index (`Mux(d1 <= d2, i1, i2)`).
    Returns K×n int64 indices."""
    f = _format(fmt)
    _, mask = _FINDER[(f.e, f.m)]
    fx_all = _tables(f.name)[1].to(codes.device)
    n = codes.shape[1]
    if T.shape[0] != n >> group or n % (1 << group):
        raise ValueError(f"mxq.lut.finder: {n} columns at group {group} need {n >> group} tables, got {T.shape[0]}")
    fx = fx_all[codes.long()]                                                   # K×n
    lut_fx = fx_all[encode(T, f).to(codes.device)]                              # (n >> G)×SIZE
    cols = torch.arange(n, device=codes.device) >> group
    d = (fx.unsqueeze(-1) - lut_fx[cols].unsqueeze(0)).abs()                    # K×n×SIZE
    if mask is not None:
        d = d & mask
    return d.argmin(dim=-1)                                                     # first minimum: the lower index
