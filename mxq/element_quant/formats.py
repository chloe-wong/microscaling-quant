"""Element format table shared by every block quantizer.

One row per OCP MX element format. `e`/`m` are the exponent and mantissa widths used by
float_em; `emax` and `max_norm` are the OCP spec values; `ocp` is the mxq.ocp ElemFormat name.
"FP32" means no quantization (pass-through) and maps to None.
"""
from dataclasses import dataclass
from typing import Optional, Union

__all__ = ["Format", "FORMATS", "get"]


@dataclass(frozen=True)
class Format:
    name: str
    e: int
    m: int
    emax: int
    max_norm: float
    ocp: str


FORMATS = {
    "MXFP8_E4M3": Format("MXFP8_E4M3", 4, 3,  8,   448.0, "fp8_e4m3"),
    "MXFP8_E5M2": Format("MXFP8_E5M2", 5, 2, 15, 57344.0, "fp8_e5m2"),
    "MXFP6_E3M2": Format("MXFP6_E3M2", 3, 2,  4,    28.0, "fp6_e3m2"),
    "MXFP6_E2M3": Format("MXFP6_E2M3", 2, 3,  2,     7.5, "fp6_e2m3"),
    "MXFP4":      Format("MXFP4",      2, 1,  2,     6.0, "fp4_e2m1"),
}

#: alternative spellings accepted by get(); values are FORMATS keys.
_ALIASES = {
    "FP8_E4M3": "MXFP8_E4M3", "E4M3": "MXFP8_E4M3", "MX_E4M3": "MXFP8_E4M3",
    "FP8_E5M2": "MXFP8_E5M2", "E5M2": "MXFP8_E5M2", "MX_E5M2": "MXFP8_E5M2",
    "FP6_E3M2": "MXFP6_E3M2", "E3M2": "MXFP6_E3M2",
    "FP6_E2M3": "MXFP6_E2M3", "E2M3": "MXFP6_E2M3",
    "FP4": "MXFP4", "FP4_E2M1": "MXFP4", "E2M1": "MXFP4", "MXFP4_E2M1": "MXFP4",
}


def get(fmt: Union[str, Format, None]) -> Optional[Format]:
    """Resolve a format name (case-insensitive, aliases allowed). "FP32"/None -> None."""
    if fmt is None or isinstance(fmt, Format):
        return fmt
    key = fmt.strip().upper()
    if key == "FP32":
        return None
    key = _ALIASES.get(key, key)
    if key not in FORMATS:
        raise ValueError(f"Unknown format '{fmt}'. Known: {sorted(FORMATS)} + FP32")
    return FORMATS[key]
