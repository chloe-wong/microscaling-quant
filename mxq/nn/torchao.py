"""mxq.nn.torchao — the same MXLinear behind TorchAO's `quantize_`, for callers that only speak TorchAO.

    from mxq.nn.torchao import MXQConfig
    cfg = MXQConfig(fmt="MXFP8_E4M3", prod=[4, 3], prod_floor=-16, ladder=[[4, 4]] * 8 + ..., size=16)
    quantize_(model, cfg)                                                            # torchao
    AutoModelForCausalLM.from_pretrained(id, quantization_config=TorchAoConfig(cfg))  # Hugging Face

`mxq.nn.patch` stays the primary API (rules, dry_run, a Handle that undoes). This is a second door into the same
MXLinear: Model2MLIR, Hugging Face and lm-eval know how to call `quantize_`, not `patch`.

MXQConfig holds numbers, not a Scheme, so `config_to_dict` can write it into a checkpoint's config.json. It is
lists throughout because torchao's encoder refuses tuples; tuples given here are turned into lists.
torchao's `config_from_dict` only looks classes up in torchao's own modules, so it cannot rebuild an MXQConfig;
use `MXQConfig.from_dict` on the dict `config_to_dict` wrote.

The handler changes each Linear IN PLACE and returns the same object. Hugging Face calls `quantize_(layer, cfg)`
with each layer as the root, and a handler that returns a new module there is dropped without an error
(measured on TinyLlama, 2026-09-29: 0 of 154 layers converted; in place: 154 of 154, logits equal to `patch`).

Requires torchao (`pip install microscaling-quant[torchao]`); `import mxq` does not import this module.
"""
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import List, Optional

import torch
from torch import nn
from torchao.core.config import AOBaseConfig
from torchao.quantization.transform_module import register_quantize_module_handler

from .. import block, matmul
from .._fp64_accum import fp64_accum
from ..element_quant.formats import get
from ..scheme import Scheme
from ..schedule import HW_FINAL
from ._linear import MXLinear

__all__ = ["MXQConfig"]

REDUCERS = ("hardware", "exact")


@dataclass
class MXQConfig(AOBaseConfig):
    """One Scheme as plain fields. Defaults are the MX-Gemmini tapeout: RNE, the 2^-23 block-max floor, E4M3
    products flushed below 2^-16, the HW_FINAL lane ladder, a 16-deep column (size).

    fmt, rounding_mode, scale_floor, via, block_size   -> block.mxgemmini.quantize (both operands)
    prod, prod_floor                                   -> matmul.MXGEMMINI(prod_e, prod_m, prod_floor)
    ladder, size                                       -> matmul.systolic(schedule=ladder, size=size)
    reduce       "hardware": the systolic column above; "exact": fp64_accum (same codes, no rounding inside)
    compiled     matmul.compiled on the Arithmetic, bit-identical to eager. None (default): compiled when the
                 layer's weight is on a GPU, eager on CPU (compilation needs Triton); True / False force it
    chunk        MXLinear's token chunk (None: its default)
    """
    fmt: str = "MXFP8_E4M3"
    rounding_mode: str = "rne"
    scale_floor: float = 2.0 ** -23
    via: Optional[List[int]] = None
    block_size: int = 32
    prod: List[int] = field(default_factory=lambda: [4, 3])
    prod_floor: Optional[int] = -16
    ladder: List[List[int]] = field(default_factory=lambda: [list(e) for e in HW_FINAL])
    size: int = 16
    reduce: str = "hardware"
    compiled: Optional[bool] = None
    chunk: Optional[int] = None
    name: str = "mxq"

    def __post_init__(self):
        if get(self.fmt) is None:
            raise ValueError("MXQConfig: fmt FP32 is no quantization; leave the layer out instead")
        if self.reduce not in REDUCERS:
            raise ValueError(f"MXQConfig: reduce {self.reduce!r}; choose from {', '.join(REDUCERS)}")
        self.prod = _pair(self.prod, "prod")
        self.ladder = [_pair(e, f"ladder[{i}]") for i, e in enumerate(self.ladder)]
        if self.via is not None:
            self.via = _pair(self.via, "via")
        if len(self.ladder) != self.size:
            raise ValueError(f"MXQConfig: ladder has {len(self.ladder)} lanes for a {self.size}-deep column "
                             "(one accumulator format per lane)")
        if self.prod_floor is not None and not _is_int(self.prod_floor):
            raise ValueError(f"MXQConfig: prod_floor {self.prod_floor!r} must be an integer exponent or None")
        if not _is_int(self.block_size) or self.block_size < 1 or self.block_size % self.size:
            raise ValueError(f"MXQConfig: block_size {self.block_size!r} must be a multiple of size {self.size}")
        if self.compiled is not None and not isinstance(self.compiled, bool):
            raise ValueError(f"MXQConfig: compiled {self.compiled!r} must be True, False or None")
        if self.chunk is not None and (not _is_int(self.chunk) or self.chunk < 1):
            raise ValueError(f"MXQConfig: chunk {self.chunk!r} must be a positive integer or None")

    def scheme(self, device=None) -> Scheme:
        """The Scheme these fields describe, for a layer on `device` (which decides `compiled=None`). Built once per
        config and device type and shared by every layer it converts, so a compiled Arithmetic is compiled once,
        not once per layer. Change a field, get a new config."""
        compile = self.compiled if self.compiled is not None else torch.device(device or "cpu").type == "cuda"
        cached = self.__dict__.get("_scheme", {}).get(compile)
        if cached is not None and cached[0] == asdict(self):
            return cached[1]
        q = partial(block.mxgemmini.quantize, fmt=self.fmt, axis=0, block_size=self.block_size,
                    rounding_mode=self.rounding_mode, scale_floor=self.scale_floor,
                    via=tuple(self.via) if self.via is not None else None)
        if self.reduce == "hardware":
            arith = matmul.MXGEMMINI(*self.prod, prod_floor=self.prod_floor)
            if compile:
                arith = matmul.compiled(arith)
            r = partial(matmul.systolic, arith=arith, schedule=[tuple(e) for e in self.ladder],
                        size=self.size, block_size=self.block_size)
        else:
            r = partial(fp64_accum, block_size=self.block_size)
        s = Scheme(self.name, a=q, b=q, reduce=r)
        self.__dict__.setdefault("_scheme", {})[compile] = (asdict(self), s)   # not a field: invisible to asdict
        return s

    @classmethod
    def from_dict(cls, d: dict) -> "MXQConfig":
        """Rebuild from `torchao.core.config.config_to_dict(cfg)` (or its "_data" part)."""
        data = d.get("_data", d) if isinstance(d, dict) else d
        if isinstance(d, dict) and d.get("_type", cls.__name__) != cls.__name__:
            raise ValueError(f"MXQConfig.from_dict: this dict is a {d['_type']}")
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _pair(v, what: str) -> List[int]:
    v = list(v) if isinstance(v, (list, tuple)) else v
    if not (isinstance(v, list) and len(v) == 2 and all(_is_int(x) and x > 0 for x in v)):
        raise ValueError(f"MXQConfig: {what} {v!r} must be two positive integers [e, m]")
    return v


@register_quantize_module_handler(MXQConfig)
def _to_mx(module: nn.Module, config: MXQConfig) -> nn.Module:
    """Route this Linear's forward through an MXLinear that shares its weight and bias; return the same object."""
    if not isinstance(module, nn.Linear):
        raise TypeError(f"MXQConfig applies to nn.Linear, got {type(module).__name__}")
    impl = MXLinear(module, config.scheme(module.weight.device), chunk=config.chunk)
    object.__setattr__(module, "_mxq", impl)          # not a registered child: no duplicate parameters
    module.forward = impl.forward
    return module
