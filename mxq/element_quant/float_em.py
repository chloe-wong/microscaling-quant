"""Element quantization to an arbitrary float(e, m) format, bit-identical to
qtorch.quant.float_quantize(x, exp=e, man=m, rounding="nearest") on CPU and CUDA.

This is the function MXQuant used (via qtorch) for the codes of the linear-layer inputs and
for every product and accumulator quantization. Pure torch, no compiled extension.

Semantics (from qtorch's float_kernel.cu / quant_cpu.cpp, `round_bitwise_nearest` +
`clip_exponent`), operating on the IEEE-754 binary32 bit pattern:
  1. round: add half an ulp of the target mantissa width and truncate the low 23-m bits
     (round to nearest, ties away from zero; a mantissa carry into the exponent is kept)
  2. clip the biased exponent E to [127 - L, 127 + L] with L = 2^(e-1) - 1:
       E > 127 + L : saturate to +-max = (2 - 2^-m) * 2^L        (Inf and NaN also land here)
       E < 127 - L : if |q| > 2^(-L-1) then +-2^-L else 0        (no true subnormals)
Not IEEE/OCP: the top exponent is reserved, so max(e=4,m=3) is 240, not 448; the smallest
magnitude is 2^-L with full mantissa, then everything below 2^(-L-1) flushes to zero.
"""
import torch

__all__ = ["quantize", "quantizer", "max_value", "min_normal"]

_U32 = 0xFFFFFFFF


def max_value(e: int, m: int) -> float:
    L = (1 << (e - 1)) - 1
    return (2.0 - 2.0 ** -m) * 2.0 ** L


def min_normal(e: int) -> float:
    return 2.0 ** -((1 << (e - 1)) - 1)


def quantize(x: torch.Tensor, e: int, m: int, round: str = "nearest") -> torch.Tensor:
    """Quantize to float(e, m). Input is cast to float32 (as qtorch requires); output float32."""
    if round != "nearest":
        raise NotImplementedError("only rounding='nearest' (qtorch semantics) is implemented")
    if not (1 <= e <= 8 and 1 <= m <= 22):
        raise ValueError(f"unsupported widths e={e}, m={m}")

    x32 = x.detach().to(torch.float32).contiguous()
    bits = x32.view(torch.int32).to(torch.int64) & _U32

    # 1. round mantissa (unsigned 32-bit wraparound like the C kernel)
    mask = (1 << (23 - m)) - 1
    half = 1 << (22 - m)
    q = ((bits + half) & ~mask) & _U32

    # 2. clip exponent
    L = (1 << (e - 1)) - 1
    min_store, max_store = 127 - L, 127 + L
    exp_store = (q >> 23) & 0xFF
    sign = bits & 0x80000000
    max_man = 0x007FFFFF & ~mask
    max_num = sign | (max_store << 23) | max_man
    min_num = sign | (min_store << 23)
    middle = (min_store - 1) << 23
    q_mag = q & 0x7FFFFFFF

    q = torch.where(exp_store > max_store, max_num, q)
    below = exp_store < min_store
    q = torch.where(below & (q_mag > middle), min_num, q)
    q = torch.where(below & ~(q_mag > middle), torch.zeros_like(q), q)

    q = torch.where(q >= 1 << 31, q - (1 << 32), q).to(torch.int32)
    return q.view(torch.float32)


def quantizer(e: int, m: int, round: str = "nearest"):
    """Return f(x) = quantize(x, e, m)."""
    return lambda x: quantize(x, e, m, round)
