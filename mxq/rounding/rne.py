"""Round to nearest, ties to even. IEEE 754 default; hardfloat round_near_even.
Bit form: add (half - 1 + lsb) so an exact tie rounds up only when the kept LSB is 1."""
import torch


def bits(bits: torch.Tensor, drop, mask, half) -> torch.Tensor:
    lsb = torch.bitwise_right_shift(bits, drop) & 1
    return (bits + half - 1 + lsb) & ~mask


def integer(k: torch.Tensor) -> torch.Tensor:
    return torch.round(k)          # torch.round is half-to-even
