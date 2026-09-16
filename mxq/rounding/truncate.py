"""Truncate toward zero: drop the low bits, no rounding. MxFPMul product stage."""
import torch


def bits(bits: torch.Tensor, drop, mask, half) -> torch.Tensor:
    return bits & ~mask


def integer(k: torch.Tensor) -> torch.Tensor:
    return torch.trunc(k)
