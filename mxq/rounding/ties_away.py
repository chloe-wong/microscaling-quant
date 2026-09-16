"""Round to nearest, ties away from zero. qtorch's "nearest": add half an ulp, drop the low bits."""
import torch


def bits(bits: torch.Tensor, drop, mask, half) -> torch.Tensor:
    return (bits + half) & ~mask


def integer(k: torch.Tensor) -> torch.Tensor:
    return torch.sign(k) * torch.floor(k.abs() + 0.5)
