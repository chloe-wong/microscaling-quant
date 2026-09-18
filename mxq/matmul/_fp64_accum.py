"""Same codes and scales in, no rounding at the product, lane or cross-block step: scales multiplied back in,
matmul in float64, float32 out. The reducer's error floor: systolic - fp64_accum is the rounding the reducer adds."""
import torch

from ..block import BLOCK, _driver

__all__ = ["fp64_accum"]


def fp64_accum(P_A: torch.Tensor, X_A: torch.Tensor, P_B: torch.Tensor, X_B: torch.Tensor,
               block_size: int = BLOCK) -> torch.Tensor:
    A = _driver.dequantize(P_A, X_A, axis=0, block_size=block_size).to(torch.float64)
    B = _driver.dequantize(P_B, X_B, axis=0, block_size=block_size).to(torch.float64)
    return (A.t() @ B).to(torch.float32)
