"""MXLinear: one nn.Linear computed through one Scheme. Inference only.

The forward is MXQuant's `MXLinearSim.forward` step for step: flatten x to (M, K) in float32, A = xᵀ, B = Wᵀ,
Y = Aᵀ·B through the Scheme, bias added in float32, result cast back to x's dtype and reshaped. Two things
differ, and neither can change a bit of the output:

    the weight codes are computed once (quantization is a pure function of the weight), not on every call;
    the M token rows are processed `chunk` at a time. Output row m depends on input row m only: block scales
    run along K separately for each token, each product is one activation value times one weight value, and
    every rounding step is elementwise, so nothing in the datapath couples two tokens. (The K axis is coupled,
    through block boundaries and accumulator positions, and is never split.)

Chunking is also faster: past roughly three million elements per contraction step the reducer's intermediates
stop being cache friendly (measured on an L40S), so the default keeps each call under that.

`weight` and `bias` are the original layer's own Parameter objects, so state_dict keys, `.to()`, and a weight
tied to an embedding all keep working. Call `refresh()` after changing the weight.
"""
from typing import Optional

import torch
from torch import nn

from ..scheme import Scheme

__all__ = ["MXLinear"]

#: measured knee on an L40S: keep (tokens per call) x (output width) under this
ELEMENTS_PER_STEP = 3_000_000


class MXLinear(nn.Module):
    def __init__(self, linear: nn.Linear, scheme: Scheme, chunk: Optional[int] = None):
        super().__init__()
        if chunk is not None and chunk < 1:
            raise ValueError(f"chunk must be >= 1, got {chunk}")
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.weight = linear.weight                       # the same Parameter, not a copy
        self.bias = linear.bias
        self.scheme = scheme
        self.chunk = chunk
        self.register_buffer("P_W", None, persistent=False)
        self.register_buffer("X_W", None, persistent=False)
        self.refresh()

    @torch.no_grad()
    def refresh(self) -> None:
        """Recompute the cached weight codes and scales from `weight`."""
        self.P_W, self.X_W = self.scheme.b(self.weight.detach().float().t().contiguous())     # B = Wᵀ, K×N

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).float()                                               # M×K
        step = self.chunk or max(1, ELEMENTS_PER_STEP // self.out_features)
        rows = [self.scheme.reduce(*self.scheme.a(part.t().contiguous()), self.P_W, self.X_W)  # A = xᵀ, K×m
                for part in flat.split(step)]
        Y = rows[0] if len(rows) == 1 else torch.cat(rows)                                    # M×N
        if self.bias is not None:
            Y = Y + self.bias.float()
        return Y.to(x.dtype).reshape(*shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, scheme={self.scheme.name!r}"
