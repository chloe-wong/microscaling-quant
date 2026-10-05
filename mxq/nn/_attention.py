"""The attention core through two Schemes: S = Q·Kᵀ through `qk`, softmax in fp32, O = P·V through `pv`.

    o = attend(q, k, v, mask, scale, qk, pv, chunk=None)       q: B×H×Tq×D, k and v: B×Hkv×Tk×D, o: B×Tq×H×D

Per batch row and query head, in the order MX-Gemmini runs them:

    Qᵀ (D×m) --qk.a-->  Kᵀ (D×Tk) --qk.b-->  S = qk.reduce   m×Tk fp32     contraction over D (head dims)
    S·scale + mask, softmax over keys        P               m×Tk fp32     host / VPU
    Pᵀ (Tk×m) --pv.a-->  V (Tk×D) --pv.b-->  O = pv.reduce   m×D  fp32     contraction over the keys
    O cast to q's dtype

Every operand is quantized once, right before its matmul, blocks along the contraction. Kᵀ and V are quantized
once per KV head and shared by its query heads (grouped-query attention); that is the same arithmetic as
repeating them, since a quantizer sees each column alone along its blocks. The m query tokens of one call are a
chunk: a multiple of max(qk.rows, pv.rows), so a LUT table (2^G query tokens) never straddles two calls, and
every chunk size gives the same bits.

q/k/v/o_proj are not here: they are nn.Linear layers, and patch() gives them MXLinear. This is the function HF
calls between them (transformers' AttentionInterface, registered as "mxq" by `register`), so one function serves
every model that routes attention through that interface. patch() sets it up from a rule; see mxq.nn._patch.
"""
from typing import Optional

import torch

from ..scheme import Scheme

__all__ = ["attend", "register", "NAME"]

#: the attention implementation name patch() switches a model to
NAME = "mxq"


@torch.no_grad()
def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor], scale: float,
           qk: Scheme, pv: Scheme, chunk: Optional[int] = None) -> torch.Tensor:
    """Attention with S = Q·Kᵀ through `qk` and O = P·V through `pv`; scale, additive mask and softmax in fp32.

    mask: None or additive, broadcastable to B×1×Tq×Tk (0 = attend, very negative = do not), as HF's eager mask.
    chunk: query tokens per call. None: all Tq in one call, the fastest measured (TinyLlama layer, 32 heads,
    T 2048, tapeout scheme compiled, L40S: 1.31 s at 2048, 2.63 s at 1464, 5.27 s at 512; peak memory 0.3 GiB)."""
    B, H, Tq, D = q.shape
    Hkv, Tk = k.shape[1], k.shape[2]
    if H % Hkv or v.shape[1] != Hkv or v.shape[2] != Tk:
        raise ValueError(f"attend: q has {H} heads, k {tuple(k.shape)}, v {tuple(v.shape)}")
    rows = max(qk.rows, pv.rows)
    step = chunk if chunk is not None else Tq
    if step < 1 or step % rows:
        raise ValueError(f"attend: chunk {step} (None: all {Tq} queries) must be a positive multiple of the schemes' "
                         f"rows = {rows}")
    group = H // Hkv
    out = q.new_empty(B, Tq, H, D)
    for b in range(B):
        mb = None if mask is None else mask[b if mask.shape[0] > 1 else 0, 0, :, :Tk]
        for h in range(Hkv):
            KT = qk.b(k[b, h].float().t().contiguous())                         # Kᵀ: D×Tk, blocks along D
            V = pv.b(v[b, h].float().contiguous())                              # V:  Tk×D, blocks along keys
            for hq in range(h * group, (h + 1) * group):
                for s in range(0, Tq, step):
                    Q = q[b, hq, s:s + step].float()                            # m×D
                    S = qk.reduce(*qk.a(Q.t().contiguous()), *KT) * scale       # m×Tk
                    if mb is not None:
                        S = S + mb[s:s + step].float()
                    P = torch.softmax(S, dim=-1)
                    O = pv.reduce(*pv.a(P.t().contiguous()), *V)                # m×D
                    out[b, s:s + step, hq] = O.to(q.dtype)
    return out


def _mxq_attention(module, query, key, value, attention_mask, dropout: float = 0.0,
                   scaling: Optional[float] = None, **kwargs):
    """transformers' attention-function signature. A module patch() gave no core runs sdpa, unchanged."""
    core = getattr(module, "_mxq_core", None)
    if core is None:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                                      **kwargs)
    if dropout:
        raise ValueError("mxq attention is inference only: dropout must be 0")
    qk, pv = core
    scale = scaling if scaling is not None else query.shape[-1] ** -0.5
    return attend(query, key, value, attention_mask, scale, qk, pv), None


def register() -> None:
    """Make NAME a transformers attention implementation, with eager's additive float mask. Safe to repeat."""
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, eager_mask
    AttentionInterface.register(NAME, _mxq_attention)
    AttentionMaskInterface.register(NAME, eager_mask)
