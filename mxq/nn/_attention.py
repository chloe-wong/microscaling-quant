"""The attention core through two Schemes: S = Q·Kᵀ through `qk`, softmax, O = P·V through `pv`.

    o = attend(q, k, v, mask, scale, qk, pv, chunk=None, vector=None)   q: B×H×Tq×D, k, v: B×Hkv×Tk×D, o: B×Tq×H×D

Per batch row and KV head, in the order MX-Gemmini runs them, with the group's n query heads side by side:

    Qᵀ (D×n·m) --qk.a-->  Kᵀ (D×Tk) --qk.b-->  S = qk.reduce   n·m×Tk fp32   contraction over D (head dims)
    S·scale + mask, softmax over keys          P               n·m×Tk fp32   host / VPU; vector="bf16": each
                                                                            step rounded to bf16 (mxq.nn._vector)
    Pᵀ (Tk×n·m) --pv.a-->  V (Tk×D) --pv.b-->  O = pv.reduce   n·m×D  fp32   contraction over the keys
    O cast to q's dtype

Every operand is quantized once, right before its matmul, blocks along the contraction: `qk.block_size` of
D, `pv.block_size` of the keys. Kᵀ and V are quantized
once per KV head and shared by its n query heads (grouped-query attention), whose m query tokens of a chunk are
the n·m columns of one call. That is the same arithmetic as one head at a time: a quantizer sees each column
alone along its blocks, every reducer stage is elementwise in the output, and softmax is a function of its row.
Columns are head-major (head i owns columns i·m to (i+1)·m), so a LUT table (2^G query tokens of one head,
qk.rows or pv.rows) never straddles two heads as long as m is a multiple of rows: chunks are multiples of rows
starting at multiples of rows, as in MXLinear, and the tokens left over at the end (Tq % rows) run one head at
a time, where the partial last table of a LUT is that head's own, as it is unchunked. So any Tq runs, and
every chunk size gives the bits of one call.

Keys the mask removes for every query of a chunk are not computed. Their score would be fl(fl(S·scale) + mask),
which is the mask value itself whenever |S·scale| is under half an ulp of it (the eager mask's dtype-min absorbs
anything below 2^103; -inf absorbs everything finite). A key is skipped only where the mask value absorbs a
bound on |S·scale| taken from the operands (Σ_k max|Q̂_k|·max|K̂_k| over the dequantized codes, times the scale
and 2^16 for the reducer's roundings), checked on the mask values in float32; so a -1e4 mask (older HF), a
finite stray value or a reducer whose scores could reach the mask's ulp all compute every key. The skipped
scores are filled with the mask values (rounded as a computed one would be under vector="bf16") and softmax
runs on the whole row, so P is the same row. P·V then
contracts over the kept keys only when P is exactly zero at the skipped ones (checked; it is not for a query
row the mask removes entirely, as HF pads, whose row is uniform): the codes and scales of Pᵀ and V are sliced
after quantization at pv.block_size boundaries, and a block of zero codes adds a tile of zeros, which every reducer's
tile_add returns unchanged (the reference model's zero short-circuit; fp64_accum's GEMM may sum in another order, so
it is checked, not assumed, in the tests). With a causal mask and two chunks that is a quarter of the key
blocks; with left padding, the padded keys of every chunk.

q/k/v/o_proj are not here: they are nn.Linear layers, and patch() gives them MXLinear. This is the function HF
calls between them (transformers' AttentionInterface, registered as "mxq" by `register`), so one function serves
every model that routes attention through that interface. patch() sets it up from a rule; see mxq.nn._patch.
"""
from typing import Optional

import torch

from ..scheme import Scheme
from ._vector import rounder, softmax

__all__ = ["attend", "register", "NAME"]

#: the attention implementation name patch() switches a model to
NAME = "mxq"

#: score entries (n·m × Tk, fp32) per call when `chunk` is None: about fourteen tensors of that size live at the
#: peak, 3.6 GiB at this bound (a TinyLlama layer at 2048 is one call of half that, 1.8 GiB); a longer sequence
#: is chunked to it rather than growing with Tq·Tk. Measured at 4096: chunks of 2048 beat 1024 by a quarter.
ELEMENTS_PER_STEP = 1 << 26

#: how far a reducer's score may exceed the exact absolute sum of its products through its roundings: a factor
#: of (1 + 2^-(m+1)) per rounding, under 2^16 for any lane width over 32 roundings and 256 block adds.
SLACK = 2.0 ** 16


def _amax(P, X, block_size):
    """The largest dequantized magnitude of each contraction row of codes P (K×n) with scales X: K float64."""
    return (P.abs() * X.repeat_interleave(block_size, dim=0)[:P.shape[0]]).amax(dim=1).double()


@torch.no_grad()
def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor], scale: float,
           qk: Scheme, pv: Scheme, chunk: Optional[int] = None, vector: Optional[str] = None) -> torch.Tensor:
    """Attention with S = Q·Kᵀ through `qk` and O = P·V through `pv`; scale, additive mask and softmax in fp32,
    or with every one of their steps rounded to bf16 when vector="bf16".

    mask: None or additive, broadcastable to B×1×Tq×Tk (0 = attend, very negative = do not), as HF's eager mask.
    chunk: query tokens per call, a positive multiple of the schemes' rows. None: the largest multiple of rows
    that keeps n·chunk·Tk under ELEMENTS_PER_STEP, i.e. all of Tq for a TinyLlama layer at 2048 (32 heads, 4 KV
    heads, tapeout scheme compiled, L40S, busy host: 0.33 s per layer and 2.1 GiB peak, against 1.43 s and 0.4 GiB
    one head at a time on the same reducer and 1.9 s on the reducer before it; T 4096: 1.03 s, 4.0 GiB; a padded
    batch of 2: 0.73 s; decoding one query over 2048 keys: 0.19 s against 1.29 s; MHA, which has no group to
    batch: 1.6 s either way)."""
    B, H, Tq, D = q.shape
    Hkv, Tk = k.shape[1], k.shape[2]
    if H % Hkv or v.shape[1] != Hkv or v.shape[2] != Tk:
        raise ValueError(f"attend: q has {H} heads, k {tuple(k.shape)}, v {tuple(v.shape)}")
    rows = max(qk.rows, pv.rows)
    if chunk is not None and (chunk < 1 or chunk % rows):
        raise ValueError(f"attend: chunk {chunk} must be a positive multiple of the schemes' rows = {rows}")
    group = H // Hkv
    bq, bv = qk.block_size, pv.block_size                                         # blocks along D, along the keys
    step = chunk or max(rows, ELEMENTS_PER_STEP // (group * Tk) // rows * rows)
    tiny = torch.finfo(torch.float32).tiny
    r = rounder(vector)                                                           # None: no rounding at all
    out = q.new_empty(B, Tq, H, D)
    for b in range(B):
        mb = None
        if mask is not None:
            mb = mask[b if mask.shape[0] > 1 else 0, 0, :, :Tk].float()
            mb = mb.expand(Tq, Tk) if mb.shape[0] == 1 else mb
        for h in range(Hkv):
            KT = qk.b(k[b, h].float().t().contiguous())                         # Kᵀ: D×Tk, blocks along D
            V = pv.b(v[b, h].float().contiguous())                              # V:  Tk×D, blocks along keys
            qk.check_scales(KT[1], D, "b (Kᵀ)")
            pv.check_scales(V[1], Tk, "b (V)")
            kmax = _amax(*KT, bq) if mb is not None else None
            heads = range(h * group, (h + 1) * group)
            for s in range(0, Tq, step):
                e = min(s + step, Tq)
                whole = s + (e - s) // rows * rows                              # all heads at once up to here
                parts = [(heads, s, whole)] if whole > s else []
                parts += [(range(hq, hq + 1), whole, e) for hq in heads] if whole < e else []
                for hs, s0, e0 in parts:
                    n, m = len(hs), e0 - s0
                    Q = q[b, hs.start:hs.stop, s0:e0].float().reshape(n * m, D)     # n·m×D, head-major
                    A = qk.a(Q.t().contiguous())
                    lo, hi = 0, Tk
                    if mb is not None:                                               # keys some query still sees
                        bound = max(SLACK * abs(scale) * float((_amax(*A, bq) * kmax).sum()), tiny)
                        mc = mb[s0:e0]
                        seen = (~((mc + bound == mc) & (mc - bound == mc)).all(dim=0)).nonzero()
                        if seen.numel():
                            lo = int(seen[0]) // bv * bv
                            hi = min(Tk, -(-(int(seen[-1]) + 1) // bv) * bv)
                        else:
                            hi = min(Tk, bv)
                    if lo == 0 and hi == Tk:
                        S = r(qk.reduce(*A, *KT) * scale)                             # n·m×Tk
                        if mb is not None:
                            S = r((S.view(n, m, Tk) + mb[s0:e0]).view(n * m, Tk))
                    else:
                        S = r(mb[s0:e0]).repeat(n, 1)                                 # skipped keys: the mask
                        S.view(n, m, Tk)[:, :, lo:hi] = r(
                            r(qk.reduce(*A, KT[0][:, lo:hi], KT[1][:, lo:hi]) * scale).view(n, m, hi - lo)
                            + mb[s0:e0, lo:hi])
                    P = softmax(S, vector)
                    if (lo or hi < Tk) and (P[:, :lo].any() or P[:, hi:].any()):
                        lo, hi = 0, Tk                                               # a row with no live key: P is uniform
                    PA = pv.a(P.t().contiguous())                                    # Pᵀ: Tk×n·m, blocks along keys
                    blk = slice(lo // bv, -(-hi // bv))                              # the kept key blocks' scales
                    O = pv.reduce(PA[0][lo:hi], PA[1][blk], V[0][lo:hi], V[1][blk])  # n·m×D
                    out[b, s0:e0, hs.start:hs.stop] = O.view(n, m, D).transpose(0, 1).to(q.dtype)
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
    return attend(query, key, value, attention_mask, scale, qk, pv,
                  vector=getattr(module, "_mxq_vector", None)), None


def register() -> None:
    """Make NAME a transformers attention implementation, with eager's additive float mask. Safe to repeat."""
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, eager_mask
    AttentionInterface.register(NAME, _mxq_attention)
    AttentionMaskInterface.register(NAME, eager_mask)
