"""The raw-value LUT: tables fitted on, and indices picked from, the scaled values themselves rather than their
element codes. A proposed MX-Gemmini change, NOT the chip today and NOT any default here; `tables`, `pick` and
`finder` (kmeans.py, formats.py) remain the chip's rule.

    R = V / expand(X)                         the raw values: the operand over its block scale (exact: X is a power
                                              of two). Under block.mxgemmini's scale the block max is in [1, 2)
    T = tables_raw(R, fmt, group=G, max_iters=50)   the tables rule of kmeans.py on the raw values
    I = pick_raw(R, T, group=G)               host pick: each value's nearest entry, ties to the lower index
    I = finder_raw(R, T, fmt, group=G)        the proposed finder for requantized outputs (MXFP6_E3M2 only), as RTL
                                              arithmetic: equal to pick_raw wherever it is defined

Why: rounding to a code before fitting and picking (the chip today) rounds twice. On TinyLlama (MXFP6_E3M2, G = 1,
MLP and lm_head, bf16 matmuls) fitting on raw values and picking from them takes perplexity from 9.61 to 8.29, with
activation tables calibrated in advance (bf16 7.20). Fitting on raw values helps only with the raw pick; the
measurements and the hardware requirement are in npu-exploration microscaling-quant/experiments/lut_fit_ppl.py.

tables_raw is tables' rule (quantile seeds, weighted Lloyd passes until numpy.allclose or max_iters, empty clusters
keep their centre, equal centres merge, snap to formats.values, pad with the unused values of smallest magnitude)
on every element of a group, each counting 1; fed codes instead of raw values it equals `tables` bit for bit.

pick_raw decides by the midpoint between neighbouring entries, exactly: entries are element codes (dyadic), so
(a + b) / 2 is exact in float64, and so is the comparison. `pick` compares float32 distances instead, which is exact
for codes but not for raw values: below about 4e-9 between a negative and a positive entry both distances round to
the same number and it calls a false tie.

finder_raw (MXFP6_E3M2, every |R| < 2, every entry within [-2, 2]): each lane forms the 8-bit key
    k = 2 * floor(32 * raw) + sticky,   sticky = 1 when 32 * raw is not an integer      k in [-128, 127]
and counts the per-table thresholds t = 32 * (a + b) + 1 (one per adjacent pair of the sorted entries, t in
[-119, 121]) with t <= k; the stored sort permutation maps that position back to the table's own index, as
gemmini NearestFinder.scala does for codes. Every midpoint of two such entries is a multiple of 2^-5, so the key
decides every comparison exactly; an exact tie (raw on a midpoint) goes to the lower entry, the chip's tie rule.
"""
import torch

from .formats import SIZE, _format, _tables
from .kmeans import _RTOL, _ATOL, _check, _dedupe

__all__ = ["tables_raw", "pick_raw", "finder_raw"]

_GROUP_CHUNK = 2048                    # table groups fitted at once (memory: chunk × K·2^G float64 points)


def _fit(v: torch.Tensor, safe: torch.Tensor, max_iters: int) -> torch.Tensor:
    """One table per row of v (ng × P float64, sorted ascending), every point counting 1. ng × SIZE, ascending."""
    ng, npts, dev = v.shape[0], v.shape[1], v.device
    first = torch.ones_like(v, dtype=torch.bool)
    first[:, 1:] = v[:, 1:] != v[:, :-1]
    ndist = first.sum(dim=1)

    C = torch.full((ng, SIZE), float("inf"), dtype=torch.float64, device=dev)
    few = ndist <= SIZE                                                     # 16 or fewer: those are the centres
    if few.any():
        f = first[few]
        rank = f.long().cumsum(dim=1) - 1
        rows = torch.arange(f.shape[0], device=dev).unsqueeze(1).expand_as(f)
        Cf = torch.full((f.shape[0], SIZE), float("inf"), dtype=torch.float64, device=dev)
        Cf[rows[f], rank[f]] = v[few][f]
        C[few] = Cf

    many = ~few
    if many.any():
        vm = v[many]
        cdf = torch.arange(1, npts + 1, device=dev, dtype=torch.float64) / npts
        probes = (torch.arange(SIZE, device=dev, dtype=torch.float64) + 0.5) / SIZE
        seeds = torch.searchsorted(cdf, probes).clamp(max=npts - 1)                 # quantile seeds
        Cm = _dedupe(vm[:, seeds])
        live = torch.ones(Cm.shape[0], dtype=torch.bool, device=dev)
        for _ in range(max_iters):
            if not live.any():
                break
            idx = live.nonzero().squeeze(1)
            Cl, vl = Cm[idx], vm[idx]
            # nearest centre, ties to the lower, decided as kmeans.tables does: by the two rounded distances
            hi = torch.searchsorted(Cl.contiguous(), vl.contiguous()).clamp(max=SIZE - 1)
            lo = (hi - 1).clamp(min=0)
            lab = torch.where((vl - Cl.gather(1, lo)).abs() <= (vl - Cl.gather(1, hi)).abs(), lo, hi)
            sums = torch.zeros_like(Cl).scatter_add_(1, lab, vl)
            wsum = torch.zeros_like(Cl).scatter_add_(1, lab, torch.ones_like(vl))
            new = _dedupe(torch.where(wsum > 0, sums / wsum, Cl))                   # empty: keeps its centre
            valid = torch.isfinite(Cl)
            same = (torch.isfinite(new) == valid).all(dim=1) & \
                (((new - Cl).abs() <= _ATOL + _RTOL * Cl.abs()) | ~valid).all(dim=1)
            Cm[idx[~same]] = new[~same]
            live[idx[same]] = False
        C[many] = Cm

    valid = torch.isfinite(C)
    near = (C.unsqueeze(2) - safe.view(1, 1, -1)).abs().argmin(dim=2)            # ties: the smaller value
    member = torch.zeros(ng, safe.numel(), dtype=torch.bool, device=dev)
    member[torch.arange(ng, device=dev).unsqueeze(1).expand_as(near)[valid], near[valid]] = True
    order = safe.abs().sort(stable=True)[1]                                      # smallest magnitude, - before +
    free = ~member[:, order]
    take = free & (free.long().cumsum(dim=1) <= (SIZE - member.sum(dim=1, keepdim=True)))
    member[:, order] |= take
    T = torch.where(member, safe.unsqueeze(0), torch.tensor(float("inf"), device=dev, dtype=torch.float64))
    return T.sort(dim=1)[0][:, :SIZE]


def tables_raw(R: torch.Tensor, fmt, *, group: int, max_iters: int) -> torch.Tensor:
    """One SIZE-entry table per 2^group columns of R (K×n raw values), fitted on the values themselves; the last
    table covers the columns left over when n is not a multiple of 2^group. Returns (ceil(n / 2^group)) × SIZE
    float32, rows ascending, every entry a finder-safe value of fmt."""
    f = _format(fmt)
    _check(R, group)
    if not isinstance(max_iters, int) or isinstance(max_iters, bool) or max_iters < 0:
        raise ValueError(f"mxq.lut: max_iters {max_iters!r} must be a non-negative integer")
    if not torch.isfinite(R).all():
        raise ValueError("mxq.lut.tables_raw: R holds a non-finite value")
    safe = _tables(f.name)[2].to(R.device, torch.float64)
    K, n = R.shape
    N = 1 << group
    full = n - n % N
    rows = [R[:, :full].t().reshape(full // N, N * K)] if full else []          # each group's elements as a row
    if full < n:
        rows.append(R[:, full:].t().reshape(1, -1))
    out = []
    for Z in rows:
        for a in range(0, Z.shape[0], _GROUP_CHUNK):
            out.append(_fit(Z[a:a + _GROUP_CHUNK].double().sort(dim=1)[0], safe, max_iters))
    return torch.cat(out).float().contiguous()


def pick_raw(R: torch.Tensor, T: torch.Tensor, *, group: int) -> torch.Tensor:
    """Each raw value's nearest entry in its column group's table, ties to the lower index, decided exactly by
    the midpoints. R K×n, T ceil(n / 2^group) × SIZE ascending. Returns K×n int64."""
    ng = _check(R, group)
    if T.shape != (ng, SIZE):
        raise ValueError(f"mxq.lut.pick_raw: {R.shape[1]} columns at group {group} need ({ng}, {SIZE}) tables, "
                         f"got {tuple(T.shape)}")
    rows = T.to(R.device, torch.float64)[torch.arange(R.shape[1], device=R.device) >> group]   # n × SIZE
    mid = (rows[:, :-1] + rows[:, 1:]) / 2                                                     # exact: dyadic
    v = R.to(torch.float64).t().contiguous()                                                   # n × K
    return torch.searchsorted(mid.contiguous(), v, side="left").t().contiguous()             # # midpoints < v


def finder_raw(R: torch.Tensor, T: torch.Tensor, fmt, *, group: int) -> torch.Tensor:
    """The proposed finder, as its RTL arithmetic: an 8-bit key per raw value against 15 per-table thresholds.
    MXFP6_E3M2 only; every |R| < 2 (block.mxgemmini's scale) and every entry within [-2, 2]. Returns K×n int64,
    the table's own index (the sorted position mapped back through the sort, ties among equal entries to the
    lower index)."""
    f = _format(fmt)
    if f.name != "MXFP6_E3M2":
        raise ValueError(f"mxq.lut.finder_raw: modelled for MXFP6_E3M2 only, got {f.name}")
    ng = _check(R, group)
    if T.shape != (ng, SIZE):
        raise ValueError(f"mxq.lut.finder_raw: {R.shape[1]} columns at group {group} need ({ng}, {SIZE}) tables, "
                         f"got {tuple(T.shape)}")
    if not (R.abs() < 2).all():
        raise ValueError("mxq.lut.finder_raw: every |raw| must be < 2 (the requantizer's block max in [1, 2))")
    if not (T.abs() <= 2).all():
        raise ValueError("mxq.lut.finder_raw: every table entry must be within [-2, 2]")
    Ts, perm = T.to(R.device, torch.float64).sort(dim=1, stable=True)
    thr = (32 * (Ts[:, :-1] + Ts[:, 1:]) + 1).round().long()                    # ng × 15, integers in [-119, 121]
    x32 = R.to(torch.float64) * 32                                              # exact: a power-of-two shift
    F = torch.floor(x32)
    k = 2 * F.long() + (x32 != F).long()                                         # K × n, in [-128, 127]
    cols = torch.arange(R.shape[1], device=R.device) >> group
    pos = (k.t().unsqueeze(-1) >= thr[cols].unsqueeze(1)).sum(-1)                # n × K, the thermometer
    return perm[cols].gather(1, pos).t().contiguous()
