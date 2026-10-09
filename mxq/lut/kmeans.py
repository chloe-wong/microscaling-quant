"""Fitting the tables and picking indices, for a K×n tensor of block codes: one table per 2^G columns.

    T = tables(P, fmt, group=G, max_iters=50)   ceil(n / 2^G) × SIZE values, each row sorted ascending
    I = pick(P, T, group=G)                     K×n: each code's nearest entry, ties to the lower index (any width)
    P_lut = lookup(I, T, group=G)

A table is fitted to the codes of its group (2^G columns, all of K; when n is not a multiple of 2^G the last
group is the columns left over, fitted by the same rule. The chip's LUT loader takes whole groups only, and
npu-exploration's emitter refuses a partial one, so this is for simulation, where a token count is arbitrary):
  1. the group's distinct codes, each weighted by how often it occurs;
  2. 16 or fewer distinct codes: they are the centres; else seed 16 centres at the quantiles of that weighted
     set, then weighted Lloyd passes (an empty cluster keeps its centre; equal centres merge) until two passes
     agree to numpy.allclose or max_iters passes have run;
  3. each centre snapped to the nearest value the finder can tell apart (formats.values);
  4. duplicates dropped, then padded to 16 with the unused values of smallest magnitude, and sorted.

This is npu-exploration compiler/codebook.py's `build_codebooks` and `assign_indices`, vectorised over groups;
equal to them bit for bit (npu-exploration tests/selftest_codebook_mxq.py). Every sum it takes is exact (codes
times counts have few significant bits), so the order of summation, and so the device, does not matter.
"""
import torch

from .formats import SIZE, _format, _tables

__all__ = ["tables", "pick", "lookup"]

_RTOL, _ATOL = 1e-5, 1e-8              # numpy.allclose's defaults: the reference's convergence test


def _check(P: torch.Tensor, group: int) -> int:
    if P.ndim != 2:
        raise ValueError(f"mxq.lut: expects a K×n tensor, got shape {tuple(P.shape)}")
    if not isinstance(group, int) or isinstance(group, bool) or group < 0:
        raise ValueError(f"mxq.lut: group {group!r} must be a non-negative integer")
    return (P.shape[1] + (1 << group) - 1) >> group                             # a partial last group counts


def _dedupe(C: torch.Tensor) -> torch.Tensor:
    """Each row sorted ascending, equal values merged, +inf after the last centre."""
    C, _ = C.sort(dim=1)
    dup = torch.zeros_like(C, dtype=torch.bool)
    dup[:, 1:] = C[:, 1:] == C[:, :-1]
    return C.masked_fill(dup, float("inf")).sort(dim=1)[0]


def tables(P: torch.Tensor, fmt, *, group: int, max_iters: int) -> torch.Tensor:
    """One SIZE-entry table per 2^group columns of P (K×n block codes, every one a value of fmt); the last
    table covers the columns left over when n is not a multiple of 2^group."""
    f = _format(fmt)
    ng = _check(P, group)
    if not isinstance(max_iters, int) or isinstance(max_iters, bool) or max_iters < 0:
        raise ValueError(f"mxq.lut: max_iters {max_iters!r} must be a non-negative integer")
    dev = P.device
    all_vals, _, safe = (t.to(dev) for t in _tables(f.name))
    grid = torch.unique(all_vals[torch.isfinite(all_vals)])                     # sorted distinct values
    nv = grid.numel()

    # 1. counts of each distinct code per group
    P = P.to(torch.float32)
    b = torch.searchsorted(grid, P.contiguous())
    if (b >= nv).any() or not torch.equal(grid[b.clamp(max=nv - 1)], P):
        raise ValueError(f"mxq.lut: P holds a value that is not a {f.name} code; quantize it first")
    gid = (torch.arange(P.shape[1], device=dev) >> group).expand_as(b)
    counts = torch.bincount((gid * nv + b).reshape(-1), minlength=ng * nv).reshape(ng, nv)
    present = counts > 0
    npres = present.sum(dim=1)
    x = grid.double()

    # 2a. 16 or fewer distinct codes: those are the centres
    C = torch.full((ng, SIZE), float("inf"), dtype=torch.float64, device=dev)
    few = npres <= SIZE
    if few.any():
        pc = present[few]
        rank = pc.long().cumsum(dim=1) - 1
        rows = torch.arange(pc.shape[0], device=dev).unsqueeze(1).expand_as(pc)
        Cf = torch.full((pc.shape[0], SIZE), float("inf"), dtype=torch.float64, device=dev)
        Cf[rows[pc], rank[pc]] = x.expand_as(pc)[pc]
        C[few] = Cf

    # 2b. quantile seeds: the first code whose cumulative weight reaches (i + 0.5) / 16
    many = ~few
    if many.any():
        cnt = counts[many]
        cdf = cnt.cumsum(dim=1).double() / cnt.sum(dim=1, keepdim=True).double()
        probes = ((torch.arange(SIZE, device=dev, dtype=torch.float64) + 0.5) / SIZE).expand(cnt.shape[0], SIZE)
        seeds = torch.searchsorted(cdf.contiguous(), probes.contiguous()).clamp(max=nv - 1)
        C[many] = _dedupe(x[seeds])

        # 2c. weighted Lloyd passes over the distinct codes
        Cm, w = C[many], cnt.double()
        live = torch.ones(Cm.shape[0], dtype=torch.bool, device=dev)
        for _ in range(max_iters):
            if not live.any():
                break
            Cl, wl = Cm[live], w[live]
            lab = (x.view(1, -1, 1) - Cl.unsqueeze(1)).abs().argmin(dim=2)             # nearest centre, ties low
            sums = torch.zeros_like(Cl).scatter_add_(1, lab, wl * x)
            wsum = torch.zeros_like(Cl).scatter_add_(1, lab, wl)
            new = _dedupe(torch.where(wsum > 0, sums / wsum, Cl))                       # empty: keeps its centre
            valid = torch.isfinite(Cl)
            same = (torch.isfinite(new) == valid).all(dim=1) & \
                (((new - Cl).abs() <= _ATOL + _RTOL * Cl.abs()) | ~valid).all(dim=1)
            idx = live.nonzero().squeeze(1)
            Cm[idx[~same]] = new[~same]                                                 # converged rows keep theirs
            live[idx[same]] = False
        C[many] = Cm

    # 3. snap each centre to the nearest finder-safe value (ties to the smaller)
    s = safe.double()
    valid = torch.isfinite(C)
    near = (C.unsqueeze(2) - s.view(1, 1, -1)).abs().argmin(dim=2)
    member = torch.zeros(ng, s.numel(), dtype=torch.bool, device=dev)
    member[torch.arange(ng, device=dev).unsqueeze(1).expand_as(near)[valid], near[valid]] = True

    # 4. pad with unused values, smallest magnitude first (negative before positive), then sort
    order = sorted(range(s.numel()), key=lambda i: abs(safe[i].item()))
    order = torch.tensor(order, device=dev)
    free = ~member[:, order]
    take = free & (free.long().cumsum(dim=1) <= (SIZE - member.sum(dim=1, keepdim=True)))
    member[:, order] |= take
    T = torch.where(member, safe.unsqueeze(0), torch.tensor(float("inf"), device=dev)).sort(dim=1)[0][:, :SIZE]
    return T.contiguous()


def pick(P: torch.Tensor, T: torch.Tensor, *, group: int) -> torch.Tensor:
    """Each code's nearest entry in its column group's table, ties to the lower index. K×n int64.
    Any table width of two or more (one row per group, ascending); the chip's is SIZE."""
    ng = _check(P, group)
    _check_tables(T, ng, P.shape[1], group, "pick")
    size = T.shape[1]
    rows = T.to(P.device, torch.float32)[torch.arange(P.shape[1], device=P.device) >> group]   # n×size, ascending
    v = P.to(torch.float32).t().contiguous()                                                   # n×K
    # the nearest of the sorted entries is one of the two around v; every difference here is exact in float32
    hi = torch.searchsorted(rows, v).clamp(1, size - 1)
    lo = hi - 1
    d_lo = (v - rows.gather(1, lo)).abs()
    d_hi = (rows.gather(1, hi) - v).abs()
    return torch.where(d_lo <= d_hi, lo, hi).t().contiguous()


def lookup(I: torch.Tensor, T: torch.Tensor, *, group: int) -> torch.Tensor:
    """The values the indices name: K×n float32. Any table width (one row per group)."""
    ng = _check(I, group)
    _check_tables(T, ng, I.shape[1], group, "lookup")
    rows = T.to(I.device, torch.float32)[torch.arange(I.shape[1], device=I.device) >> group]   # n×width
    return rows.gather(1, I.t().contiguous()).t().contiguous()


def _check_tables(T: torch.Tensor, ng: int, n: int, group: int, who: str) -> None:
    if T.ndim != 2 or T.shape[0] != ng or T.shape[1] < 2:
        raise ValueError(f"mxq.lut.{who}: {n} columns at group {group} need ({ng}, width) tables with width >= 2, "
                         f"got {tuple(T.shape)}")
