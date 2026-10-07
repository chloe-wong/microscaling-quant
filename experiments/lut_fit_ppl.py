#!/usr/bin/env python3
"""TinyLlama WikiText-2 perplexity of MXFP6_E3M2 LUT operands at the chip's range: which part of MXQuant's table
fitting matters (experiment 1), and whether it survives tables fixed in advance, as the chip's finder needs them for
chained outputs (experiment 2).

Every operand: the chip's block scale (block.mxgemmini: block max in [1, 2), floor 2^-23), so every value and
table entry is within [-2, 2] (finder-safe), then a 16-entry table per 2^G rows of A / columns of B spanning all of
K, fitted by ONE deterministic rule (mxq's: quantile seeds, weighted Lloyd passes until two agree or 50, empty
clusters keep their centre, equal centres merge, centres snapped to the nearest E3M2 value, padded with the unused
values of smallest magnitude). Three switches:

    fit    code   cluster the FP6 codes (RNE of the scaled value)                    mxq
           raw    cluster the scaled bf16 values themselves                           MXQuant
    weight cnt    every element counts 1                                              mxq
           s2     every element counts its block scale^2                              MXQuant
    pick   code   the entry nearest the FP6 code   (= the chip's finder today)        mxq
           raw    the entry nearest the scaled bf16 value (= the proposed finder)     MXQuant

Experiment 1, tables fitted on the data they quantize (host-side for every operand):
    fit_<fit>_<weight>_pick_<pick>_g<G>         fit_code_cnt_pick_code == mxq_g<G> == block.lut, bit for bit

Experiment 2, the activation (A) tables fitted ONCE on WikiText-2 train windows through the bf16 model, then fixed;
the weight (B) operand exactly as in its experiment-1 twin, so each pair differs only in where A's tables come from:
    cal_pos_<fit>_<weight>_pick_<pick>_g<G>     one table per 2^G token positions (the chip's layout, row >> G)
    cal_shared_<fit>_<weight>_pick_<pick>_g<G>  one table for every position of the layer
Calibration keeps histograms, not values: codes exactly; raw values in bins of 2^-8 (1/16 of the finest E3M2 step),
fitted at the bin centres. Every activation is treated as a chained (finder-picked) output.

Every matmul is a bf16 matmul of the dequantized operands (exact in bf16), as nn.Linear: the quantization in
isolation. Layers: --layers mlp (default) = every nn.Linear but the attention projections, lm_head quantized.

    python experiments/lut_fit_ppl.py --check                                  # CPU, no model: rule == block.lut
    python experiments/lut_fit_ppl.py --check-gpu [cuda:N]                     # every piece on the GPU == CPU
    python experiments/lut_fit_ppl.py --sweep --exp ablation --gpus 0,1,2,3    # experiment 1, 8 configs
    python experiments/lut_fit_ppl.py --sweep --exp calibrated --gpus 0,1,2,3  # experiment 2: calibrates, 8 configs
    python experiments/lut_fit_ppl.py --sweep --exp both --gpus 0,1,2,3 --groups 0,1
    python experiments/lut_fit_ppl.py --sweep --configs cal_pos_raw_s2_pick_raw_g1 --gpus 0

Experiment 2 options (env or sweep flags): --cal-nsamples (16 train windows), --cal-seed (0), --cal-fits
(code_cnt,raw_s2). The calibration is cached in experiments/results/lutcal_g<G>_s<seqlen>_n<N>_seed<S>.pt and
reused. Results: experiments/results/lutfit_<config>_<layers>.json, and a table at the end of --sweep.
"""
import json, os, re, subprocess, sys
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = Path(os.environ.get("MXQ_REPO", HERE.parent.parent))
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from mxq import Scheme, block, lut, scale_factor  # noqa: E402
from mxq.block._driver import dequantize  # noqa: E402
from mxq.nn import is_attention  # noqa: E402
from experiments import llm_ppl, recipes  # noqa: E402

FMT = "MXFP6_E3M2"
SIZE = lut.SIZE
MAX_ITERS = 50
COMMON = dict(axis=0, block_size=32, rounding_mode="rne", scale_floor=scale_factor.HARDWARE_FLOOR)
_RTOL, _ATOL = 1e-5, 1e-8                                 # numpy.allclose's defaults, as mxq.lut
GROUP_CHUNK = 2048                                        # table groups fitted at once (memory)
RAW_BIN = 2.0 ** -8                                       # calibration histogram bin for raw values
N_LAYERS = int(os.environ.get("LUTFIT_LAYERS", "22"))     # TinyLlama-1.1B
RESULTS = Path(os.environ.get("LUTFIT_RESULTS", REPO / "experiments" / "results"))   # calibration cache + sweep table


def _env_ints(name, default):
    s = os.environ.get(name, default)
    return tuple(int(v) for v in s.split(",")) if s else ()


GROUPS = _env_ints("LUTFIT_GROUPS", "1")
SEQLEN = int(os.environ.get("LUTFIT_SEQLEN", "2048"))
CAL_N = int(os.environ.get("LUTFIT_CAL_NSAMPLES", "16"))
CAL_SEED = int(os.environ.get("LUTFIT_CAL_SEED", "0"))
CAL_FITS = tuple(os.environ.get("LUTFIT_CAL_FITS", "code_cnt,raw_s2").split(","))
MODEL_ID = os.environ.get("LUTFIT_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")


# ------------------------------------------------------------------------------------------- values, the rule

def _safe(device):
    """The values a table may hold at the chip's range: finder-safe E3M2 values with |v| <= 2, ascending."""
    s = lut.values(FMT)
    return s[s.abs() <= 2].to(device=device, dtype=torch.float64)


def _dedupe(C):
    C, _ = C.sort(dim=1)
    dup = torch.zeros_like(C, dtype=torch.bool)
    dup[:, 1:] = C[:, 1:] == C[:, :-1]
    return C.masked_fill(dup, float("inf")).sort(dim=1)[0]


def _assign(v, C):
    """Index of the nearest of the sorted centres C (inf-padded) for each v, ties to the lower index. Decided as
    mxq.lut does, by the two rounded distances |v - c| (argmin, first minimum), not by a rounded midpoint: the two
    differ on exact ties between centres that are not dyadic."""
    hi = torch.searchsorted(C.contiguous(), v.contiguous(), side="left").clamp(max=C.shape[1] - 1)
    lo = (hi - 1).clamp(min=0)
    d_lo = (v - C.gather(1, lo)).abs()
    d_hi = (v - C.gather(1, hi)).abs()
    return torch.where(d_lo <= d_hi, lo, hi)


def _fit_rows(v, w):
    """One table per row from weighted points. v, w: ng × P float64, v sorted ascending per row, w >= 0.
    mxq.lut.tables's rule, on weighted points rather than counted codes. Returns ng × SIZE float64, ascending."""
    ng, dev = v.shape[0], v.device
    present = w > 0
    first = present.clone()
    # a point is a new distinct value if it differs from the previous present point (rows are sorted)
    vp = torch.where(present, v, torch.full_like(v, float("nan")))
    prev = torch.cummax(torch.where(present, torch.arange(v.shape[1], device=dev).expand_as(v),
                                    torch.full_like(v, -1, dtype=torch.long)), dim=1)[0]
    prev = torch.cat([torch.full((ng, 1), -1, device=dev, dtype=torch.long), prev[:, :-1]], dim=1)
    prev_v = torch.where(prev >= 0, v.gather(1, prev.clamp(min=0)), torch.full_like(v, float("nan")))
    first &= ~(vp == prev_v)
    ndist = first.sum(dim=1)

    C = torch.full((ng, SIZE), float("inf"), dtype=torch.float64, device=dev)
    few = ndist <= SIZE
    if few.any():
        f = first[few]
        rank = f.long().cumsum(dim=1) - 1
        rows = torch.arange(f.shape[0], device=dev).unsqueeze(1).expand_as(f)
        Cf = torch.full((f.shape[0], SIZE), float("inf"), dtype=torch.float64, device=dev)
        Cf[rows[f], rank[f]] = v[few][f]
        C[few] = Cf

    many = ~few
    if many.any():
        vm, wm = v[many], w[many]
        cdf = wm.cumsum(dim=1) / wm.sum(dim=1, keepdim=True)
        probes = ((torch.arange(SIZE, device=dev, dtype=torch.float64) + 0.5) / SIZE).expand(vm.shape[0], SIZE)
        seeds = torch.searchsorted(cdf.contiguous(), probes.contiguous()).clamp(max=vm.shape[1] - 1)
        Cm = _dedupe(vm.gather(1, seeds))
        live = torch.ones(Cm.shape[0], dtype=torch.bool, device=dev)
        for _ in range(MAX_ITERS):
            if not live.any():
                break
            idx = live.nonzero().squeeze(1)
            Cl, vl, wl = Cm[idx], vm[idx], wm[idx]
            lab = _assign(vl, Cl)
            sums = torch.zeros_like(Cl).scatter_add_(1, lab, wl * vl)
            wsum = torch.zeros_like(Cl).scatter_add_(1, lab, wl)
            new = _dedupe(torch.where(wsum > 0, sums / wsum, Cl))
            valid = torch.isfinite(Cl)
            same = (torch.isfinite(new) == valid).all(dim=1) & \
                (((new - Cl).abs() <= _ATOL + _RTOL * Cl.abs()) | ~valid).all(dim=1)
            Cm[idx[~same]] = new[~same]
            live[idx[same]] = False
        C[many] = Cm

    s = _safe(dev)
    valid = torch.isfinite(C)
    near = (C.unsqueeze(2) - s.view(1, 1, -1)).abs().argmin(dim=2)                 # ties: the smaller value
    member = torch.zeros(ng, s.numel(), dtype=torch.bool, device=dev)
    member[torch.arange(ng, device=dev).unsqueeze(1).expand_as(near)[valid], near[valid]] = True
    order = s.abs().sort(stable=True)[1]                  # smallest magnitude first, negative before positive
    free = ~member[:, order]
    take = free & (free.long().cumsum(dim=1) <= (SIZE - member.sum(dim=1, keepdim=True)))
    member[:, order] |= take
    return torch.where(member, s.unsqueeze(0), torch.tensor(float("inf"), device=dev, dtype=torch.float64)) \
        .sort(dim=1)[0][:, :SIZE].contiguous()


def fit_points(v, w):
    """Tables for ng groups of weighted points (ng × P each, any order), in chunks of groups."""
    out = []
    for a in range(0, v.shape[0], GROUP_CHUNK):
        vs, order = v[a:a + GROUP_CHUNK].double().sort(dim=1)
        out.append(_fit_rows(vs, w[a:a + GROUP_CHUNK].double().gather(1, order)))
    return torch.cat(out)


def _operand(V):
    """The chip's codes P, scales X ((K/32)×n), and the scaled values R = V / X (exact: X is a power of two)."""
    P, X = block.mxgemmini.quantize(V, FMT, **COMMON)
    Xe = X.repeat_interleave(32, dim=0)[:V.shape[0]]
    return P, X, V.float() / Xe, Xe


def _grouped(Z, group):
    """K×n -> (n >> group) × (K * 2^group): each table group's elements as one row."""
    K, n = Z.shape
    return Z.t().reshape(n >> group, (1 << group) * K)


def tables_from_data(P, R, Xe, fit, weight, group):
    if P.shape[1] % (1 << group):
        raise ValueError(f"lut_fit: {P.shape[1]} columns are not whole groups of 2^{group}")
    v = _grouped(P if fit == "code" else R, group)
    w = torch.ones_like(v) if weight == "cnt" else _grouped(Xe * Xe, group)
    return fit_points(v, w)


TIES = ("low", "even", "zero")


def nearest(Z, T, group, tie="low"):
    """Each element's nearest entry in its column group's table, as values. K×n. An exact tie between two
    neighbouring entries goes to: low  the lower index (the chip's finder today; lut.pick, bit for bit)
                                  even the even index
                                  zero the entry of smaller magnitude
    Every difference here is exact in float32 (values within [-2, 2], entries on the E3M2 grid)."""
    T32 = T.to(torch.float32)
    if tie == "low":
        return lut.lookup(lut.pick(Z.float(), T32, group=group), T32, group=group)
    rows = T32[torch.arange(Z.shape[1], device=Z.device) >> group]                    # n × SIZE, ascending
    v = Z.float().t().contiguous()                                                      # n × K
    hi = torch.searchsorted(rows, v).clamp(1, SIZE - 1)
    lo = hi - 1
    a, b = rows.gather(1, lo), rows.gather(1, hi)
    d_lo, d_hi = (v - a).abs(), (b - v).abs()
    tie_lo = (lo % 2 == 0) if tie == "even" else (a.abs() <= b.abs())
    return torch.where((d_lo < d_hi) | ((d_lo == d_hi) & tie_lo), a, b).t().contiguous()


# -------------------------------------------------------------------------------------------- calibration

_CAL = {}                                               # cache path -> loaded histograms


def cal_path(group):
    return RESULTS / f"lutcal_g{group}_s{SEQLEN}_n{CAL_N}_seed{CAL_SEED}.pt"


def target_names():
    return [f"model.layers.{i}.mlp.{p}" for i in range(N_LAYERS) for p in ("gate_proj", "up_proj", "down_proj")] + \
        ["lm_head"]


def _codes_axis(device):
    return _safe(device).float()                        # every E3M2 code at the chip's range is a safe value


def _raw_axis(device):
    nb = int(4 / RAW_BIN)
    return (-2 + (torch.arange(nb, device=device, dtype=torch.float64) + 0.5) * RAW_BIN)


def new_histograms(npos, device):
    return {k: torch.zeros(npos, (_codes_axis(device) if k.startswith("code") else _raw_axis(device)).numel(),
                           dtype=torch.float64, device=device) for k in ("code_cnt", "code_s2", "raw_cnt", "raw_s2")}


def accumulate(H, x, group):
    """Add one window's layer input x (S × K, token s at position s) to a layer's histograms H, in place."""
    dev = x.device
    codes, raw = _codes_axis(dev), _raw_axis(dev)
    P, _, R, Xe = _operand(x.t().float().contiguous())                         # K × S
    pos = (torch.arange(P.shape[1], device=dev) >> group).expand_as(P)
    ci = torch.searchsorted(codes, P.contiguous())
    ri = ((R + 2) / RAW_BIN).floor().long().clamp(0, raw.numel() - 1)
    w2 = Xe.double() ** 2
    for key, idx, nb in (("code", ci, codes.numel()), ("raw", ri, raw.numel())):
        flat = (pos * nb + idx).reshape(-1)
        H[f"{key}_cnt"].view(-1).index_add_(0, flat, torch.ones_like(flat, dtype=torch.float64))
        H[f"{key}_s2"].view(-1).index_add_(0, flat, w2.reshape(-1))


def calibrate(group):
    """Histograms of every target layer's input over CAL_N train windows, per position group: codes and raw-value
    bins, each counted and scale^2-weighted. Saved to cal_path(group)."""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    torch.manual_seed(0)
    model = llm_ppl.load_model(MODEL_ID, SEQLEN)
    dev = next(model.parameters()).device
    names = target_names()
    mods = dict(model.named_modules())
    missing = [n for n in names if n not in mods]
    extra = [n for n, m in mods.items() if isinstance(m, nn.Linear) and n not in names and ".self_attn." not in n]
    if missing or extra:
        raise SystemExit(f"lut_fit: model layers differ from the target list: missing {missing[:3]}, extra {extra[:3]}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    ids = tok("\n\n".join([x for x in data["text"] if x.strip()]), return_tensors="pt")["input_ids"][0]
    nwin = ids.numel() // SEQLEN
    ids = ids[:nwin * SEQLEN].reshape(nwin, SEQLEN)
    g = torch.Generator().manual_seed(CAL_SEED)
    ids = ids[torch.randperm(nwin, generator=g)[:CAL_N]]

    H = {n: new_histograms(SEQLEN >> group, dev) for n in names}

    def hook(name):
        def f(module, inputs):
            accumulate(H[name], inputs[0].reshape(-1, inputs[0].shape[-1]), group)   # S × K, one window
        return f

    handles = [mods[n].register_forward_pre_hook(hook(n)) for n in names]
    with torch.no_grad():
        for i in range(ids.shape[0]):
            model(ids[i:i + 1].to(dev), use_cache=False)
            print(f"  calibration window {i + 1}/{ids.shape[0]}", flush=True)
    for h in handles:
        h.remove()
    out = {"group": group, "seqlen": SEQLEN, "nsamples": CAL_N, "seed": CAL_SEED, "model": MODEL_ID,
           "raw_bin": RAW_BIN, "hist": {n: {k: t.cpu() for k, t in d.items()} for n, d in H.items()}}
    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.save(out, cal_path(group))
    print(f"calibration saved: {cal_path(group)}")


def cal_tables(layer, src, fit, weight, group, device):
    """The fixed A tables of a layer: (seqlen >> group) × SIZE for 'pos', 1 × SIZE for 'shared'."""
    path = cal_path(group)
    key = (str(path), layer, src, fit, weight, str(torch.device(device)))   # per device: patch's smoke test runs on cpu
    if key not in _CAL:
        if str(path) not in _CAL:
            if not path.exists():
                raise SystemExit(f"lut_fit: no calibration at {path}; run --calibrate (or --sweep, which does)")
            _CAL[str(path)] = torch.load(path)
        h = _CAL[str(path)]["hist"][layer][f"{fit}_{weight}"].to(device)
        axis = (_codes_axis(device) if fit == "code" else _raw_axis(device)).double()
        if src == "shared":
            h = h.sum(dim=0, keepdim=True)
        _CAL[key] = fit_points(axis.expand(h.shape[0], -1), h)
    return _CAL[key]


# ------------------------------------------------------------------------------------------------ quantizers

def lut_data(V, *, fit, weight, pick, group, tie="low"):
    """Experiment 1: tables fitted on V itself. V K×n -> (table values K×n, scales)."""
    P, X, R, Xe = _operand(V)
    T = tables_from_data(P, R, Xe, fit, weight, group)
    return nearest(P if pick == "code" else R, T, group, tie), X


def lut_cal(V, *, layer, src, fit, weight, pick, group, tie="low"):
    """Experiment 2, activations: tables fixed by the calibration; column j is token position j % seqlen."""
    P, X, R, _ = _operand(V)
    n = P.shape[1]
    smoke = P.shape[0] == 32 and n < SEQLEN          # mxq.nn.patch's smoke matmul: one 32-block, 2 groups
    if src == "pos" and n % SEQLEN and not smoke:
        raise ValueError(f"lut_fit: cal_pos needs whole windows per call ({n} columns, seqlen {SEQLEN}); "
                         f"run with --chunk {SEQLEN} (--sweep does)")
    T = cal_tables(layer, src, fit, weight, group, V.device)
    tg = torch.arange(0, n, 1 << group, device=V.device)
    T = T[(tg % SEQLEN) >> group] if src == "pos" else T.expand(n >> group, SIZE)
    return nearest(P if pick == "code" else R, T, group, tie), X


def bf16_matmul(P_A, X_A, P_B, X_B):
    A = dequantize(P_A, X_A, axis=0).to(torch.bfloat16)
    B = dequantize(P_B, X_B, axis=0).to(torch.bfloat16)
    return (A.t() @ B).float()


# --------------------------------------------------------------------------------------------------- configs

FITS = [("code", "cnt"), ("code", "s2"), ("raw", "cnt"), ("raw", "s2")]
PICKS = ("code", "raw")


def ablation_names(groups):
    return [f"fit_{f}_{w}_pick_{p}_g{g}" for g in groups for f, w in FITS for p in PICKS]


def calibrated_names(groups, fits):
    return [f"cal_{s}_{fw}_pick_{p}_g{g}" for g in groups for s in ("pos", "shared") for fw in fits for p in PICKS]


def hardware_names(groups):
    """Experiment 3: what the finder circuit must do. hw_today first, hw_ideal (slow) last."""
    return [n for g in groups for n in [f"hw_today_g{g}"] +
            [f"hw_{f}_{fi}_{t}_g{g}" for f in ("code", "raw") for fi in ("code", "raw") for t in TIES] +
            [f"hw_ideal_g{g}"]]


def rules_for(name, layers):
    m = re.fullmatch(r"fit_(code|raw)_(cnt|s2)_pick_(code|raw)_g(\d+)", name)
    if m:
        f, w, p, g = m.group(1), m.group(2), m.group(3), int(m.group(4))
        q = partial(lut_data, fit=f, weight=w, pick=p, group=g)
        return LAYERS[layers](Scheme(name, a=q, b=q, reduce=bf16_matmul, rows=1 << g))
    m = re.fullmatch(r"cal_(pos|shared)_(code|raw)_(cnt|s2)_pick_(code|raw)_g(\d+)", name)
    if m:
        s, f, w, p, g = m.group(1), m.group(2), m.group(3), m.group(4), int(m.group(5))
        if layers != "mlp":
            raise SystemExit("lut_fit: calibrated configs cover the mlp layer set only")
        b = partial(lut_data, fit=f, weight=w, pick=p, group=g)
        per = [(n, Scheme(name, a=partial(lut_cal, layer=n, src=s, fit=f, weight=w, pick=p, group=g), b=b,
                          reduce=bf16_matmul, rows=1 << g)) for n in target_names()]
        return [(is_attention, None)] + per
    m = re.fullmatch(r"hw_(?:(today|ideal)|(code|raw)_(code|raw)_(low|even|zero))_g(\d+)", name)
    if m:
        g = int(m.group(5))
        if layers != "mlp":
            raise SystemExit("lut_fit: hardware configs cover the mlp layer set only")
        best_host = partial(lut_data, fit="raw", weight="cnt", pick="raw", group=g)
        if m.group(1) == "ideal":
            return LAYERS[layers](Scheme(name, a=best_host, b=best_host, reduce=bf16_matmul, rows=1 << g))
        if m.group(1) == "today":
            f, fi, t = "code", "code", "low"
            b = partial(lut_data, fit="code", weight="cnt", pick="code", group=g)
        else:
            f, fi, t = m.group(2), m.group(3), m.group(4)
            b = best_host
        per = [(n, Scheme(name, a=partial(lut_cal, layer=n, src="pos", fit=f, weight="cnt", pick=fi, group=g,
                                          tie=t), b=b, reduce=bf16_matmul, rows=1 << g)) for n in target_names()]
        return [(is_attention, None)] + per
    raise KeyError(name)


LAYERS = {"mlp": recipes.mlp_and_head, "all": lambda s: [(nn.Linear, s)]}


class _Rules(dict):
    """recipes.RULES entries built on first lookup, so any G given by name works without registering it here."""
    def __missing__(self, key):
        m = re.fullmatch(r"lutfit_(.+)_(mlp|all)", key)
        if not m:
            raise KeyError(key)
        self[key] = rules_for(m.group(1), m.group(2))
        return self[key]


recipes.RULES = _Rules(recipes.RULES)                    # llm_ppl.main looks its rule up here at run time


# ------------------------------------------------------------------------------------------------ self-check

def check():
    """CPU, no model. The rule at fit=code, weight=cnt, pick=code equals block.lut bit for bit; the chip's finder
    picks what the code pick picks; every table is 16 distinct E3M2 values within [-2, 2]; calibrated tables from
    a histogram of one tensor equal the data tables of that tensor (codes exactly)."""
    torch.manual_seed(0)
    ok = True
    allv = lut.formats._tables(FMT)[0]
    grid = set(torch.unique(allv[torch.isfinite(allv)]).tolist())
    K = 256
    cases = {"act": torch.randn(K, 32).bfloat16().float() * 0.5,
             "weight": (torch.randn(K, 64) * 0.02).bfloat16().float(),
             "outlier": torch.randn(K, 16).bfloat16().float()}
    cases["outlier"][::37, 3] = 40.0
    for g in (0, 1, 2):
        for nm, V in cases.items():
            mine, X = lut_data(V, fit="code", weight="cnt", pick="code", group=g)
            ref, Xr = block.lut.quantize(V, FMT, group=g, max_iters=MAX_ITERS, **COMMON)
            same = torch.equal(mine, ref) and torch.equal(X, Xr)
            ok &= same
            P, _, R, Xe = _operand(V)
            line = [f"g{g} {nm:7s} code/cnt/code == block.lut: {same}"]
            for f, w in FITS:
                T = tables_from_data(P, R, Xe, f, w, g)
                good = bool(torch.isfinite(T).all()) and all(len(set(r.tolist())) == SIZE for r in T) and \
                    set(T.unique().tolist()) <= grid and T.abs().max() <= 2
                fp = lut.finder(lut.encode(P, FMT), T.float(), FMT, group=g)
                hp = lut.pick(P, T.float(), group=g)
                good &= torch.equal(fp, hp)
                ok &= good
                errs = []
                for p in PICKS:
                    Q = nearest(P if p == "code" else R, T, g)
                    errs.append(((dequantize(Q, X) - V).norm() / V.norm()).item())
                line.append(f"{f}/{w}: tables ok {good}, err code-pick {errs[0]:.2%} raw-pick {errs[1]:.2%}")
            print("  " + " | ".join(line))
    # histogram path == data path for codes: one tensor binned as a calibration would bin it
    V = cases["act"]
    P, _, R, Xe = _operand(V)
    codes = _codes_axis("cpu")
    for g in (0, 1):
        npos = V.shape[1] >> g
        for w in ("cnt", "s2"):
            h = torch.zeros(npos, codes.numel(), dtype=torch.float64)
            pos = (torch.arange(V.shape[1]) >> g).expand_as(P)
            wt = torch.ones_like(P, dtype=torch.float64) if w == "cnt" else Xe.double() ** 2
            h.view(-1).index_add_(0, (pos * codes.numel() + torch.searchsorted(codes, P.contiguous())).reshape(-1),
                                  wt.reshape(-1))
            Tc = fit_points(codes.double().expand(npos, -1), h)
            Td = tables_from_data(P, R, Xe, "code", w, g)
            same = torch.equal(Tc, Td)
            ok &= same
            print(f"  g{g} calibration histogram (codes, {w}) tables == data tables: {same}")
    # tie rules: off ties all three agree; on an exact tie low/even/zero pick as documented; low == lut.pick
    T = torch.tensor([[-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, -2.0, -1.5, -0.75, 0.75, 0.125]])
    T = T.sort(dim=1)[0].double()                       # one table, 16 entries, ascending
    Z = torch.tensor([[-0.625, -0.375, 0.375, 0.625, 0.0625, 0.3, -1.9, 1.125]]).t()   # 8 rows, 1 column
    # ties at -0.625, -0.375, 0.375, 0.625, 0.0625 and 1.125; 0.3 and -1.9 are not ties
    want = {"low": [-0.75, -0.5, 0.25, 0.5, 0.0, 0.25, -2.0, 1.0],
            "even": [-0.5, -0.5, 0.25, 0.75, 0.0, 0.25, -2.0, 1.25],
            "zero": [-0.5, -0.25, 0.25, 0.5, 0.0, 0.25, -2.0, 1.0]}
    for tie in TIES:
        got = nearest(Z, T, 0, tie).t()[0].tolist()
        good = got == want[tie]
        ok &= good
        print(f"  tie rule {tie:4s}: {got}{'' if good else '  <-- FAIL'}")
    V = cases["act"]
    P, _, R, Xe = _operand(V)
    T = tables_from_data(P, R, Xe, "raw", "cnt", 1)
    lo = nearest(P, T, 1, "low")
    T32 = T.float()
    rows = T32[torch.arange(P.shape[1]) >> 1]
    v = P.t()
    hi = torch.searchsorted(rows, v.contiguous()).clamp(1, SIZE - 1)
    a, b = rows.gather(1, hi - 1), rows.gather(1, hi)
    generic_low = torch.where((v - a).abs() <= (b - v).abs(), a, b).t()
    good = torch.equal(lo, generic_low)
    ok &= good
    print(f"  tie rule low == lut.pick on real-shaped codes: {good}")
    for n in hardware_names((1,)):
        rules_for(n, "mlp")
    print(f"  hardware configs build: {', '.join(hardware_names((1,)))}")
    print("CHECK", "PASS" if ok else "FAIL")
    return ok


def check_device(device):
    """Every piece on `device` (default cuda) against the CPU, no model: table fitting (8 configs, G 0/1/2), the
    all-mxq corner against block.lut on the device, calibration histograms and calibrated tables (pos, shared),
    the bf16 matmul, and an MXLinear patched with a LUT Scheme. Count weights sum exactly, so those must match
    the CPU bit for bit; scale^2 sums round, and a GPU's scatter order may differ, so those may differ in a few
    tables (reported, with a rerun on the device to show whether the device itself is deterministic)."""
    import time
    from mxq.nn import patch
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"CHECK-DEVICE FAIL: {device} requested but torch.cuda.is_available() is False "
              f"(torch {torch.__version__}, built with CUDA {torch.version.cuda})")
        return False
    dev = torch.device(device)
    print(f"device {dev}: {torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu'}, torch {torch.__version__}")
    torch.manual_seed(0)
    ok = True

    def same_or_close(a, b, exact):
        a, b = a.cpu(), b.cpu()
        if torch.equal(a, b):
            return True, "equal"
        frac = (a != b).float().mean().item()
        return (not exact) and frac < 1e-3, f"{frac:.2e} of elements differ"

    K = 256
    cases = {"act": torch.randn(K, 64).bfloat16().float() * 0.5,
             "weight": (torch.randn(K, 128) * 0.02).bfloat16().float(),
             "outlier": torch.randn(K, 32).bfloat16().float()}
    cases["outlier"][::37, 3] = 40.0
    for g in (0, 1, 2):
        for nm, V in cases.items():
            Vd = V.to(dev)
            mine, X = lut_data(Vd, fit="code", weight="cnt", pick="code", group=g)
            ref, Xr = block.lut.quantize(Vd, FMT, group=g, max_iters=MAX_ITERS, **COMMON)
            good = mine.device.type == dev.type and torch.equal(mine, ref) and torch.equal(X, Xr)
            ok &= good
            line = [f"g{g} {nm:7s} on-device == block.lut: {good}"]
            for f, w in FITS:
                for p in PICKS:
                    Qc, Xc = lut_data(V, fit=f, weight=w, pick=p, group=g)
                    Qd, Xd = lut_data(Vd, fit=f, weight=w, pick=p, group=g)
                    g1, why = same_or_close(Qd, Qc, exact=(w == "cnt"))
                    g1 &= torch.equal(Xd.cpu(), Xc) and bool(torch.isfinite(Qd).all())
                    if not torch.equal(Qd.cpu(), Qc):
                        Qd2, _ = lut_data(Vd, fit=f, weight=w, pick=p, group=g)
                        why += f", device rerun {'identical' if torch.equal(Qd2, Qd) else 'DIFFERS'}"
                    ok &= g1
                    if why != "equal" or not g1:
                        line.append(f"{f}/{w}/{p}: {why}{'' if g1 else ' <-- FAIL'}")
            if len(line) == 1:
                line.append("all 8 configs equal to cpu")
            print("  " + " | ".join(line))

    # calibration: histograms of two windows on the device == on the CPU; calibrated tables and lut_cal on device
    S, Kc, g = SEQLEN, 64, 1
    xs = [(torch.randn(S, Kc) * (1 + 3 * torch.rand(1, Kc))).bfloat16() for _ in range(2)]
    Hc, Hd = new_histograms(S >> g, "cpu"), new_histograms(S >> g, dev)
    for x in xs:
        accumulate(Hc, x, g)
        accumulate(Hd, x.to(dev), g)
    for k in Hc:
        exact = k.endswith("cnt")
        g1, why = (torch.equal(Hd[k].cpu(), Hc[k]), "equal") if exact else \
            (torch.allclose(Hd[k].cpu(), Hc[k], rtol=1e-12, atol=0), "close (rtol 1e-12)")
        ok &= g1
        print(f"  calibration histogram {k}: {why if g1 else 'MISMATCH <-- FAIL'}")
    layer = "check.layer"
    key = str(cal_path(g))
    saved = _CAL.pop(key, None)
    try:
        for src in ("pos", "shared"):
            for fw in ("code_cnt", "raw_s2"):
                f, w = fw.split("_")
                for p, tie in [(p, t) for p in PICKS for t in TIES]:
                    # as in a run: the calibration loaded on cpu, patch's smoke matmul on cpu first, then the
                    # forward on the device, all against one cache
                    _CAL.clear()
                    _CAL[key] = {"hist": {layer: {k: t.cpu() for k, t in Hc.items()}}}
                    lut_cal(torch.randn(32, 4), layer=layer, src=src, fit=f, weight=w, pick=p, group=g, tie=tie)
                    outs = [lut_cal(xs[0].float().t().contiguous().to(d), layer=layer, src=src, fit=f, weight=w,
                                    pick=p, group=g, tie=tie) for d in ("cpu", dev)]
                    g1, why = same_or_close(outs[1][0], outs[0][0], exact=(w == "cnt"))
                    g1 &= outs[1][0].device.type == dev.type
                    ok &= g1
                    print(f"  lut_cal {src:6s} {fw:8s} pick {p:4s} tie {tie:4s}: {why}{'' if g1 else '  <-- FAIL'}")
    finally:
        _CAL.clear()
        if saved is not None:
            _CAL[key] = saved

    # the bf16 matmul and an MXLinear on the device
    A, B = torch.randn(K, 16), torch.randn(K, 24) * 0.05
    q = partial(lut_data, fit="raw", weight="cnt", pick="raw", group=1)
    (PA, XA), (PB, XB) = q(A), q(B)
    Yc = bf16_matmul(PA, XA, PB, XB)
    Yd = bf16_matmul(PA.to(dev), XA.to(dev), PB.to(dev), XB.to(dev))
    rel = ((Yd.cpu() - Yc).norm() / Yc.norm()).item()
    g1 = bool(torch.isfinite(Yd).all()) and rel < 2 ** -7
    ok &= g1
    print(f"  bf16 matmul on device vs cpu: rel {rel:.2e}{'' if g1 else '  <-- FAIL'}")
    net = nn.Sequential(nn.Linear(K, 48, bias=False)).to(dev, torch.bfloat16)
    x = torch.randn(2, 8, K, device=dev, dtype=torch.bfloat16)
    y_ref = net(x).float()
    handle = patch(net, [(nn.Linear, Scheme("check", a=q, b=q, reduce=bf16_matmul, rows=2))])
    y = net(x)
    handle.revert()
    rel = ((y.float() - y_ref).norm() / y_ref.norm()).item()
    g1 = y.device.type == dev.type and y.dtype == torch.bfloat16 and bool(torch.isfinite(y).all()) and rel < 0.5
    ok &= g1
    print(f"  patched MXLinear forward on device: {tuple(y.shape)} {y.dtype}, vs unquantized {rel:.1%}"
          f"{'' if g1 else '  <-- FAIL'}")

    # cost of one TinyLlama-sized activation (gate_proj input, 2048 tokens) per fit
    V = (torch.randn(2048, 2048) * 0.5).bfloat16().float().to(dev)
    for f, w in FITS:
        lut_data(V, fit=f, weight=w, pick="raw", group=1)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t = time.time()
        lut_data(V, fit=f, weight=w, pick="raw", group=1)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        print(f"  timing: one 2048x2048 activation, fit {f}/{w}: {time.time() - t:.3f}s")
    print("CHECK-DEVICE", "PASS" if ok else "FAIL")
    return ok


# ------------------------------------------------------------------------------------------------------ sweep

def _calibrated(name):
    """Does this config read calibrated A tables (so it needs the calibration and whole windows per call)?"""
    return name.startswith("cal_") or (name.startswith("hw_") and not name.startswith("hw_ideal"))


def sweep(argv):
    import argparse, time
    ap = argparse.ArgumentParser(description="the chosen experiments' configurations in turn, then a table")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--exp", choices=["ablation", "calibrated", "both", "hardware"], default="both")
    ap.add_argument("--configs", default=None, help="comma list of config names (overrides --exp)")
    ap.add_argument("--groups", default=None, help="LUT group sizes G (default 1)")
    ap.add_argument("--cal-nsamples", type=int, default=None)
    ap.add_argument("--cal-seed", type=int, default=None)
    ap.add_argument("--cal-fits", default=None, help="fits for experiment 2 (default code_cnt,raw_s2)")
    ap.add_argument("--layers", choices=list(LAYERS), default="mlp")
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--nsamples", type=int, default=None)
    ap.add_argument("--seqlen", type=int, default=None)
    ap.add_argument("--sequential", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    env = dict(os.environ)
    for k, v in (("LUTFIT_GROUPS", a.groups), ("LUTFIT_CAL_NSAMPLES", a.cal_nsamples),
                 ("LUTFIT_CAL_SEED", a.cal_seed), ("LUTFIT_CAL_FITS", a.cal_fits), ("LUTFIT_SEQLEN", a.seqlen)):
        if v is not None:
            env[k] = str(v)
    groups = tuple(int(v) for v in env.get("LUTFIT_GROUPS", "1").split(","))
    fits = tuple(env.get("LUTFIT_CAL_FITS", "code_cnt,raw_s2").split(","))
    if a.configs:
        names = a.configs.split(",")
    else:
        names = (ablation_names(groups) if a.exp in ("ablation", "both") else []) + \
            (calibrated_names(groups, fits) if a.exp in ("calibrated", "both") else []) + \
            (hardware_names(groups) if a.exp == "hardware" else [])
    for n in names:
        rules_for(n, a.layers)                                                  # refuse a bad name up front
    seqlen = int(env.get("LUTFIT_SEQLEN", "2048"))
    first_gpu = a.gpus.split(",")[0] if a.gpus else None
    cal_env = dict(env, **({"CUDA_VISIBLE_DEVICES": first_gpu} if first_gpu is not None else {}))
    for g in sorted({int(n.rsplit("_g", 1)[1]) for n in names if _calibrated(n)}):
        path = RESULTS / f"lutcal_g{g}_s{seqlen}_n{env.get('LUTFIT_CAL_NSAMPLES', '16')}_seed" \
                         f"{env.get('LUTFIT_CAL_SEED', '0')}.pt"
        if a.dry_run or path.exists():
            continue
        print(f"=== calibrating G={g} -> {path}", flush=True)
        if subprocess.run([sys.executable, str(HERE), "--calibrate", str(g)], env=cal_env).returncode:
            raise SystemExit("calibration failed")
    rows, t0 = [], time.time()
    for i, n in enumerate(names):
        rule = f"lutfit_{n}_{a.layers}"
        cmd = [sys.executable, str(HERE), "--rules", rule, "--out", str(RESULTS / f"{rule}.json")]
        for flag, v in (("--gpus", a.gpus), ("--nsamples", a.nsamples), ("--seqlen", a.seqlen)):
            if v is not None:
                cmd += [flag, str(v)]
        if _calibrated(n):
            cmd += ["--chunk", str(seqlen)]                                     # whole windows: positions known
        cmd += ["--sequential"] * a.sequential + ["--dry-run"] * a.dry_run
        print(f"=== [{i + 1}/{len(names)}] {n}  ({time.time() - t0:.0f}s elapsed): {' '.join(cmd)}", flush=True)
        if subprocess.run(cmd, env=env).returncode:
            rows.append((n, "FAILED", ""))
            continue
        if a.dry_run:
            continue
        r = json.loads((RESULTS / f"{rule}.json").read_text())
        rows.append((n, f"{r['perplexity']:.4f}", f"{len(r['per_sample'])} x {r['seqlen']}"))
    if rows:
        print(f"\nTinyLlama WikiText-2 perplexity, {FMT} LUTs at the chip's range, bf16 matmuls, layers {a.layers}")
        print("  (references: lutcmp_mxq_g1 = fit_code_cnt_pick_code_g1, lutcmp_mxquant_cap2_g1, none)")
        for n, p, s in rows:
            print(f"  {n:40s} {p:>10s}   {s}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(0 if check() else 1)
    elif "--check-gpu" in sys.argv:
        i = sys.argv.index("--check-gpu")
        dev = sys.argv[i + 1] if len(sys.argv) > i + 1 and not sys.argv[i + 1].startswith("-") else "cuda"
        sys.exit(0 if check_device(dev) else 1)
    elif "--calibrate" in sys.argv:
        calibrate(int(sys.argv[sys.argv.index("--calibrate") + 1]))
    elif "--sweep" in sys.argv:
        sweep(sys.argv[1:])
    else:
        if "--seqlen" in sys.argv and int(sys.argv[sys.argv.index("--seqlen") + 1]) != SEQLEN:
            raise SystemExit("lut_fit: --seqlen must match LUTFIT_SEQLEN (the sweep sets both)")
        llm_ppl.__file__ = str(HERE)      # --gpus workers re-run this file, so they see the rules registered above
        llm_ppl.main()
