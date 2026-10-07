#!/usr/bin/env python3
"""TinyLlama WikiText-2 perplexity: MXQuant's FP6 LUT (level2) against microscaling-quant's (mxq.lut), the
quantization style in isolation.

Both put each MXFP6_E3M2 element through a 16-entry table, one table per 2^G rows of A (tokens) / columns of B
(output channels), spanning all of K. They differ in how the operand is built:

    mxq        block.lut, the chip's rule: block scale 2^floor(log2 amax) (elements <= 2), RNE to E3M2 codes,
               then deterministic count-weighted k-means over the codes (quantile seeds, up to 50 passes), tables of
               finder-safe values.
    mxquant    MXQuant's level2 as shipped (methods/channel_groups.py, granularity 'channel', channel_group_size
               = 2^G): OCP-style scale 2^(floor(log2 amax) - 4), so elements reach 28; k-means++ seeds, 10
               scale-weighted Lloyd passes on the unrounded values, centres snapped to its 65-value codebook.
    mxquant_capC   the same MXQuant k-means with every element and table entry inside [-C, C]: the block scale
               moves up so the block max lands in [C/2, C) (emax = log2 C - 1), and the codebook keeps only the
               values with |v| <= C. C = 2 is mxq's element range, so mxquant_cap2 vs mxq separates the table
               fitting from the scale convention.

Every matmul is the same and carries no quantization of its own: operands dequantized (codes x scales, exact in
bf16), then a bf16 matmul, as the model's own nn.Linear computes. The no-LUT FP6 baselines run the same way:

    none               the model as loaded (bf16)
    fp6_direct_mxq     MXFP6_E3M2 without a LUT, mxq's chip rule (block.mxgemmini)
    fp6_direct_mxquant MXFP6_E3M2 without a LUT, MXQuant's own simulation (block.mxquant)
    {mxq,mxquant,mxquant_capC}_g{G}

Layer set: --layers mlp (default) = every nn.Linear but the attention projections, lm_head quantized; --layers all.

    python experiments/lut_mxquant_vs_mxq_ppl.py --check                       # CPU self-check, no model
    python experiments/lut_mxquant_vs_mxq_ppl.py --sweep --gpus 0,1,2,3        # everything at G=1, cap 2
    python experiments/lut_mxquant_vs_mxq_ppl.py --sweep --gpus 0,1,2,3 --groups 0,1,5 --caps 2,4
    python experiments/lut_mxquant_vs_mxq_ppl.py --sweep --configs mxquant_g1,mxq_g1 --gpus 0
    python experiments/lut_mxquant_vs_mxq_ppl.py --sweep --dry-run --configs mxquant_g1

MXQuant is imported from MXQUANT_ROOT (default: npu-exploration/MXQuant). Its k-means++ seeding is random; every
quantizer call runs under a fixed seed (env MXQUANT_SEED, default 0) on a forked RNG, so reruns repeat.
Results: experiments/results/lutcmp_<config>_<layers>.json, and a table at the end of --sweep.
"""
import json, math, os, subprocess, sys, types
from contextlib import contextmanager
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = Path(os.environ.get("MXQ_REPO", HERE.parent.parent))
MXQUANT_ROOT = Path(os.environ.get("MXQUANT_ROOT", REPO.parent / "MXQuant"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(MXQUANT_ROOT / "microxcaling"))

import torch  # noqa: E402
from torch import nn  # noqa: E402

for _m in ("matplotlib", "matplotlib.pyplot"):            # channel_groups imports pyplot for a debug plot only
    try:
        __import__(_m)
    except ImportError:
        sys.modules[_m] = types.ModuleType(_m)

from mxq import Scheme, block, fp64_accum, scale_factor  # noqa: E402
from mxq.block._driver import dequantize  # noqa: E402
from mxq.lut.formats import _tables  # noqa: E402
from experiments import llm_ppl, recipes  # noqa: E402

import mx.elemwise_ops as _E  # noqa: E402
from mx import mx_ops as _mx_ops  # noqa: E402
from mx.level2 import config as _l2cfg  # noqa: E402
from mx.level2.methods import channel_groups as _cg  # noqa: E402

if _E._quantize_level2 is not _cg._quantize_level2:
    raise SystemExit("MXQuant's elemwise_ops is not on the channel_groups level2 method (flags at the top of "
                     "mx/elemwise_ops.py); this experiment compares that one")

FMT, MX_FMT = "MXFP6_E3M2", "fp6_e3m2"
MAX_ITERS = 50                                           # mxq's k-means, as the run recipes fit tables
SEED = int(os.environ.get("MXQUANT_SEED", "0"))
ROW_CHUNK = 1024                                         # rows of a weight quantized per MXQuant call (memory)
COMMON = dict(axis=0, block_size=32, rounding_mode="rne", scale_floor=scale_factor.HARDWARE_FLOOR)


def bf16_matmul(P_A, X_A, P_B, X_B):
    """Y = Âᵀ·B̂ in bf16: dequantized operands (exact in bf16), torch's bf16 matmul, as nn.Linear."""
    A = dequantize(P_A, X_A, axis=0).to(torch.bfloat16)
    B = dequantize(P_B, X_B, axis=0).to(torch.bfloat16)
    return (A.t() @ B).float()


# ---------------------------------------------------------------------------------------------- MXQuant operand

@contextmanager
def _mxquant_setup(cap, group_rows, captured):
    """MXQuant's level2 singleton, codebook and format emax for one call; everything restored after."""
    cfg = _l2cfg.get_l2_config()
    saved_cfg = dict(vars(cfg))
    saved = (_E._quantize_level2, _cg._fp6_codebook, _mx_ops._get_format_params)
    cfg.apply_level2, cfg.granularity, cfg.algorithm = True, "channel", "kmeans"
    cfg.num_signposts, cfg.channel_group_size = 16, group_rows

    def level2(tensor, **kw):
        captured.append(kw["shared_exp"])
        return saved[0](tensor, **kw)
    _E._quantize_level2 = level2
    if cap is not None:
        book = [v for v in _cg.FP6_CODEBOOK_VALUES if abs(v) <= cap]
        _cg._fp6_codebook = lambda device, dtype: torch.tensor(book, device=device, dtype=dtype)

        def params(fmt):
            ebits, mbits, emax, max_norm, min_norm = saved[2](fmt)
            return ebits, mbits, int(math.log2(cap)) - 1, max_norm, min_norm
        _mx_ops._get_format_params = params
    try:
        yield
    finally:
        _E._quantize_level2, _cg._fp6_codebook, _mx_ops._get_format_params = saved
        vars(cfg).update(saved_cfg)


def _mxquant_rows(Vt, group_rows, cap):
    """MXQuant on n×K rows that form whole tables of `group_rows`. Returns codes n×K, scales n×ceil(K/32)."""
    captured = []
    with _mxquant_setup(cap, group_rows, captured):
        out = _mx_ops._quantize_mx(Vt, 8, MX_FMT, axes=[-1], block_size=32, round="nearest")
    (se,) = captured                                                  # n × ceil(K/32) × 1
    X = torch.exp2(se.squeeze(-1).float())
    P = out / X.repeat_interleave(32, dim=1)[:, :Vt.shape[1]]
    return P, X


def mxquant_lut(V, *, group, cap=None):
    """Block-quantizer contract for MXQuant's level2: V is K×n (blocks along K) -> P K×n codes, X ceil(K/32)×n.
    2^group columns share a table; a short last group (n not a multiple) is its own table."""
    Vt = V.detach().t().float().contiguous()
    n, N = Vt.shape[0], 1 << group
    full = n - n % N
    step = max(N, ROW_CHUNK // N * N)
    spans = [(s, min(s + step, full), N) for s in range(0, full, step)]
    if full < n:
        spans.append((full, n, n - full))
    devs = [Vt.device] if Vt.is_cuda else []
    with torch.random.fork_rng(devices=devs):
        torch.manual_seed(SEED)
        parts = [_mxquant_rows(Vt[a:b], g, cap) for a, b, g in spans]
    P = torch.cat([p for p, _ in parts]).t().contiguous()
    X = torch.cat([x for _, x in parts]).t().contiguous()
    return P, X


# --------------------------------------------------------------------------------------------------- configs

def schemes(groups, caps):
    direct = {"fp6_direct_mxq": partial(block.mxgemmini.quantize, fmt=FMT, **COMMON),
              "fp6_direct_mxquant": partial(block.mxquant.quantize, fmt=FMT, axis=0)}
    out = {n: Scheme(n, a=q, b=q, reduce=bf16_matmul) for n, q in direct.items()}
    for g in groups:
        arms = {"mxq": partial(block.lut.quantize, fmt=FMT, group=g, max_iters=MAX_ITERS, **COMMON),
                "mxquant": partial(mxquant_lut, group=g)}
        arms.update({f"mxquant_cap{c}": partial(mxquant_lut, group=g, cap=c) for c in caps})
        for arm, q in arms.items():
            name = f"{arm}_g{g}"
            out[name] = Scheme(name, a=q, b=q, reduce=bf16_matmul, rows=1 << g)
    return out


def _ints(s):
    return tuple(int(v) for v in s.split(",")) if s else ()


GROUPS = _ints(os.environ.get("LUTCMP_GROUPS", "1"))
CAPS = _ints(os.environ.get("LUTCMP_CAPS", "2"))
for _c in CAPS:
    if _c < 1 or _c & (_c - 1) or _c > 16:
        raise SystemExit(f"cap {_c}: must be a power of two <= 16")
SCHEMES = schemes(GROUPS, CAPS)
LAYERS = {"mlp": recipes.mlp_and_head, "all": lambda s: [(nn.Linear, s)]}
recipes.RULES.update({f"lutcmp_{n}_{l}": fn(s) for n, s in SCHEMES.items() for l, fn in LAYERS.items()})


# ------------------------------------------------------------------------------------------------ self-check

def check():
    """CPU, no model: every LUT operand is <= 16-entry tables of E3M2 codes inside its range; MXQuant's repeat
    under the fixed seed; every scheme's bf16 matmul is finite and within bf16 rounding of the exact one."""
    torch.manual_seed(0)
    allv = _tables(FMT)[0]
    grid = set(torch.unique(allv[torch.isfinite(allv)]).tolist())
    K, M, Nout = 256, 8, 32
    A = torch.randn(K, M) * 0.5
    B = torch.randn(K, Nout) * 0.02
    B[3, 5] = 0.4                                                                  # an outlier
    ok = True
    for name, s in SCHEMES.items():
        g = int(name.rsplit("_g", 1)[1]) if "_g" in name else None
        (PA, XA), (PB, XB) = s.a(A), s.b(B)
        if name.startswith("mxquant"):
            again = s.b(B)
            if not (torch.equal(again[0], PB) and torch.equal(again[1], XB)):
                print(f"  {name}: NOT reproducible under the fixed seed"); ok = False
        if g is not None:
            arm = name.rsplit("_g", 1)[0]
            cap = int(arm[len("mxquant_cap"):]) if arm.startswith("mxquant_cap") else (2 if arm == "mxq" else 28)
            for nm, P, X, V in (("A", PA, XA, A), ("B", PB, XB, B)):
                most = max(len(torch.unique(P[:, i:i + (1 << g)])) for i in range(0, P.shape[1], 1 << g))
                off = set(torch.unique(P).tolist()) - grid
                big = P.abs().max().item()
                err = ((dequantize(P, X, axis=0) - V).norm() / V.norm()).item()
                bad = most > 16 or bool(off) or big > cap
                ok &= not bad
                print(f"  {name:18s} {nm}: <= {most:2d} values/table, max |code| {big:5.2f} (cap {cap:2d}), "
                      f"non-E3M2 {sorted(off)[:4]}, operand err {err:.3%}{'  <-- FAIL' if bad else ''}")
        Y = s.reduce(PA, XA, PB, XB)
        ref = fp64_accum(PA, XA, PB, XB)
        rel = ((Y - ref).norm() / ref.norm()).item()
        good = bool(torch.isfinite(Y).all()) and rel < 2 ** -7
        ok &= good
        print(f"  {name:18s} bf16 matmul: finite {bool(torch.isfinite(Y).all())}, vs exact on same codes {rel:.4%}"
              f"{'' if good else '  <-- FAIL'}")
    print("CHECK", "PASS" if ok else "FAIL")
    return ok


# ------------------------------------------------------------------------------------------------------ sweep

def sweep(argv):
    import argparse, time
    ap = argparse.ArgumentParser(description="every configuration in turn, then a table")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--configs", default=None, help="comma list of: none, " + ", ".join(SCHEMES))
    ap.add_argument("--groups", default=None, help="LUT group sizes G (default 1)")
    ap.add_argument("--caps", default=None, help="mxquant_cap values C, powers of two <= 16 (default 2)")
    ap.add_argument("--layers", choices=list(LAYERS), default="mlp")
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--nsamples", type=int, default=None)
    ap.add_argument("--seqlen", type=int, default=None)
    ap.add_argument("--sequential", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    env = dict(os.environ)
    if a.groups is not None:
        env["LUTCMP_GROUPS"] = a.groups
    if a.caps is not None:
        env["LUTCMP_CAPS"] = a.caps
    known = schemes(_ints(env.get("LUTCMP_GROUPS", "1")), _ints(env.get("LUTCMP_CAPS", "2")))
    names = a.configs.split(",") if a.configs else ["none"] + list(known)
    bad = [n for n in names if n != "none" and n not in known]
    if bad:
        raise SystemExit(f"unknown config(s) {bad}; known: none, {', '.join(known)}")
    results = REPO / "experiments" / "results"
    rows, t0 = [], time.time()
    for i, n in enumerate(names):
        rule = "none" if n == "none" else f"lutcmp_{n}_{a.layers}"
        cmd = [sys.executable, str(HERE), "--rules", rule]
        for flag, v in (("--gpus", a.gpus), ("--nsamples", a.nsamples), ("--seqlen", a.seqlen)):
            if v is not None:
                cmd += [flag, str(v)]
        cmd += ["--sequential"] * a.sequential + ["--dry-run"] * a.dry_run
        print(f"=== [{i + 1}/{len(names)}] {n}  ({time.time() - t0:.0f}s elapsed): {' '.join(cmd)}", flush=True)
        if subprocess.run(cmd, env=env).returncode:
            rows.append((n, "FAILED", ""))
            continue
        if a.dry_run:
            continue
        r = json.loads((results / f"{rule}.json").read_text())
        rows.append((n, f"{r['perplexity']:.4f}", f"{len(r['per_sample'])} x {r['seqlen']}"))
    if rows:
        print(f"\nTinyLlama WikiText-2 perplexity, {FMT}, bf16 matmuls, layers {a.layers}")
        for n, p, s in rows:
            print(f"  {n:22s} {p:>10s}   {s}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(0 if check() else 1)
    elif "--sweep" in sys.argv:
        sweep(sys.argv[1:])
    else:
        llm_ppl.__file__ = str(HERE)      # --gpus workers re-run this file, so they see the rules registered above
        llm_ppl.main()
