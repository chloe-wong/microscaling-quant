#!/usr/bin/env python3
"""TinyLlama WikiText-2 perplexity of MX-Gemmini's FP6 LUT formats against the LUT group size G.

A LUT operand carries 4-bit indices into a 16-entry table of FP6 values, one table per 2^G rows of A (tokens) /
columns of B (output channels), spanning all of K (mxq.block.lut, k-means tables as the run recipes fit them). Small G:
tables fitted to fewer rows / channels (more accurate, more tables); large G: one table serves many (G = 13 is one per
tensor for TinyLlama: 2^13 > every width and the 2048-token window).

Every configuration runs on the MX-Gemmini datapath (recipes.ARITHMETIC + recipes.LADDER: product truncation, the
accumulator ladder, bf16 across tiles), with the hardware block scale (2^floor(log2 amax), floor 2^-23) and RNE:

    none             the model as loaded (bf16)
    fp8_e4m3         MXFP8 E4M3 (recipes.OPERANDS, the hw_mxfp8_tapeout operands)
    fp6_<f>_direct   MXFP6 <f> on its element grid, no LUT
    fp6_<f>_lut_g<G> MXFP6 <f> through the LUT, group G, k-means max_iters 50      (<f> = e3m2 | e2m3)

Every operand's table is fitted to that operand on the host and picked by value (the first stage of a chip kernel);
on the chip a chained activation is instead projected by the requantizer's finder onto a table estimated in advance.

Layer set: --layers mlp (default) = recipes.mlp_and_head: every nn.Linear but the attention projections, lm_head
quantized (MXQuant's set, as hw_mxfp8_tapeout); --layers all = every nn.Linear.

Put this file in a microscaling-quant checkout that has mxq/lut (origin/main) under experiments/, or point MXQ_REPO at one:

    python experiments/lut_granularity_ppl.py --sweep --gpus 0,1,2,3                       # every configuration
    python experiments/lut_granularity_ppl.py --sweep --gpus 0,1,2,3 --configs fp6_e3m2_lut_g1,fp6_e3m2_lut_g5
    python experiments/lut_granularity_ppl.py --rules lutppl_fp6_e2m3_lut_g0_mlp --gpus 0,1   # one, llm_ppl's own flags
    python experiments/lut_granularity_ppl.py --sweep --dry-run --configs fp6_e3m2_lut_g1      # which layer gets what

--sweep passes --nsamples / --seqlen / --sequential / --gpus / --layers through (defaults: llm_ppl's 16 seeded windows
of 2048) and prints a table from the result JSONs (experiments/results/lutppl_*.json).
"""
import json, os, subprocess, sys
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = Path(os.environ.get("MXQ_REPO", HERE.parent.parent))
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from mxq import Scheme, block, matmul, scale_factor  # noqa: E402

if not hasattr(block, "lut"):
    raise SystemExit(f"mxq at {REPO} has no LUT support (mxq/block/lut.py, mxq/lut/): check out microscaling-quant "
                     "origin/main or later there, or point MXQ_REPO at a checkout that has it")
from experiments import llm_ppl, recipes  # noqa: E402

GROUPS = (0, 1, 2, 3, 5, 13)
FORMATS = {"e3m2": "MXFP6_E3M2", "e2m3": "MXFP6_E2M3"}
MAX_ITERS = 50
# compiled == MXGEMMINI bit for bit; torch.compile's CPU backend fails on it, so CPU runs take the eager one (slow)
ARITH = recipes.ARITHMETIC if torch.cuda.is_available() else matmul.MXGEMMINI()
REDUCE = partial(matmul.systolic, arith=ARITH, schedule=recipes.LADDER)
COMMON = dict(axis=0, block_size=32, rounding_mode="rne", scale_floor=scale_factor.HARDWARE_FLOOR)


def schemes():
    out = {"fp8_e4m3": Scheme("fp8_e4m3", a=recipes.OPERANDS, b=recipes.OPERANDS, reduce=REDUCE)}
    for f, fmt in FORMATS.items():
        q = partial(block.mxgemmini.quantize, fmt=fmt, **COMMON)
        out[f"fp6_{f}_direct"] = Scheme(f"fp6_{f}_direct", a=q, b=q, reduce=REDUCE)
        for g in GROUPS:
            q = partial(block.lut.quantize, fmt=fmt, group=g, max_iters=MAX_ITERS, **COMMON)
            out[f"fp6_{f}_lut_g{g}"] = Scheme(f"fp6_{f}_lut_g{g}", a=q, b=q, reduce=REDUCE, rows=1 << g)
    return out


SCHEMES = schemes()
LAYERS = {"mlp": recipes.mlp_and_head, "all": lambda s: [(nn.Linear, s)]}
recipes.RULES.update({f"lutppl_{n}_{l}": fn(s) for n, s in SCHEMES.items() for l, fn in LAYERS.items()})


def sweep(argv):
    import argparse
    ap = argparse.ArgumentParser(description="every configuration in turn, then a table")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--configs", default=None, help="comma list of: none, " + ", ".join(SCHEMES))
    ap.add_argument("--layers", choices=list(LAYERS), default="mlp")
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--nsamples", type=int, default=None)
    ap.add_argument("--seqlen", type=int, default=None)
    ap.add_argument("--sequential", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    names = a.configs.split(",") if a.configs else ["none"] + list(SCHEMES)
    bad = [n for n in names if n != "none" and n not in SCHEMES]
    if bad:
        raise SystemExit(f"unknown config(s) {bad}; known: none, {', '.join(SCHEMES)}")
    results = REPO / "experiments" / "results"
    rows = []
    import time
    t0 = time.time()
    for i, n in enumerate(names):
        rule = "none" if n == "none" else f"lutppl_{n}_{a.layers}"
        cmd = [sys.executable, str(HERE), "--rules", rule]
        for flag, v in (("--gpus", a.gpus), ("--nsamples", a.nsamples), ("--seqlen", a.seqlen)):
            if v is not None:
                cmd += [flag, str(v)]
        cmd += ["--sequential"] * a.sequential + ["--dry-run"] * a.dry_run
        print(f"=== [{i + 1}/{len(names)}] {n}  ({time.time() - t0:.0f}s elapsed): {' '.join(cmd)}", flush=True)
        if subprocess.run(cmd).returncode:
            rows.append((n, "FAILED", ""))
            continue
        if a.dry_run:
            continue
        r = json.loads((results / f"{rule}.json").read_text())
        rows.append((n, f"{r['perplexity']:.4f}", f"{len(r['per_sample'])} x {r['seqlen']}"))
    if rows:
        print(f"\nTinyLlama WikiText-2 perplexity, layers {a.layers}")
        for n, p, s in rows:
            print(f"  {n:22s} {p:>10s}   {s}")


if __name__ == "__main__":
    if "--sweep" in sys.argv:
        sweep(sys.argv[1:])
    else:
        llm_ppl.__file__ = str(HERE)      # --gpus workers re-run this file, so they see the rules registered above
        llm_ppl.main()
