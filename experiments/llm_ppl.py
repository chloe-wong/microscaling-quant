"""Perplexity of a causal LM on sampled WikiText-2 with a rule list from experiments/recipes.py.

    python experiments/llm_ppl.py --rules none                          # the model as loaded, in bf16
    python experiments/llm_ppl.py --rules hw_fp8_tapeout --gpus 0,1,2,3
    python experiments/llm_ppl.py --rules hw_fp8_tapeout --dry-run      # which layer gets which Scheme, then stop

The result lands in experiments/results/<rules>.json, which is not tracked: it holds the per-sample numbers,
the rule list and the Schemes as they actually ran, so a number can be traced back to a chain.

Data sampling, model loading and the loss follow MXQuant complete_integration_e2e/eval_complete.py
(`load_data_sampled`, `load_model`, `eval_sampled`) so the numbers are comparable to its published ones.
Samples are independent forwards, so with several GPUs each takes a slice of the samples; the per-sample sums
are added in sample order, exactly as the single-process loop adds them.
"""
import argparse, json, math, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch
import torch.nn.functional as F


def load_model(model_id, seqlen):
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(model_id)
    ctx = getattr(config, "max_position_embeddings", None)
    if ctx and seqlen > ctx:
        config.rope_scaling = {"type": "linear", "factor": float(math.ceil(seqlen / ctx))}
    model = AutoModelForCausalLM.from_pretrained(model_id, config=config, use_safetensors=True,
                                                 torch_dtype=torch.bfloat16, device_map="auto")
    return model.eval().bfloat16()


def load_samples(model_id, seqlen, nsamples, seed):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([x for x in data["text"] if x.strip()])
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    nbatch = ids.numel() // seqlen
    ids = ids[:nbatch * seqlen].reshape(nbatch, seqlen)
    torch.manual_seed(seed)
    if nsamples < nbatch:
        ids = ids[torch.randperm(nbatch)[:nsamples]]
    return ids


@torch.no_grad()
def sample_nll(model, ids):
    device = next(model.parameters()).device
    ids = ids.unsqueeze(0).to(device)
    logits = model(ids, use_cache=False).logits[0, :-1, :]
    nll = -F.log_softmax(logits.float(), dim=-1).gather(1, ids[0, 1:].unsqueeze(1)).squeeze(1)
    return nll.sum().item(), nll.numel()


def commit():
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def describe(obj):
    """A Scheme, rule list or any piece of one as plain JSON: functions by their dotted name, partials with their
    bound keywords, an Arithmetic by its name. Read off the objects, so the record says what actually ran."""
    from functools import partial
    from mxq import Scheme
    from mxq.matmul import Arithmetic
    if isinstance(obj, Scheme):
        return {"name": obj.name, "a": describe(obj.a), "b": describe(obj.b), "reduce": describe(obj.reduce)}
    if isinstance(obj, partial):
        return {"call": describe(obj.func), **{k: describe(v) for k, v in obj.keywords.items()}}
    if isinstance(obj, Arithmetic):
        return obj.name
    if isinstance(obj, type) or callable(obj):
        public = [m for m in obj.__module__.split(".") if not m.startswith("_")]       # mxq.matmul._systolic -> mxq.matmul
        return ".".join(public + [obj.__qualname__])
    if isinstance(obj, (list, tuple)):
        return [describe(v) for v in obj]
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", required=True, help="a key of experiments.recipes.RULES")
    ap.add_argument("--model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--nsamples", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=None, help="token rows per reducer call (default: by output width)")
    ap.add_argument("--gpus", default=None, help="e.g. 0,1,2,3: one worker per GPU, samples split between them")
    ap.add_argument("--samples", default=None, help="worker only: START:END sample indices")
    ap.add_argument("--out", default=None, help="result JSON (default experiments/results/<rules>.json)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from experiments import recipes
    from mxq.nn import patch
    rules = recipes.RULES[args.rules]
    out = Path(args.out or REPO / "experiments" / "results" / f"{args.rules}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.gpus and "," in args.gpus and not args.samples:            # launcher: one worker per GPU
        gpus = args.gpus.split(",")
        bounds = [round(i * args.nsamples / len(gpus)) for i in range(len(gpus) + 1)]
        parts, procs = [], []
        for g, lo, hi in zip(gpus, bounds, bounds[1:]):
            part = out.with_suffix(f".part{lo}_{hi}.json")
            parts.append(part)
            cmd = [sys.executable, __file__, "--rules", args.rules, "--model-id", args.model_id, "--seqlen", str(args.seqlen),
                   "--nsamples", str(args.nsamples), "--seed", str(args.seed), "--samples", f"{lo}:{hi}", "--out", str(part)]
            if args.chunk:
                cmd += ["--chunk", str(args.chunk)]
            procs.append(subprocess.Popen(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": g}))
        if any(p.wait() for p in procs):
            raise SystemExit("a worker failed")
        results = [json.loads(p.read_text()) for p in parts]
        per_sample = sorted((s for r in results for s in r["per_sample"]), key=lambda s: s["index"])
        record = {**results[0], "per_sample": per_sample, "seconds": max(r["seconds"] for r in results)}
        for p in parts:
            p.unlink()
    else:
        ids = load_samples(args.model_id, args.seqlen, args.nsamples, args.seed)
        lo, hi = (int(v) for v in args.samples.split(":")) if args.samples else (0, ids.shape[0])
        model = load_model(args.model_id, args.seqlen)
        table = []
        if rules is not None:
            handle = patch(model, rules, chunk=args.chunk, dry_run=args.dry_run)
            table = handle.table
        if args.dry_run:
            return
        t0, per_sample = time.time(), []
        for i in range(lo, hi):
            nll, n = sample_nll(model, ids[i])
            per_sample.append({"index": i, "nll": nll, "tokens": n})
            print(f"  [{args.rules}] sample {i + 1}/{ids.shape[0]}  nll/token {nll / n:.6f}  {time.time() - t0:.0f}s", flush=True)
        schemes = {s.name: describe(s) for _, s in (rules or []) if s is not None}
        rule_list = [[describe(sel), None if s is None else s.name] for sel, s in (rules or [])]
        record = {"rules": args.rules, "model": args.model_id, "seqlen": args.seqlen, "nsamples": args.nsamples, "seed": args.seed,
                  "mxq_commit": commit(), "rule_list": rule_list, "schemes": schemes, "layers": table,
                  "per_sample": per_sample, "seconds": time.time() - t0}

    total, tokens = 0.0, 0
    for s in record["per_sample"]:                                      # same order of addition as MXQuant's loop
        total += s["nll"]; tokens += s["tokens"]
    record["perplexity"] = math.exp(total / tokens)
    out.write_text(json.dumps(record, indent=1))
    if not args.samples:
        print(f"PERPLEXITY [{args.rules}] : {record['perplexity']:.6f}   ({len(record['per_sample'])} samples, {record['seconds']:.0f}s)  -> {out}")


if __name__ == "__main__":
    main()
