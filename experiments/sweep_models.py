"""One quantization config across several models: the model as loaded in bf16, then the tapeout chain.

    python experiments/sweep_models.py --gpus 0,1,2,3              # the whole slate
    python experiments/sweep_models.py --models Qwen/Qwen2.5-0.5B  # one model
    python experiments/sweep_models.py --dry-run                   # which layer gets which Scheme, per model

Two runs per model, both through experiments/llm_ppl.py, so the sampling, the loss and the layer set are the
ones every other number here was produced with:

    none                  the model as loaded, in bf16. The reference.
    hw_fp8_tapeout_rne    MXFP8 E4M3 operands, MX-Gemmini arithmetic, the tapeout ladder, and the operand
                          rounding the RTL has used since 2026-09-10.

Before either pass, every model's layer set is checked from its config alone, on the meta device, so nothing
is downloaded and nothing is held in memory. That is what catches a model whose attention cannot be found, and
it catches it in seconds rather than after an hour of arithmetic. Then the bf16 pass runs for every model,
because it takes minutes and it is where a model that will not load shows up, and only the survivors go on to
the quantized pass, which takes hours. Within each pass the cheapest model goes first, so the table fills from
the top.

Results go to experiments/results/sweep/<model>/<rules>.json, one file per run. A run whose file already
exists is skipped, so an interrupted sweep resumes and nothing is recomputed or overwritten.

A model whose attention module does not expose q_proj and k_proj will stop with "rules [0] choose no Linear
layer". That is correct and deliberate: the layer set would otherwise be silently wrong. Models that fuse QKV
into one matrix, such as GPT-NeoX and Falcon, need a different selector, not a bigger run.
"""
import argparse, json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

#: (model id, why it is on the slate). One axis each; a model that does not add an axis is not here.
MODELS = [
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0",
     "the control. Every number we already have is on this model, so it is how we know the sweep agrees "
     "with the runs that came before it"),
    ("Qwen/Qwen2.5-0.5B",
     "the smallest useful model, and the most quantization-sensitive. Its 151936-token vocabulary makes "
     "lm_head about a third of the quantized work, which stresses the one non-MLP layer we touch"),
    ("Qwen/Qwen2.5-1.5B",
     "the same family at three times the size: isolates scale from family"),
    ("HuggingFaceTB/SmolLM2-1.7B",
     "the same size as Qwen2.5-1.5B from a different family and training set: isolates family from scale"),
    ("microsoft/phi-2",
     "a two-matrix MLP instead of a gated three-matrix one, and known for large activation outliers, which "
     "is the classic way a block-scaled format fails"),
    ("mistralai/Mistral-7B-v0.3",
     "the scale a deployment sits at, with grouped-query attention at width 4096"),
    ("Qwen/Qwen2.5-7B",
     "the third point of a 0.5B, 1.5B, 7B curve inside one family"),
    ("meta-llama/Llama-2-7b-hf",
     "the reference point the quantization literature uses. Nearly every published WikiText-2 perplexity is "
     "on this model, so it is how an outside reader places our number"),
    ("meta-llama/Llama-3.1-8B",
     "a modern llama, with a 128256-token vocabulary and grouped-query attention. Paired with Llama-2-7B it "
     "separates generation from size"),
    ("meta-llama/Llama-2-13b-hf",
     "7B to 13B inside one family. The largest that fits one L40S in bf16 with room left for the simulator"),
]

#: Checked and rejected, so the same ground is not covered twice:
#:   DeepSeek-V2-Lite, DeepSeek-V3   multi-head latent attention: the projections are q_a_proj, q_b_proj,
#:                                   kv_a_proj_with_mqa and kv_b_proj, so there is no q_proj or k_proj for
#:                                   is_attention to find, and the pre-flight drops them. Supporting them
#:                                   means a selector for MLA, not a bigger run.
#:   deepseek-llm-7b-base            llama-shaped and otherwise eligible, but the repo ships only .bin
#:                                   weights and llm_ppl loads with use_safetensors=True.
#:   GPT-NeoX, Pythia, Falcon        fused query_key_value, same problem as MLA.


def slug(model_id):
    return model_id.replace("/", "_")


def cost(model_id):
    """Roughly how long a quantized run will take, as multiply-accumulates per token in the layers we
    quantize: the MLP matrices plus lm_head. Used only to order the slate. None if the config is unreadable."""
    try:
        from transformers import AutoConfig
        c = AutoConfig.from_pretrained(model_id)
    except Exception:
        return None
    h = getattr(c, "hidden_size", 0)
    i = getattr(c, "intermediate_size", None) or getattr(c, "ffn_dim", None) or 4 * h
    n = 3 if getattr(c, "hidden_act", "") in ("silu", "swish") and c.model_type not in ("opt", "phi") else 2
    return getattr(c, "num_hidden_layers", 0) * n * h * i + h * getattr(c, "vocab_size", 0)


def check_layers(model_id, rules):
    """Which Linears would be quantized, answered from the config alone on the meta device: no weights are
    downloaded and none are held. Raises if the rule list does not fit the model, which is the point."""
    import contextlib, io
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from experiments import recipes
    from mxq.nn import patch
    cfg = AutoConfig.from_pretrained(model_id)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        handle = patch(model, recipes.RULES[rules], dry_run=True)
    return sum(1 for *_, scheme in handle.table if scheme is not None), len(handle.table), handle


def run(model_id, rules, args):
    """One llm_ppl run. Returns (record, error); an existing result is returned without recomputing it."""
    out = Path(args.out_dir) / slug(model_id) / f"{rules}.json"
    if out.exists() and not args.force:
        print(f"[skip] {model_id} [{rules}] already in {out}", flush=True)
        return json.loads(out.read_text()), None
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(HERE / "llm_ppl.py"), "--rules", rules, "--model-id", model_id,
           "--out", str(out), "--nsamples", str(args.nsamples), "--seed", str(args.seed),
           "--seqlen", str(args.seqlen)]
    if args.gpus:
        cmd += ["--gpus", args.gpus]
    if args.dry_run:
        cmd += ["--dry-run"]
    print(f"\n=== {model_id}  [{rules}] ===", flush=True)
    code = subprocess.run(cmd).returncode
    if args.dry_run:
        return None, None
    if code != 0 or not out.exists():
        return None, f"llm_ppl exited {code}"
    return json.loads(out.read_text()), None


def report(rows, dropped, layers, args):
    """The table, rewritten after every run so a long sweep can be read while it is still going."""
    names = list(rows) + list(dropped)
    w = max([len(n) for n in names] + [5])
    head = (f"{'model':{w}s}  {'layers':>9s}  {'bf16':>10s}  {'quantized':>10s}  {'delta':>8s}  "
            f"{'percent':>8s}  {'minutes':>8s}")
    lines = [head, "-" * len(head)]
    for m, r in rows.items():
        b = r["bf16"]["perplexity"]
        on, total = layers.get(m, (0, 0))
        tag = f"{on}/{total}"
        q = r.get("quant")
        if q is None:
            lines.append(f"{m:{w}s}  {tag:>9s}  {b:10.6f}  {'pending':>10s}")
            continue
        p = q["perplexity"]
        lines.append(f"{m:{w}s}  {tag:>9s}  {b:10.6f}  {p:10.6f}  {p - b:8.4f}  {100 * (p - b) / b:7.2f}%  "
                     f"{q['seconds'] / 60:8.1f}")
    for m, e in dropped.items():
        lines.append(f"{m:{w}s}  {e}")
    text = "\n".join(lines)
    print("\n" + text + "\n", flush=True)
    d = Path(args.out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.txt").write_text(f"{args.quant_rules}, {args.nsamples} samples of {args.seqlen} tokens, "
                                   f"seed {args.seed}\n\n{text}\n")
    (d / "summary.json").write_text(json.dumps(
        {"rules": args.quant_rules, "nsamples": args.nsamples, "seqlen": args.seqlen, "seed": args.seed,
         "models": {m: {"layers_quantized": layers.get(m, (0, 0))[0], "layers_total": layers.get(m, (0, 0))[1],
                        **{k: {"perplexity": v["perplexity"], "seconds": v["seconds"],
                               "commit": v["mxq_commit"]} for k, v in r.items()}}
                    for m, r in rows.items()},
         "dropped": dropped}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=None, help="comma-separated model ids (default: the slate above)")
    ap.add_argument("--quant-rules", default="hw_fp8_tapeout_rne", help="a key of experiments.recipes.RULES")
    ap.add_argument("--nsamples", type=int, default=16)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpus", default=None, help="e.g. 0,1,2,3: samples split between them, per run")
    ap.add_argument("--out-dir", default=str(HERE / "results" / "sweep"))
    ap.add_argument("--force", action="store_true", help="recompute runs that already have a result file")
    ap.add_argument("--dry-run", action="store_true",
                    help="check every model's layer set from its config and print it; download and run nothing")
    ap.add_argument("--why", action="store_true", help="print the slate and why each model is on it")
    args = ap.parse_args()

    if args.why:
        for m, why in MODELS:
            print(f"{m}\n    {why}\n")
        return

    models = [m.strip() for m in args.models.split(",")] if args.models else [m for m, _ in MODELS]
    order = {m: cost(m) for m in models}
    models.sort(key=lambda m: order[m] if order[m] else float("inf"))

    rows, dropped, layers = {}, {}, {}

    print(f"pre-flight: the layer set of {len(models)} models, from their configs alone")
    surviving = []
    for m in models:
        try:
            on, total, handle = check_layers(m, args.quant_rules)
        except Exception as e:
            dropped[m] = f"layer set: {type(e).__name__}: {str(e)[:110]}"
            print(f"  {m:42s} DROPPED  {dropped[m]}", flush=True)
            continue
        layers[m] = (on, total)
        surviving.append(m)
        print(f"  {m:42s} {on:4d} of {total:4d} Linears quantized", flush=True)
        if args.dry_run:
            print(handle)
    models = surviving
    if args.dry_run:
        return

    print(f"\npass 1 of 2: bf16 reference, {len(models)} models")
    for m in models:
        rec, err = run(m, "none", args)
        if err:
            dropped[m] = err
        else:
            rows[m] = {"bf16": rec}
    report(rows, dropped, layers, args)

    print(f"pass 2 of 2: {args.quant_rules}, {len(rows)} models")
    for m in list(rows):
        rec, err = run(m, args.quant_rules, args)
        if err:
            dropped[m] = err
        else:
            rows[m]["quant"] = rec
        report(rows, dropped, layers, args)


if __name__ == "__main__":
    main()
