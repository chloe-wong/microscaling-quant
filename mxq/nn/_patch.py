"""patch(model, rules): choose, layer by layer or by layer type, which Scheme each nn.Linear runs through.

    rules = [("lm_head", None),                 # a layer name; * is a wildcard.  None = leave the layer as it is
             ("*.mlp.down_proj", FP6),          # a Scheme
             (is_attention, None),              # a function (name, module, parent) -> bool
             (nn.Linear, FP8)]                  # a layer type
    handle = patch(model, rules)                # first matching rule wins, top to bottom
    print(handle)                               # every Linear, the rule it got, its Scheme
    handle.revert()                             # put the original layers back
    patch(model, rules, dry_run=True)           # print the same table, change nothing

A rule that ends up choosing no layer raises (a typo, or a rule hidden behind an earlier one). A Linear that no
rule matches is left alone and shows in the table as "no rule". Before anything is replaced, each Scheme is run
once on a tiny matmul, so a schedule of the wrong length or a misspelt format fails here and not an hour into a run.

One Linear can be reachable under several names, because a model may hold the same module in two places. Every
such name is listed and every attribute holding it is replaced, so the table counts what the model will actually
run. Two names for one attribute that ask for different Schemes is ambiguous and raises.

The attention core (S = Q·Kᵀ, softmax, O = P·V) is not a Linear. A rule whose value is a pair of Schemes
`(qk, pv)` chooses attention modules instead (a module holding q_proj and k_proj; the selector sees its name, the
module and its parent) and runs their core through mxq.nn.attend:

    rules = [("model.layers.*.self_attn", (FP8, FP8)),   # Q·Kᵀ through the first, P·V through the second
             (is_attention, None),                       # q/k/v/o_proj: still Linear rules
             (nn.Linear, FP8)]

The model is switched to the "mxq" attention implementation (transformers' AttentionInterface, with eager's
additive mask); an attention module no core rule chose runs sdpa as before. With no core rule, nothing about
attention changes. `revert` undoes both kinds.

The vector ops around the matmuls (softmax, RMSNorm) run as transformers computes them unless `vector` says
otherwise (mxq.nn._vector has what each setting rounds and why):

    patch(model, rules, vector="bf16")                      # softmax and every RMSNorm: each step rounded to bf16
    patch(model, rules, vector={"rmsnorm": "bf16"})         # one op; an op left out stays as transformers has it

softmax: every attention module runs through attend with that setting. A module no core rule chose gets the
`exact` core (no quantization, float64 sums), because sdpa computes softmax inside one fused kernel where no step
can be rounded. rmsnorm: the forward of every module whose class name ends in RMSNorm is replaced, after a check
that the replacement with no rounding returns the module's own output bit for bit; a module computing anything
else is refused by name. `revert` undoes these too.
"""
from fnmatch import fnmatchcase
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from ..scheme import Scheme
from ._attention import NAME as _CORE, attend, register
from ._linear import MXLinear
from ._vector import EXACT, resolve, rmsnorm

__all__ = ["patch", "is_attention"]

Selector = Union[str, type, Callable[[str, nn.Module, nn.Module], bool]]
Rule = Tuple[Selector, Union[None, Scheme, Tuple[Scheme, Scheme]]]


def _is_core(value) -> bool:
    return isinstance(value, tuple) and len(value) == 2 and all(isinstance(s, Scheme) for s in value)


def _holds_core(module: nn.Module) -> bool:
    """An attention module: it holds q_proj and k_proj (what is_attention asks of a projection's parent)."""
    return hasattr(module, "q_proj") and hasattr(module, "k_proj")


def _smoke(scheme: Scheme) -> None:
    g = torch.Generator().manual_seed(0)
    n = 2 * scheme.rows                                                         # a LUT activation: two whole groups
    K = 3 * scheme.block_size                                                   # three blocks: matmul checks the scales
    scheme.matmul(torch.randn(K, n, generator=g), torch.randn(K, n, generator=g))


def _smoke_core(qk: Scheme, pv: Scheme) -> None:
    g = torch.Generator().manual_seed(0)
    t = 2 * max(qk.block_size, pv.block_size) * max(qk.rows, pv.rows)          # whole key blocks, whole groups
    q, k, v = (torch.randn(1, n, t, 32, generator=g) for n in (2, 1, 1))
    attend(q, k, v, None, 32 ** -0.5, qk, pv)


def _is_rmsnorm(module: nn.Module) -> bool:
    return type(module).__name__.endswith("RMSNorm")


def _eps(module: nn.Module) -> float:
    for attr in ("variance_epsilon", "eps"):                                    # Llama and most; Phi-3, others
        if isinstance(getattr(module, attr, None), float):
            return getattr(module, attr)
    raise ValueError(f"{type(module).__name__}: no float variance_epsilon or eps attribute")


def _check_rmsnorm(name: str, module: nn.Module) -> float:
    """The replacement with no rounding must give the module's own output, bit for bit. Returns eps."""
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 1:
        raise ValueError(f"{name} ({type(module).__name__}): no 1-D weight, not the RMSNorm mxq knows")
    eps = _eps(module)
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(3, weight.shape[0], generator=g) * 3).to(device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        theirs, ours = module(x), rmsnorm(x, weight, eps, None)
    if theirs.dtype != ours.dtype or not torch.equal(theirs, ours):
        raise ValueError(f"{name} ({type(module).__name__}) does not compute weight · x / sqrt(mean(x²) + eps) the way "
                         "transformers' LlamaRMSNorm does; vector rmsnorm cannot replace it")
    return eps


def _rmsnorm_forward(module: nn.Module, eps: float, precision: str, x: torch.Tensor) -> torch.Tensor:
    return rmsnorm(x, module.weight, eps, precision)


def _set_attention(model: nn.Module, name: str) -> None:
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation(name)
    else:
        model.config._attn_implementation = name


def is_attention(name: str, module: nn.Module, parent: nn.Module) -> bool:
    """Is this Linear one of an attention module's projections?

    True when the module holding it has both `q_proj` and `k_proj`. That is the test MXQuant's eval_complete.py
    uses to find attention, so a rule list written with it keeps MXQuant's layer set exactly. It asks what the
    parent is, not what it is called, so it holds for any model that names its projections the usual way and no
    model-specific string has to appear in a rule list.
    """
    return hasattr(parent, "q_proj") and hasattr(parent, "k_proj")


def _matches(selector: Selector, name: str, module: nn.Module, parent: nn.Module) -> bool:
    if isinstance(selector, str):
        return fnmatchcase(name, selector)
    if isinstance(selector, type):
        return isinstance(module, selector)
    return bool(selector(name, module, parent))


class Handle:
    """What patch did: `table` rows are (layer name, in, out, rule index or None, Scheme name or None);
    `cores` rows are (attention module name, rule index or None, (qk name, pv name) or None)."""

    def __init__(self, model: nn.Module, table: List[tuple], replaced: List[Tuple[nn.Module, str, nn.Module]],
                 cores: Optional[List[tuple]] = None, cored: Optional[List[nn.Module]] = None,
                 previous: Optional[str] = None, vector: Optional[dict] = None, norms: Optional[List[str]] = None,
                 normed: Optional[List[nn.Module]] = None):
        self._model, self.table, self._replaced = model, table, replaced
        self.cores, self._cored, self._previous = cores or [], cored or [], previous
        self.vector, self.norms, self._normed = vector or resolve(None), norms or [], normed or []

    def revert(self) -> None:
        for parent, leaf, original in self._replaced:
            setattr(parent, leaf, original)
        self._replaced = []
        for module in self._cored:
            del module._mxq_core
            if hasattr(module, "_mxq_vector"):
                del module._mxq_vector
        if self._cored:
            _set_attention(self._model, self._previous)
        self._cored = []
        for module in self._normed:
            del module.forward                                                  # the class's forward again
        self._normed = []

    def __str__(self) -> str:
        width = max([len(r[0]) for r in self.table] + [len(r[0]) for r in self.cores] + [5])
        lines = [f"{'layer':{width}s}  {'in':>6s} {'out':>6s}  rule  scheme"]
        for name, k, n, rule, scheme in self.table:
            lines.append(f"{name:{width}s}  {k:6d} {n:6d}  {'-' if rule is None else rule:>4}  "
                         f"{'no rule' if rule is None else (scheme or 'left as is')}")
        if any(names is not None for _, _, names in self.cores):
            lines.append(f"{'attention core':{width}s}  {'':13s}  rule  Q·Kᵀ, P·V")
            for name, rule, names in self.cores:
                lines.append(f"{name:{width}s}  {'':13s}  {'-' if rule is None else rule:>4}  "
                             f"{'sdpa (no rule)' if names is None else ', '.join(names)}"
                             f"{'  (vector softmax)' if rule is None and names is not None else ''}")
        if any(self.vector.values()):
            n = {"softmax": sum(names is not None for _, _, names in self.cores), "rmsnorm": len(self.norms)}
            lines.append("vector  " + ", ".join(f"{op} {p or 'as transformers'}" + (f" ({n[op]} modules)" if p else "")
                                                for op, p in self.vector.items()))
        return "\n".join(lines)


def patch(model: nn.Module, rules: Sequence[Rule], chunk: Optional[int] = None, dry_run: bool = False,
          cache_weights: bool = True, vector=None) -> Handle:
    vector = resolve(vector)
    for i, rule in enumerate(rules):
        if len(rule) != 2 or not (rule[1] is None or isinstance(rule[1], Scheme) or _is_core(rule[1])):
            raise ValueError(f"rule {i} must be (selector, Scheme or None or (qk Scheme, pv Scheme)), got {rule!r}")
    linear_rules = [i for i, (_, value) in enumerate(rules) if not _is_core(value)]
    core_rules = [i for i, (_, value) in enumerate(rules) if _is_core(value)]

    # remove_duplicate=False so a module held in two places is seen under both names, not just the first.
    walk = list(model.named_modules(remove_duplicate=False))
    holder = {name: module for name, module in walk}

    chosen = []                                                   # (name, module, parent, rule index or None)
    for name, module in walk:
        if not isinstance(module, nn.Linear) or not name:
            continue
        parent = holder[name.rpartition(".")[0]]
        hit = next((i for i in linear_rules if _matches(rules[i][0], name, module, parent)), None)
        chosen.append((name, module, parent, hit))
    cores = []                                                    # (name, module, rule index or None)
    for name, module in walk:
        if not name or not _holds_core(module):
            continue
        parent = holder[name.rpartition(".")[0]]
        hit = next((i for i in core_rules if _matches(rules[i][0], name, module, parent)), None)
        cores.append((name, module, hit))
    unused = [i for i in linear_rules if all(hit != i for _, _, _, hit in chosen)]
    if unused:
        raise ValueError(f"rules {unused} choose no Linear layer (misspelt, or hidden behind an earlier rule): "
                         f"{[rules[i][0] for i in unused]}")
    unused = [i for i in core_rules if all(hit != i for _, _, hit in cores)]
    if unused:
        raise ValueError(f"rules {unused} choose no attention module (one holding q_proj and k_proj): "
                         f"{[rules[i][0] for i in unused]}")
    for module in {id(m): m for _, m, hit in cores if hit is not None}.values():
        if len({hit for _, m, hit in cores if m is module}) > 1:
            raise ValueError(f"{[n for n, m, _ in cores if m is module]} are the same attention module but match "
                             "different rules; one of them has to go")

    # An attribute of a module is one place a layer runs from. Several names can reach one attribute when a
    # whole subtree is held twice; they are one site and must agree on what to do with it.
    sites: dict = {}
    for name, module, parent, hit in chosen:
        sites.setdefault((id(parent), name.rpartition(".")[2]), []).append((name, module, parent, hit))
    for rows in sites.values():
        if len({r[3] for r in rows}) > 1:
            raise ValueError(f"{[r[0] for r in rows]} are the same layer but match different rules "
                             f"{[r[3] for r in rows]}; one of them has to go")

    if vector["softmax"] and not cores:
        raise ValueError("vector softmax: the model has no attention module (one holding q_proj and k_proj)")
    norms = list({id(m): (n, m) for n, m in walk if n and _is_rmsnorm(m)}.values()) if vector["rmsnorm"] else []
    if vector["rmsnorm"] and not norms:
        raise ValueError("vector rmsnorm: the model has no module whose class name ends in RMSNorm")

    for i in linear_rules:                                                      # fail now, not mid-run
        if rules[i][1] is not None:
            _smoke(rules[i][1])
    for i in core_rules:
        _smoke_core(*rules[i][1])
    if vector["softmax"] and any(hit is None for _, _, hit in cores):
        _smoke_core(EXACT, EXACT)
    eps = [_check_rmsnorm(n, m) for n, m in norms]

    table = [(name, m.in_features, m.out_features, hit,
              None if hit is None or rules[hit][1] is None else rules[hit][1].name)
             for name, m, _, hit in chosen]
    def core_of(hit):                                                           # (qk, pv), or None for sdpa
        return rules[hit][1] if hit is not None else (EXACT, EXACT) if vector["softmax"] else None
    core_table = [(name, hit, None if core_of(hit) is None else tuple(s.name for s in core_of(hit)))
                  for name, _, hit in cores]
    replaced = []
    if not dry_run:
        for rows in sites.values():
            name, module, parent, hit = rows[0]
            if hit is None or rules[hit][1] is None:
                continue
            leaf = name.rpartition(".")[2]
            setattr(parent, leaf, MXLinear(module, rules[hit][1], chunk, cache_weights))
            replaced.append((parent, leaf, module))
    cored, previous = [], None
    if not dry_run and any(core_of(hit) is not None for _, _, hit in cores):
        register()
        previous = model.config._attn_implementation
        for _, module, hit in {id(m): (n, m, h) for n, m, h in cores if core_of(h) is not None}.values():
            module._mxq_core = core_of(hit)
            if vector["softmax"]:
                module._mxq_vector = vector["softmax"]
            cored.append(module)
        _set_attention(model, _CORE)
    normed = []
    if not dry_run:
        for (_, module), e in zip(norms, eps):
            module.forward = partial(_rmsnorm_forward, module, e, vector["rmsnorm"])
            normed.append(module)
    handle = Handle(model, table, replaced, core_table, cored, previous, vector, [n for n, _ in norms], normed)
    if dry_run:
        print(handle)
    return handle
