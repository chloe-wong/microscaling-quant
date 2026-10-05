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
"""
from fnmatch import fnmatchcase
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from ..scheme import Scheme
from ._attention import NAME as _CORE, attend, register
from ._linear import MXLinear

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
    scheme.matmul(torch.randn(32, n, generator=g), torch.randn(32, n, generator=g))


def _smoke_core(qk: Scheme, pv: Scheme) -> None:
    g = torch.Generator().manual_seed(0)
    t = 64 * max(qk.rows, pv.rows)                                              # whole 32-blocks of keys, whole groups
    q, k, v = (torch.randn(1, n, t, 32, generator=g) for n in (2, 1, 1))
    attend(q, k, v, None, 32 ** -0.5, qk, pv)


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
                 previous: Optional[str] = None):
        self._model, self.table, self._replaced = model, table, replaced
        self.cores, self._cored, self._previous = cores or [], cored or [], previous

    def revert(self) -> None:
        for parent, leaf, original in self._replaced:
            setattr(parent, leaf, original)
        self._replaced = []
        for module in self._cored:
            del module._mxq_core
        if self._cored:
            _set_attention(self._model, self._previous)
        self._cored = []

    def __str__(self) -> str:
        width = max([len(r[0]) for r in self.table] + [len(r[0]) for r in self.cores] + [5])
        lines = [f"{'layer':{width}s}  {'in':>6s} {'out':>6s}  rule  scheme"]
        for name, k, n, rule, scheme in self.table:
            lines.append(f"{name:{width}s}  {k:6d} {n:6d}  {'-' if rule is None else rule:>4}  "
                         f"{'no rule' if rule is None else (scheme or 'left as is')}")
        if any(rule is not None for _, rule, _ in self.cores):
            lines.append(f"{'attention core':{width}s}  {'':13s}  rule  Q·Kᵀ, P·V")
            for name, rule, names in self.cores:
                lines.append(f"{name:{width}s}  {'':13s}  {'-' if rule is None else rule:>4}  "
                             f"{'sdpa (no rule)' if names is None else ', '.join(names)}")
        return "\n".join(lines)


def patch(model: nn.Module, rules: Sequence[Rule], chunk: Optional[int] = None, dry_run: bool = False,
          cache_weights: bool = True) -> Handle:
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

    for i in linear_rules:                                                      # fail now, not mid-run
        if rules[i][1] is not None:
            _smoke(rules[i][1])
    for i in core_rules:
        _smoke_core(*rules[i][1])

    table = [(name, m.in_features, m.out_features, hit,
              None if hit is None or rules[hit][1] is None else rules[hit][1].name)
             for name, m, _, hit in chosen]
    core_table = [(name, hit, None if hit is None else tuple(s.name for s in rules[hit][1]))
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
    if not dry_run and any(hit is not None for _, _, hit in cores):
        register()
        previous = model.config._attn_implementation
        for _, module, hit in {id(m): (n, m, h) for n, m, h in cores if h is not None}.values():
            module._mxq_core = rules[hit][1]
            cored.append(module)
        _set_attention(model, _CORE)
    handle = Handle(model, table, replaced, core_table, cored, previous)
    if dry_run:
        print(handle)
    return handle
