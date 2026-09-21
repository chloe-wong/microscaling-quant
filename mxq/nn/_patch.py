"""patch(model, rules): choose, layer by layer or by layer type, which Scheme each nn.Linear runs through.

    rules = [("lm_head", None),                 # a layer name; * is a wildcard.  None = leave the layer as it is
             ("*.mlp.down_proj", FP6),          # a Scheme
             (nn.Linear, FP8)]                  # a layer type; or a function (name, module) -> bool
    handle = patch(model, rules)                # first matching rule wins, top to bottom
    print(handle)                               # every Linear, the rule it got, its Scheme
    handle.revert()                             # put the original layers back
    patch(model, rules, dry_run=True)           # print the same table, change nothing

A rule that ends up choosing no layer raises (a typo, or a rule hidden behind an earlier one). A Linear that no
rule matches is left alone and shows in the table as "no rule". Before anything is replaced, each Scheme is run
once on a tiny matmul, so a schedule of the wrong length or a misspelt format fails here and not an hour into a run.
"""
from fnmatch import fnmatchcase
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from ..scheme import Scheme
from ._linear import MXLinear

__all__ = ["patch"]

Selector = Union[str, type, Callable[[str, nn.Module], bool]]
Rule = Tuple[Selector, Optional[Scheme]]


def _matches(selector: Selector, name: str, module: nn.Module) -> bool:
    if isinstance(selector, str):
        return fnmatchcase(name, selector)
    if isinstance(selector, type):
        return isinstance(module, selector)
    return bool(selector(name, module))


class Handle:
    """What patch did: `table` rows are (layer name, in, out, rule index or None, Scheme name or None)."""

    def __init__(self, model: nn.Module, table: List[tuple], replaced: List[Tuple[str, nn.Module]]):
        self._model, self.table, self._replaced = model, table, replaced

    def revert(self) -> None:
        for name, original in self._replaced:
            parent, _, leaf = name.rpartition(".")
            setattr(self._model.get_submodule(parent) if parent else self._model, leaf, original)
        self._replaced = []

    def __str__(self) -> str:
        width = max([len(r[0]) for r in self.table] + [5])
        lines = [f"{'layer':{width}s}  {'in':>6s} {'out':>6s}  rule  scheme"]
        for name, k, n, rule, scheme in self.table:
            lines.append(f"{name:{width}s}  {k:6d} {n:6d}  {'-' if rule is None else rule:>4}  "
                         f"{'no rule' if rule is None else (scheme or 'left as is')}")
        return "\n".join(lines)


def patch(model: nn.Module, rules: Sequence[Rule], chunk: Optional[int] = None, dry_run: bool = False) -> Handle:
    for i, rule in enumerate(rules):
        if len(rule) != 2 or not (rule[1] is None or isinstance(rule[1], Scheme)):
            raise ValueError(f"rule {i} must be (selector, Scheme or None), got {rule!r}")

    chosen = []                                                   # (name, module, rule index or None)
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or not name:
            continue
        hit = next((i for i, (selector, _) in enumerate(rules) if _matches(selector, name, module)), None)
        chosen.append((name, module, hit))
    unused = [i for i in range(len(rules)) if all(hit != i for _, _, hit in chosen)]
    if unused:
        raise ValueError(f"rules {unused} choose no Linear layer (misspelt, or hidden behind an earlier rule): "
                         f"{[rules[i][0] for i in unused]}")

    for scheme in {id(s): s for _, s in rules if s is not None}.values():     # fail now, not mid-run
        g = torch.Generator().manual_seed(0)
        scheme.matmul(torch.randn(32, 2, generator=g), torch.randn(32, 2, generator=g))

    table = [(name, m.in_features, m.out_features, hit, None if hit is None or rules[hit][1] is None else rules[hit][1].name)
             for name, m, hit in chosen]
    replaced = []
    if not dry_run:
        for name, module, hit in chosen:
            if hit is None or rules[hit][1] is None:
                continue
            parent, _, leaf = name.rpartition(".")
            setattr(model.get_submodule(parent) if parent else model, leaf, MXLinear(module, rules[hit][1], chunk))
            replaced.append((name, module))
    handle = Handle(model, table, replaced)
    if dry_run:
        print(handle)
    return handle
