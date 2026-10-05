"""mxq.nn — putting Schemes into a model.

    MXLinear(linear, scheme)     one nn.Linear computed through one Scheme (inference only)
    patch(model, rules)          choose a Scheme per layer name or per layer type; first matching rule wins
    is_attention                 a selector for the attention projections, by what the parent holds
    attend(q, k, v, mask, scale, qk, pv)   the attention core with Q·Kᵀ and P·V through two Schemes; patch() routes
                                 attention modules to it with a (qk, pv) rule

mxq.nn.torchao (optional, needs torchao; not imported here): MXQConfig, the same MXLinear behind torchao.quantize_
and Hugging Face TorchAoConfig.
"""
from ._attention import attend
from ._linear import MXLinear
from ._patch import is_attention, patch

__all__ = ["MXLinear", "patch", "is_attention", "attend"]
