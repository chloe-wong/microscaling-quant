"""mxq.nn — putting Schemes into a model.

    MXLinear(linear, scheme)     one nn.Linear computed through one Scheme (inference only)
    patch(model, rules)          choose a Scheme per layer name or per layer type; first matching rule wins
    is_attention                 a selector for the attention projections, by what the parent holds

mxq.nn.torchao (optional, needs torchao; not imported here): MXQConfig, the same MXLinear behind torchao.quantize_
and Hugging Face TorchAoConfig.
mxq.nn.operand_capture (optional TorchAO handler; not imported here): operand Q/DQ and graph capture for
targets whose software contracts approve the BF16, block-32 MX profile.
"""
from ._linear import MXLinear
from ._patch import is_attention, patch

__all__ = ["MXLinear", "patch", "is_attention"]
