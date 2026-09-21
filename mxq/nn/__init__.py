"""mxq.nn — putting Schemes into a model.

    MXLinear(linear, scheme)     one nn.Linear computed through one Scheme (inference only)
    patch(model, rules)          choose a Scheme per layer name or per layer type; first matching rule wins
"""
from ._linear import MXLinear
from ._patch import patch

__all__ = ["MXLinear", "patch"]
