"""mxq.nn — putting Schemes into a model.

    MXLinear(linear, scheme)     one nn.Linear computed through one Scheme (inference only)
"""
from ._linear import MXLinear

__all__ = ["MXLinear"]
