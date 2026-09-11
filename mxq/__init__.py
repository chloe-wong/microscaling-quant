"""mxq — granular microscaling ops.

Every quantization in this codebase goes through here. Chains and experiments call
mxq.<algorithm>.<format>(x); they never re-implement or copy the algorithm.

    mxq.ocp        OCP MX v1.0 element formats — Microsoft reference code, verbatim
    mxq.mxgemmini  (planned) our element quantizers
    mxq.block      (planned) block-32 scaling; takes an element quantizer as argument
    mxq.precision  (planned) product / accumulator precision ops
"""
from . import ocp

__all__ = ["ocp"]
