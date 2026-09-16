"""mxq.matmul — reducers: Y = Aᵀ·B from codes and scales. See arithmetic.py."""
from .arithmetic import Arithmetic, MXQUANT, MXGEMMINI

__all__ = ["Arithmetic", "MXQUANT", "MXGEMMINI"]
