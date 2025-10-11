"""
OmniNumpy - Safe Frontend
=========================
Exposes safe_* functions alongside normal API for drop-in compatibility.
"""

from . import backend
from . import core
from . import interface

# Export safe-aware modules
from .array_core import array, safe_array, zeros, safe_zeros, ones, safe_ones
from .dtype_core import bool, int32, float64, pi
from .linalg_core import dot, safe_dot, matmul, safe_matmul
from .math_core import add, safe_add, subtract, safe_subtract
from .random_core import random, safe_random, randn, safe_randn
from .reduce_core import sum, safe_sum, mean, safe_mean
from .shape_core import shape, safe_shape, reshape, safe_reshape, transpose, safe_transpose

__all__ = [
    "backend", "core", "interface",
    "array", "safe_array", "zeros", "safe_zeros", "ones", "safe_ones",
    "bool", "int32", "float64", "pi",
    "dot", "safe_dot", "matmul", "safe_matmul",
    "add", "safe_add", "subtract", "safe_subtract",
    "random", "safe_random", "randn", "safe_randn",
    "sum", "safe_sum", "mean", "safe_mean",
    "shape", "safe_shape", "reshape", "safe_reshape", "transpose", "safe_transpose"
]
