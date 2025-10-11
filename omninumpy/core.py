"""
Core compatibility layer (safe-first).
"""

from . import backend
from .array_core import array, safe_array, zeros, safe_zeros, ones, safe_ones
from .math_core import add, safe_add, subtract, safe_subtract
from .linalg_core import dot, safe_dot, matmul, safe_matmul
from .reduce_core import sum, safe_sum, mean, safe_mean
from .shape_core import reshape, safe_reshape, transpose, safe_transpose
from .random_core import random, safe_random, randn, safe_randn
from .dtype_core import int32, float64, pi, bool

# Default exports (safe variants encouraged)
__all__ = [
    "array", "safe_array",
    "zeros", "safe_zeros", "ones", "safe_ones",
    "add", "safe_add", "subtract", "safe_subtract",
    "dot", "safe_dot", "matmul", "safe_matmul",
    "sum", "safe_sum", "mean", "safe_mean",
    "reshape", "safe_reshape", "transpose", "safe_transpose",
    "random", "safe_random", "randn", "safe_randn",
    "int32", "float64", "pi", "bool",
]
