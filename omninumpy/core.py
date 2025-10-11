"""
Core compatibility layer for OmniNumpy.

This module provides a NumPy-compatible interface that works across different backends
and handles NumPy 2.x compatibility issues.
"""

import sys
import warnings
from typing import Any, Union, Optional, Sequence

# ---------------------------------------------------------------------
# Backend loader
# ---------------------------------------------------------------------

def _get_backend():
    """Get the current backend module."""
    from . import backend as backend_module
    return backend_module.get_current_backend_module()

# ---------------------------------------------------------------------
# NumPy 2.x compatibility
# ---------------------------------------------------------------------

class NumPy2CompatibilityMixin:
    """Mixin class to handle NumPy 2.x breaking changes."""
    
    @staticmethod
    def _handle_deprecated_params(**kwargs):
        """Handle deprecated parameters that changed in NumPy 2.x."""
        if 'keepdims' in kwargs and 'keepdim' not in kwargs:
            # Some backends might use keepdim instead of keepdims
            pass
        return kwargs

# ---------------------------------------------------------------------
# Optional: import public API from individual core modules
# ---------------------------------------------------------------------

from .core.array_core import array, zeros, ones, empty
from .core.math_core import add, subtract, multiply, divide
from .core.reduce_core import sum, mean
from .core.shape_core import reshape, transpose
from .core.linalg_core import dot, matmul, linalg
from .core.random_core import random, randn
from .core.compat_core import emulate
from .core.dtype_core import (
    int8, int16, int32, int64,
    float16, float32, float64,
    complex64, complex128,
    bool_, pi, e
)

__all__ = [
    "array", "zeros", "ones", "empty",
    "add", "subtract", "multiply", "divide",
    "sum", "mean",
    "reshape", "transpose",
    "dot", "matmul", "linalg",
    "random", "randn",
    "emulate",
    "int8", "int16", "int32", "int64",
    "float16", "float32", "float64",
    "complex64", "complex128",
    "bool_", "pi", "e",
]
