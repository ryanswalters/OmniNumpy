"""
Core compatibility layer for OmniNumpy.

This module provides a NumPy-compatible interface that works across different backends
and handles NumPy 2.x compatibility issues.
"""

import sys
import importlib
import pathlib
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
# Dynamic import of submodules inside omninumpy/core/
# ---------------------------------------------------------------------

# Path to the actual core/ package directory
_package_dir = pathlib.Path(__file__).parent / "core"

for module_path in _package_dir.glob("*.py"):
    if module_path.name.startswith("_"):
        continue
    mod_name = module_path.stem
    full_name = f"{__name__}.{mod_name}"
    if full_name not in sys.modules:
        sys.modules[full_name] = importlib.import_module(f"{__name__}.core.{mod_name}")

# Now you can import directly from this module OR its submodules:
# from omninumpy.core.array_core import array

# ---------------------------------------------------------------------
# Optional: Re-export public API symbols
# ---------------------------------------------------------------------

array = sys.modules[f"{__name__}.array_core"].array
zeros = sys.modules[f"{__name__}.array_core"].zeros
ones = sys.modules[f"{__name__}.array_core"].ones
empty = sys.modules[f"{__name__}.array_core"].empty

add = sys.modules[f"{__name__}.math_core"].add
subtract = sys.modules[f"{__name__}.math_core"].subtract
multiply = sys.modules[f"{__name__}.math_core"].multiply
divide = sys.modules[f"{__name__}.math_core"].divide

sum = sys.modules[f"{__name__}.reduce_core"].sum
mean = sys.modules[f"{__name__}.reduce_core"].mean

reshape = sys.modules[f"{__name__}.shape_core"].reshape
transpose = sys.modules[f"{__name__}.shape_core"].transpose

dot = sys.modules[f"{__name__}.linalg_core"].dot
matmul = sys.modules[f"{__name__}.linalg_core"].matmul
linalg = sys.modules[f"{__name__}.linalg_core"].linalg

random = sys.modules[f"{__name__}.random_core"].random
randn = sys.modules[f"{__name__}.random_core"].randn

emulate = sys.modules[f"{__name__}.compat_core"].emulate

dtype_mod = sys.modules[f"{__name__}.dtype_core"]
int8 = dtype_mod.int8
int16 = dtype_mod.int16
int32 = dtype_mod.int32
int64 = dtype_mod.int64
float16 = dtype_mod.float16
float32 = dtype_mod.float32
float64 = dtype_mod.float64
complex64 = dtype_mod.complex64
complex128 = dtype_mod.complex128
bool_ = dtype_mod.bool_
pi = dtype_mod.pi
e = dtype_mod.e

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
