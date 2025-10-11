"""
OmniNumpy Core Module - Safe Version
-----------------------------------
Auto-generated with backend override, dtype/shape safety, and NumPy fallback.
"""

from __future__ import annotations
from typing import Any, Optional, Sequence, Tuple
import numpy as _np

try:
    from omninumpy import backend as _backend
except ImportError:
    _backend = None

def _get_backend(backend_name: Optional[str] = None):
    """Return the current backend module or NumPy as fallback."""
    if backend_name:
        try:
            return __import__(backend_name)
        except ImportError:
            pass
    if _backend is not None:
        try:
            return _backend.get_current_backend_module()
        except Exception:
            pass
    return _np

def dot(a: Any, b: Any, *, dtype: Optional[Any] = None, backend_name: Optional[str] = None) -> Any:
    """
    Compute the dot product of two arrays with dtype casting and shape validation.
    """
    backend_mod = _get_backend(backend_name)
    a = backend_mod.array(a, dtype=dtype)
    b = backend_mod.array(b, dtype=dtype)

    if a.ndim == 1 and b.ndim == 1 and a.shape[0] != b.shape[0]:
        raise ValueError(f"Shape mismatch for dot: {a.shape} vs {b.shape}")
    if a.ndim == 2 and b.ndim == 2 and a.shape[1] != b.shape[0]:
        raise ValueError(f"Shape mismatch for dot: {a.shape} vs {b.shape}")
    return backend_mod.dot(a, b)

def safe_dot(a: Any, b: Any, *, dtype: Optional[Any] = None, backend_name: Optional[str] = None) -> Any:
    """Safe dot product with fallback."""
    try:
        return dot(a, b, dtype=dtype, backend_name=backend_name)
    except Exception:
        return _np.dot(_np.array(a, dtype=dtype), _np.array(b, dtype=dtype))

def matmul(a: Any, b: Any, *, dtype: Optional[Any] = None, backend_name: Optional[str] = None) -> Any:
    """
    Matrix multiplication with dtype handling.
    """
    backend_mod = _get_backend(backend_name)
    a = backend_mod.array(a, dtype=dtype)
    b = backend_mod.array(b, dtype=dtype)
    return backend_mod.matmul(a, b) if hasattr(backend_mod, "matmul") else backend_mod.dot(a, b)

def safe_matmul(a: Any, b: Any, *, dtype: Optional[Any] = None, backend_name: Optional[str] = None) -> Any:
    """Safe matmul with fallback."""
    try:
        return matmul(a, b, dtype=dtype, backend_name=backend_name)
    except Exception:
        return _np.matmul(_np.array(a, dtype=dtype), _np.array(b, dtype=dtype))
