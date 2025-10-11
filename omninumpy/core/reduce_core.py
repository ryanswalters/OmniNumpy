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

def sum(a: Any, axis: Optional[int] = None, dtype: Optional[Any] = None, keepdims: bool = False, *, backend_name: Optional[str] = None) -> Any:
    """Sum array elements."""
    return _get_backend(backend_name).sum(a, axis=axis, dtype=dtype, keepdims=keepdims)

def safe_sum(a: Any, axis: Optional[int] = None, dtype: Optional[Any] = None, keepdims: bool = False, *, backend_name: Optional[str] = None) -> Any:
    """Safe sum with fallback."""
    try:
        return sum(a, axis=axis, dtype=dtype, keepdims=keepdims, backend_name=backend_name)
    except Exception:
        return _np.sum(a, axis=axis, dtype=dtype, keepdims=keepdims)

def mean(a: Any, axis: Optional[int] = None, dtype: Optional[Any] = None, keepdims: bool = False, *, backend_name: Optional[str] = None) -> Any:
    """Compute mean."""
    return _get_backend(backend_name).mean(a, axis=axis, dtype=dtype, keepdims=keepdims)

def safe_mean(a: Any, axis: Optional[int] = None, dtype: Optional[Any] = None, keepdims: bool = False, *, backend_name: Optional[str] = None) -> Any:
    """Safe mean with fallback."""
    try:
        return mean(a, axis=axis, dtype=dtype, keepdims=keepdims, backend_name=backend_name)
    except Exception:
        return _np.mean(a, axis=axis, dtype=dtype, keepdims=keepdims)
