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

def array(object: Any, dtype: Optional[Any] = None, *, backend_name: Optional[str] = None, **kwargs) -> Any:
    """
    Create an array from any object with optional dtype conversion.
    """
    backend_mod = _get_backend(backend_name)
    return backend_mod.array(object, dtype=dtype, **kwargs)

def safe_array(object: Any, dtype: Optional[Any] = None, *, backend_name: Optional[str] = None, **kwargs) -> Any:
    """
    Safe array creation with dtype coercion and fallback to NumPy.
    """
    try:
        return array(object, dtype=dtype, backend_name=backend_name, **kwargs)
    except Exception:
        return _np.array(object, dtype=dtype)

def zeros(shape: Sequence[int], dtype: Any = float, *, backend_name: Optional[str] = None) -> Any:
    """Create a zero-filled array."""
    return _get_backend(backend_name).zeros(shape, dtype=dtype)

def safe_zeros(shape: Sequence[int], dtype: Any = float, *, backend_name: Optional[str] = None) -> Any:
    """Safe zero array creation."""
    try:
        return zeros(shape, dtype=dtype, backend_name=backend_name)
    except Exception:
        return _np.zeros(shape, dtype=dtype)

def ones(shape: Sequence[int], dtype: Any = float, *, backend_name: Optional[str] = None) -> Any:
    """Create a one-filled array."""
    return _get_backend(backend_name).ones(shape, dtype=dtype)

def safe_ones(shape: Sequence[int], dtype: Any = float, *, backend_name: Optional[str] = None) -> Any:
    """Safe one array creation."""
    try:
        return ones(shape, dtype=dtype, backend_name=backend_name)
    except Exception:
        return _np.ones(shape, dtype=dtype)
