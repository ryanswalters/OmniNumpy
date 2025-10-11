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

def shape(a: Any, *, backend_name: Optional[str] = None) -> Tuple[int, ...]:
    """Return shape of array."""
    return _get_backend(backend_name).shape(a)

def safe_shape(a: Any, *, backend_name: Optional[str] = None) -> Tuple[int, ...]:
    """Safe shape with fallback."""
    try:
        return shape(a, backend_name=backend_name)
    except Exception:
        return _np.shape(a)

def reshape(a: Any, newshape: Sequence[int], *, backend_name: Optional[str] = None) -> Any:
    """Reshape an array."""
    return _get_backend(backend_name).reshape(a, newshape)

def safe_reshape(a: Any, newshape: Sequence[int], *, backend_name: Optional[str] = None) -> Any:
    """Safe reshape with fallback."""
    try:
        return reshape(a, newshape, backend_name=backend_name)
    except Exception:
        return _np.reshape(a, newshape)

def transpose(a: Any, axes: Optional[Sequence[int]] = None, *, backend_name: Optional[str] = None) -> Any:
    """Transpose an array."""
    return _get_backend(backend_name).transpose(a, axes)

def safe_transpose(a: Any, axes: Optional[Sequence[int]] = None, *, backend_name: Optional[str] = None) -> Any:
    """Safe transpose with fallback."""
    try:
        return transpose(a, axes, backend_name=backend_name)
    except Exception:
        return _np.transpose(a, axes)
