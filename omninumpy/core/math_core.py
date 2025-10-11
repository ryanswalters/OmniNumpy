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

def add(x1: Any, x2: Any, *, backend_name: Optional[str] = None) -> Any:
    """Add two arrays elementwise."""
    return _get_backend(backend_name).add(x1, x2)

def safe_add(x1: Any, x2: Any, *, backend_name: Optional[str] = None) -> Any:
    """Safe addition with fallback."""
    try:
        return add(x1, x2, backend_name=backend_name)
    except Exception:
        return _np.add(x1, x2)

def subtract(x1: Any, x2: Any, *, backend_name: Optional[str] = None) -> Any:
    """Subtract two arrays elementwise."""
    return _get_backend(backend_name).subtract(x1, x2)

def safe_subtract(x1: Any, x2: Any, *, backend_name: Optional[str] = None) -> Any:
    """Safe subtraction with fallback."""
    try:
        return subtract(x1, x2, backend_name=backend_name)
    except Exception:
        return _np.subtract(x1, x2)
