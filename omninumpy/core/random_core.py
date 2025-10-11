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

def random(shape: Sequence[int], *, backend_name: Optional[str] = None) -> Any:
    """Generate random numbers in [0,1)."""
    return _get_backend(backend_name).random.random(shape)

def safe_random(shape: Sequence[int], *, backend_name: Optional[str] = None) -> Any:
    """Safe random generator with fallback."""
    try:
        return random(shape, backend_name=backend_name)
    except Exception:
        return _np.random.random(shape)

def randn(shape: Sequence[int], *, backend_name: Optional[str] = None) -> Any:
    """Generate standard normal random numbers."""
    return _get_backend(backend_name).random.randn(shape)

def safe_randn(shape: Sequence[int], *, backend_name: Optional[str] = None) -> Any:
    """Safe randn generator with fallback."""
    try:
        return randn(shape, backend_name=backend_name)
    except Exception:
        return _np.random.randn(*shape)
