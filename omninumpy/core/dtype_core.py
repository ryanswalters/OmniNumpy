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

def bool(*, backend_name: Optional[str] = None) -> Any:
    """Return boolean dtype from backend."""
    return _get_backend(backend_name).bool

def int32(*, backend_name: Optional[str] = None) -> Any:
    """Return int32 dtype from backend."""
    return _get_backend(backend_name).int32

def float64(*, backend_name: Optional[str] = None) -> Any:
    """Return float64 dtype from backend."""
    return _get_backend(backend_name).float64

def pi(*, backend_name: Optional[str] = None) -> float:
    """Return pi constant from backend."""
    return _get_backend(backend_name).pi
