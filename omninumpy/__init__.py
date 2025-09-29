"""
OmniNumpy - Compatibility layer for NumPy 2.x with GPU acceleration support.

Stop fighting NumPy version hell. OmniNumpy is the compatibility layer that lets 
legacy code run on NumPy 2.x while unlocking GPU acceleration with zero refactoring.
Drop-in replacement. Backend agnostic. Just works.

Usage:
    import omninumpy as np
    
    # Your existing NumPy code works unchanged
    arr = np.array([1, 2, 3])
    result = np.sum(arr)
"""

# Import backend management first
from .backend import (
    get_backend, set_backend, list_backends, 
    get_backend_module, auto_backend
)

# Import version
from .version import __version__

# Import numpy directly to start with
import numpy as _np

# Initialize with best available backend
_current_backend_name = auto_backend()

def _get_current_backend():
    """Get the current backend module."""
    return get_backend_module()

# Core array creation functions with compatibility
def array(object, dtype=None, *, copy=None, order='K', subok=False, ndmin=0, like=None):
    """Create an array using the current backend."""
    backend = _get_current_backend()
    
    # Handle parameters based on backend
    if backend.__name__ == 'numpy':
        # NumPy 2.x handles copy parameter differently
        kwargs = {'dtype': dtype, 'order': order, 'subok': subok, 'ndmin': ndmin}
        if copy is not None:
            kwargs['copy'] = copy
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return backend.array(object, **kwargs)
    else:
        # Other backends might not support all parameters
        kwargs = {}
        if dtype is not None:
            kwargs['dtype'] = dtype
        return backend.array(object, **kwargs)

def zeros(shape, dtype=float, order='C', *, like=None):
    """Return a new array of given shape and type, filled with zeros."""
    backend = _get_current_backend()
    kwargs = {'dtype': dtype}
    if backend.__name__ == 'numpy':
        kwargs['order'] = order
    return backend.zeros(shape, **kwargs)

def ones(shape, dtype=None, order='C', *, like=None):
    """Return a new array of given shape and type, filled with ones."""
    backend = _get_current_backend()
    kwargs = {}
    if dtype is not None:
        kwargs['dtype'] = dtype
    if backend.__name__ == 'numpy':
        kwargs['order'] = order
    return backend.ones(shape, **kwargs)

def sum(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=None):
    """Sum of array elements over a given axis."""
    backend = _get_current_backend()
    kwargs = {}
    if axis is not None:
        kwargs['axis'] = axis
    if dtype is not None:
        kwargs['dtype'] = dtype
    if keepdims:
        kwargs['keepdims'] = keepdims
    
    # Handle backend-specific parameters
    if backend.__name__ != 'numpy':
        # Remove numpy-specific parameters for other backends
        pass
    else:
        if out is not None:
            kwargs['out'] = out
        if initial is not None:
            kwargs['initial'] = initial
        if where is not None:
            kwargs['where'] = where
    
    return backend.sum(a, **kwargs)

# Mathematical functions
def add(x1, x2, out=None, **kwargs):
    """Add arguments element-wise."""
    return _get_current_backend().add(x1, x2)

def subtract(x1, x2, out=None, **kwargs):
    """Subtract arguments element-wise."""
    return _get_current_backend().subtract(x1, x2)

def multiply(x1, x2, out=None, **kwargs):
    """Multiply arguments element-wise."""
    return _get_current_backend().multiply(x1, x2)

def divide(x1, x2, out=None, **kwargs):
    """Divide arguments element-wise."""
    return _get_current_backend().divide(x1, x2)

def dot(a, b, out=None):
    """Dot product of two arrays."""
    return _get_current_backend().dot(a, b)

# For everything else, delegate to the backend directly
def __getattr__(name):
    """Delegate attribute access to the current backend."""
    # Handle version specially
    if name == 'version':
        backend = _get_current_backend()
        if hasattr(backend, 'version'):
            return backend.version
        elif hasattr(backend, '__version__'):
            return backend.__version__
        else:
            return "unknown"
    
    backend = _get_current_backend()
    if hasattr(backend, name):
        return getattr(backend, name)
    raise AttributeError(f"module 'omninumpy' has no attribute '{name}'")