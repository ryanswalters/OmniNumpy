"""
Core compatibility layer for OmniNumpy.

This module provides a NumPy-compatible interface that works across different backends
and handles NumPy 2.x compatibility issues.
"""

import sys
import warnings
from typing import Any, Union, Optional, Sequence

def _get_backend():
    """Get the current backend module."""
    from . import backend as backend_module
    return backend_module.get_current_backend_module()

# NumPy 2.x compatibility layer
class NumPy2CompatibilityMixin:
    """Mixin class to handle NumPy 2.x breaking changes."""
    
    @staticmethod
    def _handle_deprecated_params(**kwargs):
        """Handle deprecated parameters that changed in NumPy 2.x."""
        # Handle common parameter name changes
        if 'keepdims' in kwargs and 'keepdim' not in kwargs:
            # Some backends might use keepdim instead of keepdims
            pass
        return kwargs

# Create proxy functions for common NumPy functions with compatibility handling
def array(object, dtype=None, *, copy=None, order='K', subok=False, ndmin=0, like=None):
    """Create an array using the current backend."""
    backend = _get_backend()
    
    # Handle NumPy 2.x copy parameter changes
    kwargs = {'dtype': dtype, 'order': order, 'subok': subok, 'ndmin': ndmin}
    
    # Remove None values and unsupported parameters based on backend
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    
    # Handle copy parameter for different backends
    if copy is not None:
        if hasattr(backend, 'array') and 'copy' in backend.array.__code__.co_varnames:
            kwargs['copy'] = copy
    
    # Remove parameters not supported by the backend
    if backend.__name__ == 'cupy':
        # CuPy might not support all NumPy parameters
        kwargs.pop('subok', None)
        kwargs.pop('like', None)
    elif 'jax' in str(type(backend)):
        # JAX has different parameter support
        kwargs.pop('subok', None)
        kwargs.pop('like', None)
        kwargs.pop('order', None)
    
    try:
        return backend.array(object, **kwargs)
    except TypeError as e:
        # Fallback: try with minimal parameters
        try:
            return backend.array(object, dtype=dtype)
        except Exception:
            return backend.array(object)

def zeros(shape, dtype=float, order='C', *, like=None):
    """Return a new array of given shape and type, filled with zeros."""
    backend = _get_backend()
    kwargs = {'dtype': dtype, 'order': order}
    
    # Remove unsupported parameters
    if 'jax' in str(type(backend)) or backend.__name__ == 'cupy':
        kwargs.pop('order', None)
    
    try:
        return backend.zeros(shape, **kwargs)
    except TypeError:
        return backend.zeros(shape, dtype=dtype)

def ones(shape, dtype=None, order='C', *, like=None):
    """Return a new array of given shape and type, filled with ones."""
    backend = _get_backend()
    kwargs = {}
    if dtype is not None:
        kwargs['dtype'] = dtype
    if 'numpy' in str(type(backend)):
        kwargs['order'] = order
    
    try:
        return backend.ones(shape, **kwargs)
    except TypeError:
        return backend.ones(shape)

def empty(shape, dtype=float, order='C', *, like=None):
    """Return a new array of given shape and type, without initializing entries."""
    backend = _get_backend()
    kwargs = {'dtype': dtype}
    if 'numpy' in str(type(backend)):
        kwargs['order'] = order
    
    try:
        return backend.empty(shape, **kwargs)
    except TypeError:
        return backend.empty(shape, dtype=dtype)

# Mathematical functions
def add(x1, x2, out=None, **kwargs):
    """Add arguments element-wise."""
    backend = _get_backend()
    return backend.add(x1, x2)

def subtract(x1, x2, out=None, **kwargs):
    """Subtract arguments element-wise."""
    backend = _get_backend()
    return backend.subtract(x1, x2)

def multiply(x1, x2, out=None, **kwargs):
    """Multiply arguments element-wise."""
    backend = _get_backend()
    return backend.multiply(x1, x2)

def divide(x1, x2, out=None, **kwargs):
    """Divide arguments element-wise."""
    backend = _get_backend()
    return backend.divide(x1, x2)

def dot(a, b, out=None):
    """Dot product of two arrays."""
    backend = _get_backend()
    return backend.dot(a, b)

def matmul(x1, x2, out=None, **kwargs):
    """Matrix product of two arrays."""
    backend = _get_backend()
    if hasattr(backend, 'matmul'):
        return backend.matmul(x1, x2)
    else:
        return backend.dot(x1, x2)

# Reduction functions
def sum(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=None):
    """Sum of array elements over a given axis."""
    backend = _get_backend()
    kwargs = {}
    if axis is not None:
        kwargs['axis'] = axis
    if dtype is not None:
        kwargs['dtype'] = dtype
    if keepdims:
        kwargs['keepdims'] = keepdims
    
    # Handle backend-specific parameters
    if 'jax' in str(type(backend)):
        kwargs.pop('out', None)
        kwargs.pop('initial', None)
        kwargs.pop('where', None)
    
    try:
        return backend.sum(a, **kwargs)
    except TypeError:
        # Fallback with minimal parameters
        return backend.sum(a, axis=axis)

def mean(a, axis=None, dtype=None, out=None, keepdims=False):
    """Compute the arithmetic mean along the specified axis."""
    backend = _get_backend()
    kwargs = {}
    if axis is not None:
        kwargs['axis'] = axis
    if dtype is not None:
        kwargs['dtype'] = dtype
    if keepdims:
        kwargs['keepdims'] = keepdims
    
    if 'jax' in str(type(backend)):
        kwargs.pop('out', None)
    
    try:
        return backend.mean(a, **kwargs)
    except TypeError:
        return backend.mean(a, axis=axis)

# Shape manipulation
def reshape(a, newshape, order='C'):
    """Gives a new shape to an array without changing its data."""
    backend = _get_backend()
    if 'jax' in str(type(backend)) or backend.__name__ == 'cupy':
        return backend.reshape(a, newshape)
    else:
        return backend.reshape(a, newshape, order=order)

def transpose(a, axes=None):
    """Reverse or permute the axes of an array."""
    backend = _get_backend()
    if axes is None:
        return backend.transpose(a)
    else:
        return backend.transpose(a, axes)

# Linear algebra
def linalg():
    """Linear algebra routines."""
    backend = _get_backend()
    return backend.linalg

# Constants and dtypes
def _create_dtype_proxy(dtype_name):
    """Create a proxy for numpy dtypes that works across backends."""
    def dtype_proxy():
        backend = _get_backend()
        return getattr(backend, dtype_name)
    return dtype_proxy

# Common dtypes
int8 = property(lambda self: _get_backend().int8)
int16 = property(lambda self: _get_backend().int16)
int32 = property(lambda self: _get_backend().int32) 
int64 = property(lambda self: _get_backend().int64)
float16 = property(lambda self: _get_backend().float16)
float32 = property(lambda self: _get_backend().float32)
float64 = property(lambda self: _get_backend().float64)
complex64 = property(lambda self: _get_backend().complex64)
complex128 = property(lambda self: _get_backend().complex128)
bool_ = property(lambda self: _get_backend().bool_)

# Mathematical constants
pi = property(lambda self: _get_backend().pi)
e = property(lambda self: _get_backend().e)