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
        
        # Handle NumPy 2.x copy parameter compatibility
        if copy is not None:
            if copy is False:
                # NumPy 2.x requires using asarray for copy=False behavior
                try:
                    return backend.asarray(object, dtype=dtype, order=order)
                except Exception:
                    # Fallback to regular array creation
                    kwargs['copy'] = copy
            else:
                kwargs['copy'] = copy
        
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        try:
            return backend.array(object, **kwargs)
        except ValueError as e:
            if 'copy' in str(e) and copy is False:
                # Handle NumPy 2.x copy=False compatibility issue
                # Use asarray as recommended by NumPy 2.x migration guide
                return backend.asarray(object, dtype=dtype, order=order)
            else:
                raise
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
=======
""
OmniNumPy: NumPy Compatibility Layer with Backend Switching

This module provides a drop-in replacement for NumPy that offers:
- Backward compatibility with NumPy 1.x APIs via emulate()
- Backend switching between NumPy, PyTorch, CuPy, and JAX
- Automatic backend selection based on available hardware

LIMITATIONS (Important - Read Before Use):
- Only 4 functions are fully backend-aware: array, dot, matmul, linalg.inv
- All other NumPy functions fall back to NumPy or raise AttributeError in non-NumPy backends
- JAX backend uses basic jax.numpy without JIT compilation or GPU/TPU device handling
- This is a prototype demonstrating the concept, not a complete NumPy replacement

For production use, understand these limitations and test thoroughly.
"""

import numpy as _np
import time
from functools import wraps

BACKEND = "numpy"
DEVICE = None  # For JAX: "cpu", "gpu", "tpu"
JIT_ENABLED = False
JIT_THRESHOLD = 1000  # Array size threshold for JIT compilation
JIT_THRESHOLDS = {
    'dot': 1000,
    'matmul': 1000,
    'mean': 5000,
    'sum': 5000,
    'stack': 1000,
    'concatenate': 1000,
    'reshape': 2000,
    'transpose': 2000,
    'astype': 2000,
    'clip': 2000,
    'where': 2000,
    'add': 10000,
    'subtract': 10000,
    'multiply': 10000,
    'divide': 10000,
    'linalg_inv': 500,
    'linalg_svd': 300,
    'linalg_eig': 300,
    'linalg_cholesky': 500,
    'linalg_qr': 500,
    'linalg_det': 1000,
    'linalg_norm': 2000,
    'linalg_solve': 500,
}
legacy_attrs = {}
_timings = {}

def _track_perf(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        _timings[func.__name__] = _timings.get(func.__name__, 0) + elapsed
        return result
    return wrapper

def get_timings():
    return _timings.copy()

def validate_array(arr, shape=None, dtype=None):
    if shape and arr.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {arr.shape}")
    if dtype and arr.dtype != dtype:
        raise TypeError(f"Expected dtype {dtype}, got {arr.dtype}")
    return arr

def to_numpy(arr):
    """Convert array to NumPy format"""
    if BACKEND == "torch":
        return arr.detach().cpu().numpy()
    elif BACKEND == "cupy":
        return arr.get()  # CuPy to NumPy
    elif BACKEND == "jax":
        jnp = _get_jax_numpy()
        return jnp.asarray(arr)  # JAX to NumPy
    else:
        return _np.asarray(arr)

def to_backend(arr):
    """Convert NumPy array to current backend"""
    if BACKEND == "torch":
        torch = _get_torch()
        return torch.tensor(arr)
    elif BACKEND == "cupy":
        cp = _get_cupy()
        return cp.array(arr)
    elif BACKEND == "jax":
        jnp = _get_jax_numpy()
        result = jnp.array(arr)
        return _device_put_if_needed(result)
    else:
        return _np.array(arr)

def _get_torch():
    if not hasattr(_get_torch, '_torch'):
        import torch
        _get_torch._torch = torch
    return _get_torch._torch

def _get_cupy():
    if not hasattr(_get_cupy, '_cupy'):
        import cupy
        _get_cupy._cupy = cupy
    return _get_cupy._cupy

def _get_jax():
    if not hasattr(_get_jax, '_jax'):
        import jax
        _get_jax._jax = jax
    return _get_jax._jax

def _get_jax_numpy():
    if not hasattr(_get_jax_numpy, '_jnp'):
        import jax.numpy as jnp
        _get_jax_numpy._jnp = jnp
    return _get_jax_numpy._jnp

def _get_jax_random():
    if not hasattr(_get_jax_random, '_jr'):
        import jax.random as jr
        _get_jax_random._jr = jr
    return _get_jax_random._jr

def auto_backend():
    """Picks fastest available backend with smart device detection"""
    # Priority: JAX GPU/TPU > CuPy > PyTorch GPU > CPU backends

    # Check JAX first (most advanced)
    try:
        jax = _get_jax()
        # Prefer TPU if available
        if jax.devices("tpu"):
            return "jax:tpu"
        # Then GPU
        elif jax.devices("gpu"):
            return "jax:gpu"
        # Fall back to CPU
        else:
            return "jax:cpu"
    except (ImportError, RuntimeError):
        pass

    # Check CuPy (GPU-only library)
    try:
        _get_cupy()
        return "cupy"
    except ImportError:
        pass

    # Check PyTorch
    try:
        torch = _get_torch()
        if torch.cuda.is_available():
            return "torch"
    except ImportError:
        pass

    # Fall back to NumPy
    return "numpy"

def set_backend(name):
    global BACKEND, DEVICE
    if ":" in name:
        backend, device = name.split(":", 1)
        if backend not in ("numpy", "torch", "cupy", "jax"):
            raise ValueError(f"Unsupported backend: {backend}")
        if backend == "jax":
            if device not in ("cpu", "gpu", "tpu"):
                raise ValueError(f"Unsupported JAX device: {device}")
            BACKEND = backend
            DEVICE = device
        else:
            raise ValueError(f"Device specification not supported for backend: {backend}")
    else:
        if name not in ("numpy", "torch", "cupy", "jax"):
            raise ValueError("Unsupported backend")
        BACKEND = name
        DEVICE = None

def _array(x, *args, **kwargs):
    if BACKEND == "torch":
        torch = _get_torch()
        return torch.tensor(x, *args, **kwargs)
    elif BACKEND == "cupy":
        cp = _get_cupy()
        return cp.array(x, *args, **kwargs)
    elif BACKEND == "jax":
        jnp = _get_jax_numpy()
        result = jnp.array(x, *args, **kwargs)
        return _device_put_if_needed(result)
    else:
        return _np.array(x, *args, **kwargs)

def _dot(a, b, *args, **kwargs):
    if BACKEND == "torch":
        torch = _get_torch()
        return torch.matmul(a, b, *args, **kwargs)
    elif BACKEND == "cupy":
        cp = _get_cupy()
        return cp.dot(a, b, *args, **kwargs)
    elif BACKEND == "jax":
        jnp = _get_jax_numpy()
        if _should_use_jit([a, b], 'dot'):
            global _jax_dot_jit
            if _jax_dot_jit is None:
                jax = _get_jax()
                _jax_dot_jit = jax.jit(jnp.dot)
            result = _jax_dot_jit(a, b, *args, **kwargs)
        else:
            result = jnp.dot(a, b, *args, **kwargs)
        return _device_put_if_needed(result)
    else:
        return _np.dot(a, b, *args, **kwargs)

def _matmul(a, b, *args, **kwargs):
    if BACKEND == "torch":
        torch = _get_torch()
        return torch.matmul(a, b, *args, **kwargs)
    elif BACKEND == "cupy":
        cp = _get_cupy()
        return cp.matmul(a, b, *args, **kwargs)
    elif BACKEND == "jax":
        jnp = _get_jax_numpy()
        if _should_use_jit([a, b], 'matmul'):
            global _jax_matmul_jit
            if _jax_matmul_jit is None:
                jax = _get_jax()
                _jax_matmul_jit = jax.jit(jnp.matmul)
            result = _jax_matmul_jit(a, b, *args, **kwargs)
        else:
            result = jnp.matmul(a, b, *args, **kwargs)
        return _device_put_if_needed(result)
    else:
        return _np.matmul(a, b, *args, **kwargs)

def _linalg_inv(a, *args, **kwargs):
    if BACKEND == "torch":
        torch = _get_torch()
        return torch.linalg.inv(a, *args, **kwargs)
    elif BACKEND == "cupy":
        cp = _get_cupy()
        return cp.linalg.inv(a, *args, **kwargs)
    elif BACKEND == "jax":
        jnp = _get_jax_numpy()
        if _should_use_jit([a], 'linalg_inv'):
            global _jax_linalg_inv_jit
            if _jax_linalg_inv_jit is None:
                jax = _get_jax()
                _jax_linalg_inv_jit = jax.jit(jnp.linalg.inv)
            result = _jax_linalg_inv_jit(a, *args, **kwargs)
        else:
            result = jnp.linalg.inv(a, *args, **kwargs)
        return _device_put_if_needed(result)
    else:
        return _np.linalg.inv(a, *args, **kwargs)

def _mean(a, *args, **kwargs):
    if BACKEND == "torch":
        torch = _get_torch()
        return torch.mean(a, *args, **kwargs)
    elif BACKEND == "cupy":
        cp = _get_cupy()
        return cp.mean(a, *args, **kwargs)
    elif BACKEND == "jax":
        jnp = _get_jax_numpy()
        if _should_use_jit([a], 'mean'):
            global _jax_mean_jit
            if _jax_mean_jit is None:
                jax = _get_jax()
                _jax_mean_jit = jax.jit(jnp.mean)
            result = _jax_mean_jit(a, *args, **kwargs)
        else:
            result = jnp.mean(a, *args, **kwargs)
        return _device_put_if_needed(result)
    else:
        return _np.mean(a, *args, **kwargs)

def _sum(a, *args, **kwargs):
    if BACKEND == "torch":
        torch = _get_torch()
        return torch.sum(a, *args, **kwargs)
    elif BACKEND == "cupy":
        cp = _get_cupy()
        return cp.sum(a, *args, **kwargs)
    elif BACKEND == "jax":
        jnp = _get_jax_numpy()
        result = jnp.sum(a, *args, **kwargs)
        return _device_put_if_needed(result)
    else:
        return _np.sum(a, *args, **kwargs)

def _stack(arrays, *args, **kwargs):
    if BACKEND == "torch":
        torch = _get_torch()
        return torch.stack(arrays, *args, **kwargs)
    elif BACKEND == "cupy":
        cp = _get_cupy()
        return cp.stack(arrays, *args, **kwargs)
    elif BACKEND == "jax":
        jnp = _get_jax_numpy()
        result = jnp.stack(arrays, *args, **kwargs)
        return _device_put_if_needed(result)
    else:
        return _np.stack(arrays, *args, **kwargs)

def _concatenate(arrays, *args, **kwargs):
    if BACKEND == "torch":
        torch = _get_torch()
        return torch.cat(arrays, *args, **kwargs)
    elif BACKEND == "cupy":
        cp = _get_cupy()
        return cp.concatenate(arrays, *args, **kwargs)
    elif BACKEND == "jax":
        jnp = _get_jax_numpy()
        result = jnp.concatenate(arrays, *args, **kwargs)
        return _device_put_if_needed(result)
    else:
        return _np.concatenate(arrays, *args, **kwargs)

def _eye(n, *args, **kwargs):
    if BACKEND == "torch":
        torch = _get_torch()
        return torch.eye(n, *args, **kwargs)
    elif BACKEND == "cupy":
        cp = _get_cupy()
        return cp.eye(n, *args, **kwargs)
    elif BACKEND == "jax":
        jnp = _get_jax_numpy()
        result = jnp.eye(n, *args, **kwargs)
        return _device_put_if_needed(result)
    else:
        return _np.eye(n, *args, **kwargs)

def _zeros(shape, *args, **kwargs):
    if BACKEND == "torch":
        torch = _get_torch()
        return torch.zeros(shape, *args, **kwargs)
    elif BACKEND == "cupy":
        cp = _get_cupy()
        return cp.zeros(shape, *args, **kwargs)
    elif BACKEND == "jax":
        jnp = _get_jax_numpy()
        result = jnp.zeros(shape, *args, **kwargs)
        return _device_put_if_needed(result)
    else:
        return _np.zeros(shape, *args, **kwargs)

def _ones(shape, *args, **kwargs):
    if BACKEND == "torch":
        torch = _get_torch()
        return torch.ones(shape, *args, **kwargs)
    elif BACKEND == "cupy":
        cp = _get_cupy()
        return cp.ones(shape, *args, **kwargs)
    elif BACKEND == "jax":
        jnp = _get_jax_numpy()
        result = jnp.ones(shape, *args, **kwargs)
        return _device_put_if_needed(result)
    else:
        return _np.ones(shape, *args, **kwargs)

def _reshape(a, *args, **kwargs):
    if BACKEND == "torch":
        torch = _get_torch()
        return torch.reshape(a, *args, **kwargs)
    elif BACKEND == "cupy":
        cp = _get_cupy()
        return cp.reshape(a, *args, **kwargs)
    elif BACKEND == "jax":
        jnp = _get_jax_numpy()
        result = jnp.reshape(a, *args, **kwargs)
        return _device_put_if_needed(result)
    else:
        return _np.reshape(a, *args, **kwargs)

def _transpose(a, *args, **kwargs):
    if BACKEND == "torch":
        torch = _get_torch()
        return torch.transpose(a, *args, **kwargs)
    elif BACKEND == "cupy":
        cp = _get_cupy()
        return cp.transpose(a, *args, **kwargs)
    elif BACKEND == "jax":
        jnp = _get_jax_numpy()
        result = jnp.transpose(a, *args, **kwargs)
        return _device_put_if_needed(result)
    else:
        return _np.transpose(a, *args, **kwargs)

def _astype(a, dtype, *args, **kwargs):
    if BACKEND == "torch":
        torch = _get_torch()
        return a.to(dtype)
    elif BACKEND == "cupy":
        cp = _get_cupy()
        return a.astype(dtype, *args, **kwargs)
    elif BACKEND == "jax":
        jnp = _get_jax_numpy()
        result = jnp.astype(a, dtype, *args, **kwargs)
        return _device_put_if_needed(result)
    else:
        return _np.astype(a, dtype, *args, **kwargs)

def _clip(a, *args, **kwargs):
    if BACKEND == "torch":
        torch = _get_torch()
        return torch.clamp(a, *args, **kwargs)
    elif BACKEND == "cupy":
        cp = _get_cupy()
        return cp.clip(a, *args, **kwargs)
    elif BACKEND == "jax":
        jnp = _get_jax_numpy()
        result = jnp.clip(a, *args, **kwargs)
        return _device_put_if_needed(result)
    else:
        return _np.clip(a, *args, **kwargs)

def _where(condition, *args, **kwargs):
    if BACKEND == "torch":
        torch = _get_torch()
        return torch.where(condition, *args, **kwargs)
    elif BACKEND == "cupy":
        cp = _get_cupy()
        return cp.where(condition, *args, **kwargs)
    elif BACKEND == "jax":
        jnp = _get_jax_numpy()
        result = jnp.where(condition, *args, **kwargs)
        return _device_put_if_needed(result)
    else:
        return _np.where(condition, *args, **kwargs)

def _add(a, b, *args, **kwargs):
    if BACKEND == "torch":
        torch = _get_torch()
        return torch.add(a, b, *args, **kwargs)
    elif BACKEND == "cupy":
        cp = _get_cupy()
        return cp.add(a, b, *args, **kwargs)
    elif BACKEND == "jax":
        jnp = _get_jax_numpy()
        result = jnp.add(a, b, *args, **kwargs)
        return _device_put_if_needed(result)
    else:
        return _np.add(a, b, *args, **kwargs)

def _subtract(a, b, *args, **kwargs):
    if BACKEND == "torch":
        torch = _get_torch()
        return torch.subtract(a, b, *args, **kwargs)
    elif BACKEND == "cupy":
        cp = _get_cupy()
        return cp.subtract(a, b, *args, **kwargs)
    elif BACKEND == "jax":
        jnp = _get_jax_numpy()
        result = jnp.subtract(a, b, *args, **kwargs)
        return _device_put_if_needed(result)
    else:
        return _np.subtract(a, b, *args, **kwargs)

def _multiply(a, b, *args, **kwargs):
    if BACKEND == "torch":
        torch = _get_torch()
        return torch.multiply(a, b, *args, **kwargs)
    elif BACKEND == "cupy":
        cp = _get_cupy()
        return cp.multiply(a, b, *args, **kwargs)
    elif BACKEND == "jax":
        jnp = _get_jax_numpy()
        result = jnp.multiply(a, b, *args, **kwargs)
        return _device_put_if_needed(result)
    else:
        return _np.multiply(a, b, *args, **kwargs)

def _divide(a, b, *args, **kwargs):
    if BACKEND == "torch":
        torch = _get_torch()
        return torch.divide(a, b, *args, **kwargs)
    elif BACKEND == "cupy":
        cp = _get_cupy()
        return cp.divide(a, b, *args, **kwargs)
    elif BACKEND == "jax":
        jnp = _get_jax_numpy()
        result = jnp.divide(a, b, *args, **kwargs)
        return _device_put_if_needed(result)
    else:
        return _np.divide(a, b, *args, **kwargs)

# JIT-compiled versions for JAX
_jax_dot_jit = None
_jax_matmul_jit = None
_jax_mean_jit = None
_jax_sum_jit = None
_jax_reshape_jit = None
_jax_transpose_jit = None
_jax_clip_jit = None
_jax_linalg_inv_jit = None
_jax_linalg_svd_jit = None
_jax_linalg_eig_jit = None
_jax_linalg_cholesky_jit = None
_jax_linalg_qr_jit = None
_jax_linalg_det_jit = None
_jax_linalg_norm_jit = None
_jax_linalg_solve_jit = None
_jax_add_jit = None
_jax_subtract_jit = None
_jax_multiply_jit = None
_jax_divide_jit = None

def _should_use_jit(arrays, func_name=None):
    """Determine if JIT compilation should be used based on array sizes and function"""
    if not JIT_ENABLED or BACKEND != "jax":
        return False

    # Get function-specific threshold
    threshold = JIT_THRESHOLDS.get(func_name, JIT_THRESHOLD)

    # Check if any array dimension exceeds threshold
    for arr in arrays:
        if hasattr(arr, 'shape'):
            if any(dim > threshold for dim in arr.shape):
                return True
    return False

def _get_jax_device():
    """Get the appropriate JAX device"""
    if DEVICE is None:
        return None
    jax = _get_jax()
    if DEVICE == "cpu":
        return jax.devices("cpu")[0]
    elif DEVICE == "gpu":
        try:
            return jax.devices("gpu")[0]
        except IndexError:
            raise RuntimeError("GPU device requested but no GPU available")
    elif DEVICE == "tpu":
        try:
            return jax.devices("tpu")[0]
        except IndexError:
            raise RuntimeError("TPU device requested but no TPU available")
    return None

def _device_put_if_needed(arr):
    """Move array to specified device if using JAX"""
    if BACKEND == "jax" and DEVICE is not None:
        jax = _get_jax()
        device = _get_jax_device()
        if device is not None:
            return jax.device_put(arr, device)
    return arr

_array = _track_perf(_array)
_dot = _track_perf(_dot)
_matmul = _track_perf(_matmul)
_linalg_inv = _track_perf(_linalg_inv)
_mean = _track_perf(_mean)
_sum = _track_perf(_sum)
_stack = _track_perf(_stack)
_concatenate = _track_perf(_concatenate)
_eye = _track_perf(_eye)
_zeros = _track_perf(_zeros)
_ones = _track_perf(_ones)
_reshape = _track_perf(_reshape)
_transpose = _track_perf(_transpose)
_astype = _track_perf(_astype)
_clip = _track_perf(_clip)
_where = _track_perf(_where)

def _linalg_svd(a, *args, **kwargs):
    if BACKEND == "torch":
        torch = _get_torch()
        return torch.linalg.svd(a, *args, **kwargs)
    elif BACKEND == "cupy":
        cp = _get_cupy()
        return cp.linalg.svd(a, *args, **kwargs)
    elif BACKEND == "jax":
        jnp = _get_jax_numpy()
        result = jnp.linalg.svd(a, *args, **kwargs)
        return _device_put_if_needed(result)
    else:
        return _np.linalg.svd(a, *args, **kwargs)

def _linalg_eig(a, *args, **kwargs):
    if BACKEND == "torch":
        torch = _get_torch()
        return torch.linalg.eig(a, *args, **kwargs)
    elif BACKEND == "cupy":
        cp = _get_cupy()
        return cp.linalg.eig(a, *args, **kwargs)
    elif BACKEND == "jax":
        jnp = _get_jax_numpy()
        result = jnp.linalg.eig(a, *args, **kwargs)
        return _device_put_if_needed(result)
    else:
        return _np.linalg.eig(a, *args, **kwargs)

def _linalg_cholesky(a, *args, **kwargs):
    if BACKEND == "torch":
        torch = _get_torch()
        return torch.linalg.cholesky(a, *args, **kwargs)
    elif BACKEND == "cupy":
        cp = _get_cupy()
        return cp.linalg.cholesky(a, *args, **kwargs)
    elif BACKEND == "jax":
        jnp = _get_jax_numpy()
        result = jnp.linalg.cholesky(a, *args, **kwargs)
        return _device_put_if_needed(result)
    else:
        return _np.linalg.cholesky(a, *args, **kwargs)

def _linalg_qr(a, *args, **kwargs):
    if BACKEND == "torch":
        torch = _get_torch()
        return torch.linalg.qr(a, *args, **kwargs)
    elif BACKEND == "cupy":
        cp = _get_cupy()
        return cp.linalg.qr(a, *args, **kwargs)
    elif BACKEND == "jax":
        jnp = _get_jax_numpy()
        result = jnp.linalg.qr(a, *args, **kwargs)
        return _device_put_if_needed(result)
    else:
        return _np.linalg.qr(a, *args, **kwargs)

def _linalg_det(a, *args, **kwargs):
    if BACKEND == "torch":
        torch = _get_torch()
        return torch.linalg.det(a, *args, **kwargs)
    elif BACKEND == "cupy":
        cp = _get_cupy()
        return cp.linalg.det(a, *args, **kwargs)
    elif BACKEND == "jax":
        jnp = _get_jax_numpy()
        result = jnp.linalg.det(a, *args, **kwargs)
        return _device_put_if_needed(result)
    else:
        return _np.linalg.det(a, *args, **kwargs)

def _linalg_norm(a, *args, **kwargs):
    if BACKEND == "torch":
        torch = _get_torch()
        return torch.linalg.norm(a, *args, **kwargs)
    elif BACKEND == "cupy":
        cp = _get_cupy()
        return cp.linalg.norm(a, *args, **kwargs)
    elif BACKEND == "jax":
        jnp = _get_jax_numpy()
        result = jnp.linalg.norm(a, *args, **kwargs)
        return _device_put_if_needed(result)
    else:
        return _np.linalg.norm(a, *args, **kwargs)

def _linalg_solve(a, b, *args, **kwargs):
    if BACKEND == "torch":
        torch = _get_torch()
        return torch.linalg.solve(a, b, *args, **kwargs)
    elif BACKEND == "cupy":
        cp = _get_cupy()
        return cp.linalg.solve(a, b, *args, **kwargs)
    elif BACKEND == "jax":
        jnp = _get_jax_numpy()
        result = jnp.linalg.solve(a, b, *args, **kwargs)
        return _device_put_if_needed(result)
    else:
        return _np.linalg.solve(a, b, *args, **kwargs)

_linalg_svd = _track_perf(_linalg_svd)
_linalg_eig = _track_perf(_linalg_eig)
_linalg_cholesky = _track_perf(_linalg_cholesky)
_linalg_qr = _track_perf(_linalg_qr)
_linalg_det = _track_perf(_linalg_det)
_linalg_norm = _track_perf(_linalg_norm)
_linalg_solve = _track_perf(_linalg_solve)
_add = _track_perf(_add)
_subtract = _track_perf(_subtract)
_multiply = _track_perf(_multiply)
_divide = _track_perf(_divide)

class RandomWrapper:
    def __init__(self, backend):
        self.backend = backend

    def random(self, *args, **kwargs):
        if self.backend == "numpy":
            return _np.random.random(*args, **kwargs)
        elif self.backend == "torch":
            torch = _get_torch()
            if len(args) > 0 and hasattr(args[0], '__iter__'):
                shape = args[0]
            else:
                shape = args
            return torch.rand(*shape, **kwargs)
        elif self.backend == "cupy":
            cp = _get_cupy()
            return cp.random.random(*args, **kwargs)
        elif self.backend == "jax":
            jnp = _get_jax_numpy()
            result = jnp.array(_np.random.random(*args, **kwargs))
            return _device_put_if_needed(result)
        else:
            return _np.random.random(*args, **kwargs)

    def randn(self, *args, **kwargs):
        if self.backend == "numpy":
            return _np.random.randn(*args, **kwargs)
        elif self.backend == "torch":
            torch = _get_torch()
            return torch.randn(*args, **kwargs)
        elif self.backend == "cupy":
            cp = _get_cupy()
            return cp.random.randn(*args, **kwargs)
        elif self.backend == "jax":
            jnp = _get_jax_numpy()
            jr = _get_jax_random()
            key = jr.PRNGKey(42)
            if len(args) > 0:
                shape = args[0] if hasattr(args[0], '__iter__') and not isinstance(args[0], (int, float)) else args
                result = jr.normal(key, shape, **kwargs)
            else:
                result = jr.normal(key, **kwargs)
            return _device_put_if_needed(result)
        else:
            return _np.random.randn(*args, **kwargs)

class LinalgWrapper:
    def __init__(self, backend):
        self.backend = backend

    def inv(self, a, *args, **kwargs):
        return _linalg_inv(a, *args, **kwargs)

    def svd(self, a, *args, **kwargs):
        return _linalg_svd(a, *args, **kwargs)

    def eig(self, a, *args, **kwargs):
        return _linalg_eig(a, *args, **kwargs)

    def cholesky(self, a, *args, **kwargs):
        return _linalg_cholesky(a, *args, **kwargs)

    def qr(self, a, *args, **kwargs):
        return _linalg_qr(a, *args, **kwargs)

    def det(self, a, *args, **kwargs):
        return _linalg_det(a, *args, **kwargs)

    def norm(self, a, *args, **kwargs):
        return _linalg_norm(a, *args, **kwargs)

    def solve(self, a, b, *args, **kwargs):
        return _linalg_solve(a, b, *args, **kwargs)

    def __getattr__(self, name):
        # For other linalg functions not wrapped, raise error for non-numpy backends
        if self.backend == "numpy":
            return getattr(_np.linalg, name)
        else:
            raise AttributeError(f"linalg.{name} not implemented for backend {self.backend}")

def asscalar(a):
    return a.item()

def matrix(data, dtype=None, copy=True):
    return _np.array(data, dtype=dtype, copy=copy)

def emulate(version="2.x"):
    global legacy_attrs
    if version.startswith("1."):
        legacy_attrs["int"] = _np.int64
        legacy_attrs["float"] = _np.float64
        legacy_attrs["bool"] = _np.bool_
        legacy_attrs["asscalar"] = asscalar
        legacy_attrs["matrix"] = matrix
    elif version == "2.x":
        legacy_attrs.clear()

def __getattr__(name):
    # BACKEND-AWARE FUNCTIONS (dispatch to appropriate backend):
    # - array: Creates arrays in current backend
    # - dot: Matrix multiplication using backend operations
    # - matmul: Matrix multiplication using backend operations
    # - mean: Mean calculation using backend operations
    # - sum: Sum calculation using backend operations
    # - stack: Array stacking using backend operations
    # - concatenate: Array concatenation using backend operations
    # - eye: Identity matrix creation using backend operations
    # - zeros: Zero array creation using backend operations
    # - ones: Ones array creation using backend operations
    # - reshape: Array reshaping using backend operations
    # - transpose: Array transposition using backend operations
    # - astype: Type conversion using backend operations
    # - clip: Value clipping using backend operations
    # - where: Conditional selection using backend operations
    # - linalg.inv: Matrix inversion using backend operations
    #
    # All other functions fall back to NumPy or raise AttributeError
    # in non-NumPy backends to prevent silent mixing of array types.

    if name in legacy_attrs:
        return legacy_attrs[name]
    if name == "array":
        return _array
    if name == "dot":
        return _dot
    if name == "matmul":
        return _matmul
    if name == "mean":
        return _mean
    if name == "sum":
        return _sum
    if name == "stack":
        return _stack
    if name == "concatenate":
        return _concatenate
    if name == "eye":
        return _eye
    if name == "zeros":
        return _zeros
    if name == "ones":
        return _ones
    if name == "reshape":
        return _reshape
    if name == "transpose":
        return _transpose
    if name == "astype":
        return _astype
    if name == "clip":
        return _clip
    if name == "where":
        return _where
    if name == "add":
        return _add
    if name == "subtract":
        return _subtract
    if name == "multiply":
        return _multiply
    if name == "divide":
        return _divide
    if name == "linalg":
        if BACKEND == "numpy":
            return _np.linalg
        else:
            return LinalgWrapper(BACKEND)
    if name == "random":
        if BACKEND == "numpy":
            return _np.random
        else:
            return RandomWrapper(BACKEND)
    return getattr(_np, name)

# Load configuration
import os
import json

config = {}
config_path = os.path.expanduser("~/.omninumpy.json")
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

# Apply config settings
if "jit" in config:
    JIT_ENABLED = config["jit"]

if "jit_threshold" in config:
    JIT_THRESHOLD = config["jit_threshold"]

# Auto-select backend if requested
if config.get("auto_select_backend"):
    set_backend(auto_backend())
elif "backend" in config:
    backend = config["backend"]
    if "device" in config and backend == "jax":
        backend = f"{backend}:{config['device']}"
    set_backend(backend)

if "profile" in config:
    emulate(config["profile"])

# Environment variables override
backend_env = os.getenv("OMNINP_BACKEND")
if backend_env:
    set_backend(backend_env)

profile_env = os.getenv("OMNINP_PROFILE")
if profile_env:
    emulate(profile_env)

jit_env = os.getenv("OMNINP_JIT")
if jit_env is not None:
    JIT_ENABLED = jit_env.lower() in ("true", "1", "yes")

jit_threshold_env = os.getenv("OMNINP_JIT_THRESHOLD")
if jit_threshold_env:
    try:
        JIT_THRESHOLD = int(jit_threshold_env)
    except ValueError:
        pass

