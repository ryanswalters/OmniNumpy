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
