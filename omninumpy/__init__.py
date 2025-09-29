import numpy as _np
import time
from functools import wraps

BACKEND = "numpy"
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

def _get_jax_numpy():
    if not hasattr(_get_jax_numpy, '_jnp'):
        import jax.numpy as jnp
        _get_jax_numpy._jnp = jnp
    return _get_jax_numpy._jnp

def auto_backend():
    """Picks fastest available backend"""
    try:
        torch = _get_torch()
        if torch.cuda.is_available():
            return "torch"
    except ImportError:
        pass
    try:
        _get_cupy()
        return "cupy"
    except ImportError:
        pass
    return "numpy"

def set_backend(name):
    global BACKEND
    if name in ("numpy", "torch", "cupy", "jax"):
        BACKEND = name
    else:
        raise ValueError("Unsupported backend")

def _array(x, *args, **kwargs):
    if BACKEND == "torch":
        torch = _get_torch()
        return torch.tensor(x, *args, **kwargs)
    elif BACKEND == "cupy":
        cp = _get_cupy()
        return cp.array(x, *args, **kwargs)
    elif BACKEND == "jax":
        jnp = _get_jax_numpy()
        return jnp.array(x, *args, **kwargs)
    else:
        return _np.array(x, *args, **kwargs)

_array = _track_perf(_array)

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
    if name in legacy_attrs:
        return legacy_attrs[name]
    if name == "array":
        return _array
    return getattr(_np, name)

# Load configuration
import os
import json

config = {}
config_path = os.path.expanduser("~/.omninumpy.json")
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

# Auto-select backend if requested
if config.get("auto_select_backend"):
    set_backend(auto_backend())
elif "backend" in config:
    set_backend(config["backend"])

if "profile" in config:
    emulate(config["profile"])

# Environment variables override
backend_env = os.getenv("OMNINP_BACKEND")
if backend_env:
    set_backend(backend_env)

profile_env = os.getenv("OMNINP_PROFILE")
if profile_env:
    emulate(profile_env)