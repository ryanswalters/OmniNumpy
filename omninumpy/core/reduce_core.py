from omninumpy.core import array
from omninumpy.core.array_core import array
import omninumpy.backend as backend

def _get_backend():
    return backend.get_current_backend_module()

def sum(a, axis=None, dtype=None, keepdims=False):
    return _get_backend().sum(a, axis=axis, dtype=dtype, keepdims=keepdims)

def mean(a, axis=None, dtype=None, keepdims=False):
    return _get_backend().mean(a, axis=axis, dtype=dtype, keepdims=keepdims)
