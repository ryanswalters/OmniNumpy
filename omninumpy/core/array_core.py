from omninumpy.core import array
from omninumpy.core.array_core import array
import omninumpy.backend as backend

def _get_backend():
    return backend.get_current_backend_module()

def array(object, dtype=None, *, copy=None, order='K', subok=False, ndmin=0, like=None, **kwargs):
    backend = _get_backend()
    # handle kwargs compatibly
    return backend.array(object, **kwargs)

def zeros(shape, dtype=float, order='C', *, like=None):
    return _get_backend().zeros(shape, dtype=dtype)

def ones(shape, dtype=None, order='C', *, like=None):
    return _get_backend().ones(shape, dtype=dtype)
