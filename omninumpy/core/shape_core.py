from omninumpy.core import array
from omninumpy.core.array_core import array
import omninumpy.backend as backend

__all__ = [
    "shape", "reshape", "transpose",
]           

def _get_backend():
    return backend.get_current_backend_module()
def shape(a):
    return _get_backend().shape(a)

def reshape(a, newshape):
    return _get_backend().reshape(a, newshape)

def transpose(a, axes=None):
    return _get_backend().transpose(a, axes)
