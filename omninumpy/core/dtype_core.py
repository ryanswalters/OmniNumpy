from omninumpy.core import array
from omninumpy.core.array_core import array
import omninumpy.backend as backend

def _get_backend():
    return backend.get_current_backend_module()

def int32():
    return _get_backend().int32

def float64():
    return _get_backend().float64

def pi():
    return _get_backend().pi
