from omninumpy.core import array
from omninumpy.core.array_core import array
import omninumpy.backend as backend
def _get_backend():
    return backend.get_current_backend_module()

def dot(a, b):
    return _get_backend().dot(a, b)

def matmul(a, b):
    return _get_backend().matmul(a, b) if hasattr(_get_backend(), "matmul") else _get_backend().dot(a, b)
