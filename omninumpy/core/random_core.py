from omninumpy.core import array
from omninumpy.core.array_core import array
import omninumpy.backend as backend

def _get_backend():
    return backend.get_current_backend_module()
def random(shape):
    return _get_backend().random.random(shape)

def randn(shape):
    return _get_backend().random.randn(shape)
