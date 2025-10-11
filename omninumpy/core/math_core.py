from omninumpy.core import array
from omninumpy.core.array_core import array
import omninumpy.backend as backend

def _get_backend():
    return backend.get_current_backend_module()

def add(x1, x2, out=None, **kwargs):
    """Add arguments element-wise."""
    backend = _get_backend()
    return backend.add(x1, x2, out=out, **kwargs)

def subtract(x1, x2, out=None, **kwargs):
    """Subtract arguments element-wise."""
    backend = _get_backend()
    return backend.subtract(x1, x2, out=out, **kwargs)

def multiply(x1, x2, out=None, **kwargs):
    """Multiply arguments element-wise."""
    backend = _get_backend()
    return backend.multiply(x1, x2, out=out, **kwargs)

def divide(x1, x2, out=None, **kwargs):
    """Divide arguments element-wise."""
    backend = _get_backend()
    return backend.divide(x1, x2, out=out, **kwargs)

def power(x1, x2, out=None, **kwargs):
    """Raise arguments element-wise to the power."""
    backend = _get_backend()
    return backend.power(x1, x2, out=out, **kwargs)