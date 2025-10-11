from omninumpy.core import array
from omninumpy.core.array_core import array
import omninumpy.backend as backend
legacy_attrs = {}

    
def _get_backend():
    return backend.get_current_backend_module()

def emulate(version="2.x"):
    if version.startswith("1."):
        legacy_attrs["int"] = getattr(backend, "int64", int)
        legacy_attrs["bool"] = getattr(backend, "bool", bool)
    else:
        legacy_attrs.clear()
