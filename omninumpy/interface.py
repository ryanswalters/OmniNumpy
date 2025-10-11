"""
OmniNumpy interface - safe aware
"""

import sys
from . import core
from . import backend

class OmniModule:
    def __getattr__(self, name):
        if hasattr(core, name):
            return getattr(core, name)
        if hasattr(backend, name):
            return getattr(backend, name)
        import numpy as np
        if hasattr(np, name):
            return getattr(np, name)
        raise AttributeError(f"omninumpy has no attribute {name}")

    def __dir__(self):
        return sorted(set(dir(core) + dir(backend)))

sys.modules[__name__] = OmniModule()
