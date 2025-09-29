"""
Full NumPy-compatible interface for OmniNumpy.

This module dynamically exposes all functions from the current backend
while providing compatibility wrappers where needed.
"""

import sys
from . import core
from . import backend as backend_module


class OmniNumpyModule:
    """
    A module-like object that provides full NumPy compatibility.
    
    This class dynamically delegates attribute access to the current backend
    while providing compatibility wrappers for functions that need them.
    """
    
    # Functions that need special compatibility handling
    _compatibility_functions = {
        'array', 'zeros', 'ones', 'empty', 'sum', 'mean', 
        'reshape', 'transpose', 'add', 'subtract', 'multiply', 
        'divide', 'dot', 'matmul'
    }
    
    # Properties that should come from core module
    _core_properties = {
        'int8', 'int16', 'int32', 'int64',
        'float16', 'float32', 'float64', 
        'complex64', 'complex128', 'bool_',
        'pi', 'e'
    }
    
    def __getattr__(self, name):
        # Handle backend management functions
        if name in ('get_backend', 'set_backend', 'list_backends'):
            return getattr(backend_module, name)
        
        # Handle compatibility functions
        if name in self._compatibility_functions:
            return getattr(core, name)
        
        # Handle core properties (dtypes, constants)
        if name in self._core_properties:
            backend = core._get_backend()
            if hasattr(backend, name):
                return getattr(backend, name)
            else:
                # Fallback for missing attributes
                import numpy as np
                return getattr(np, name)
        
        # For everything else, delegate to the current backend
        backend = core._get_backend()
        if hasattr(backend, name):
            attr = getattr(backend, name)
            
            # If it's a function, we might need to wrap it for compatibility
            if callable(attr) and name not in self._compatibility_functions:
                return self._wrap_function(attr, name)
            
            return attr
        
        # If not found in backend, try numpy as fallback
        try:
            import numpy as np
            if hasattr(np, name):
                return getattr(np, name)
        except ImportError:
            pass
        
        raise AttributeError(f"module 'omninumpy' has no attribute '{name}'")
    
    def _wrap_function(self, func, name):
        """Wrap backend functions to handle common compatibility issues."""
        def wrapped(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except TypeError as e:
                # Handle common parameter incompatibilities
                if 'unexpected keyword argument' in str(e):
                    # Try removing common problematic parameters
                    filtered_kwargs = kwargs.copy()
                    for param in ['out', 'where', 'order', 'subok', 'like']:
                        filtered_kwargs.pop(param, None)
                    
                    try:
                        return func(*args, **filtered_kwargs)
                    except TypeError:
                        # Last resort: try with positional args only
                        return func(*args)
                else:
                    raise
        
        return wrapped
    
    def __dir__(self):
        """Return all available attributes from the current backend."""
        backend = core._get_backend()
        backend_attrs = dir(backend) if backend else []
        
        # Add our special functions
        special_attrs = list(self._compatibility_functions)
        special_attrs.extend(['get_backend', 'set_backend', 'list_backends'])
        
        return sorted(set(backend_attrs + special_attrs))
    
    @property
    def __version__(self):
        """Return the OmniNumpy version."""
        from .version import __version__
        return __version__
    
    @property 
    def version(self):
        """Return version info for the current backend."""
        backend = core._get_backend()
        if hasattr(backend, 'version'):
            return backend.version
        elif hasattr(backend, '__version__'):
            return backend.__version__
        else:
            return "unknown"


# Create the module instance
_omni_module = OmniNumpyModule()

# Make this module behave like the OmniNumpyModule instance
sys.modules[__name__] = _omni_module