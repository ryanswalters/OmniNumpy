"""
Backend detection and management for OmniNumpy.

This module handles automatic detection and switching between different array backends
like NumPy, CuPy, JAX, etc.
"""

import os
import warnings
from typing import Dict, List, Optional, Any, Union
import importlib


class BackendManager:
    """Manages available array backends and provides automatic backend selection."""
    
    def __init__(self):
        self._backends: Dict[str, Any] = {}
        self._current_backend = None
        self._backend_priority = ['cupy', 'jax', 'numpy']
        self._discover_backends()
    
    def _discover_backends(self):
        """Discover and register all available backends."""
        # NumPy is always available as the fallback
        import numpy as np
        self._backends['numpy'] = np
        self._current_backend = 'numpy'
        
        # Try to import CuPy for GPU acceleration
        try:
            import cupy as cp
            self._backends['cupy'] = cp
            # Prefer CuPy if available and GPU is accessible
            if self._current_backend == 'numpy':
                try:
                    # Test if CuPy can create arrays (GPU available)
                    cp.array([1, 2, 3])
                    self._current_backend = 'cupy'
                except Exception:
                    pass
        except ImportError:
            pass
        
        # Try to import JAX
        try:
            import jax.numpy as jnp
            import jax
            self._backends['jax'] = jnp
            self._backends['jax_module'] = jax
            # JAX has good CPU performance too
            if self._current_backend == 'numpy':
                self._current_backend = 'jax'
        except ImportError:
            pass
    
    def get_backend(self) -> str:
        """Get the name of the current backend."""
        return self._current_backend
    
    def set_backend(self, backend_name: str) -> None:
        """Set the current backend."""
        if backend_name not in self._backends:
            available = list(self._backends.keys())
            raise ValueError(f"Backend '{backend_name}' not available. "
                           f"Available backends: {available}")
        
        self._current_backend = backend_name
        
        # Update the global current backend module
        global _current_backend_module
        _current_backend_module = self._backends[backend_name]
    
    def list_backends(self) -> List[str]:
        """List all available backends."""
        return list(self._backends.keys())
    
    def get_backend_module(self, backend_name: Optional[str] = None) -> Any:
        """Get the backend module."""
        if backend_name is None:
            backend_name = self._current_backend
        return self._backends.get(backend_name)
    
    def auto_backend(self) -> str:
        """Automatically select the best available backend."""
        for backend in self._backend_priority:
            if backend in self._backends:
                # For CuPy, verify GPU is actually available
                if backend == 'cupy':
                    try:
                        cp = self._backends[backend]
                        cp.array([1, 2, 3])
                        self.set_backend(backend)
                        return backend
                    except Exception:
                        continue
                else:
                    self.set_backend(backend)
                    return backend
        
        # Fallback to numpy
        self.set_backend('numpy')
        return 'numpy'


# Global backend manager instance
_backend_manager = BackendManager()
_current_backend_module = None

# Public API functions
def get_backend() -> str:
    """Get the name of the current backend."""
    return _backend_manager.get_backend()

def set_backend(backend_name: str) -> None:
    """Set the current backend."""
    _backend_manager.set_backend(backend_name)

def list_backends() -> List[str]:
    """List all available backends."""
    return _backend_manager.list_backends()

def get_backend_module(backend_name: Optional[str] = None) -> Any:
    """Get the backend module."""
    return _backend_manager.get_backend_module(backend_name)

def auto_backend() -> str:
    """Automatically select the best available backend."""
    return _backend_manager.auto_backend()

def get_current_backend_module():
    """Get the current backend module directly."""
    global _current_backend_module
    if _current_backend_module is None:
        _current_backend_module = _backend_manager.get_backend_module()
    return _current_backend_module

# Initialize with the best available backend
try:
    _current_backend = auto_backend()
except Exception:
    # Fallback to prevent import errors
    _current_backend = 'numpy'