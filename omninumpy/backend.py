"""
Backend detection and management for OmniNumpy (safe edition).
"""

from typing import Any, List, Optional
import importlib

import numpy as np

class BackendManager:
    def __init__(self):
        self._backends = {"numpy": np}
        self._current_backend = "numpy"
        self._discover()

    def _discover(self):
        for name in ["cupy", "jax"]:
            try:
                mod = importlib.import_module(name if name != "jax" else "jax.numpy")
                self._backends[name] = mod
                self._current_backend = name
            except ImportError:
                continue

    def get_backend(self) -> str:
        return self._current_backend

    def set_backend(self, backend_name: str):
        if backend_name not in self._backends:
            raise ValueError(f"Backend {backend_name} not available")
        self._current_backend = backend_name

    def list_backends(self) -> List[str]:
        return list(self._backends.keys())

    def get_backend_module(self, backend_name: Optional[str] = None) -> Any:
        return self._backends.get(backend_name or self._current_backend, np)

_manager = BackendManager()

def get_backend() -> str: return _manager.get_backend()
def set_backend(name: str): return _manager.set_backend(name)
def list_backends() -> List[str]: return _manager.list_backends()
def get_backend_module(name: Optional[str] = None) -> Any: return _manager.get_backend_module(name)
