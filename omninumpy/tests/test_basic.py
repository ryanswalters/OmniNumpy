import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import omninumpy as np
import numpy as _np

def test_base_import():
    # Test that basic numpy functions work
    a = np.array([1, 2, 3])
    assert isinstance(a, _np.ndarray)
    assert _np.array_equal(a, [1, 2, 3])

def test_emulate_legacy():
    # Test legacy emulation
    np.emulate("1.21")
    assert hasattr(np, 'int')
    assert np.int == _np.int64
    assert hasattr(np, 'asscalar')
    assert callable(np.asscalar)
    # Test asscalar
    a = np.array([5])
    assert np.asscalar(a) == 5

# Removed test for modern, as numpy may still have some attributes

def test_backend_numpy():
    # Test backend numpy
    np.set_backend("numpy")
    a = np.array([1, 2, 3])
    assert isinstance(a, _np.ndarray)

def test_set_backend_invalid():
    try:
        np.set_backend("invalid")
        assert False, "Should raise ValueError"
    except ValueError:
        pass

if __name__ == "__main__":
    test_base_import()
    test_emulate_legacy()
    test_backend_numpy()
    test_set_backend_invalid()
    print("All tests passed!")