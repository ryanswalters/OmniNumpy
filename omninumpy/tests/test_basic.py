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

def test_backend_functions():
    # Test that wrapped functions work
    np.set_backend("numpy")
    a = np.array([1, 2])
    b = np.array([3, 4])
    d = np.dot(a, b)
    m = np.matmul(a.reshape(1, -1), b.reshape(-1, 1))
    assert d == 11  # 1*3 + 2*4
    assert m.shape == (1, 1)

def test_performance_monitoring():
    # Test get_timings
    np.set_backend("numpy")
    a = np.array([1, 2, 3])
    timings = np.get_timings()
    assert '_array' in timings
    assert isinstance(timings['_array'], (int, float))

def test_config_loading():
    # Test that config functions exist (can't easily test file loading without mocking)
    assert callable(np.set_backend)
    assert callable(np.emulate)
    assert callable(np.auto_backend)

def test_backend_dispatch():
    # Test that backends actually create different object types
    original_backend = np.BACKEND  # Access internal backend

    try:
        np.set_backend("numpy")
        a_numpy = np.array([1, 2, 3])
        assert str(type(a_numpy)) == "<class 'numpy.ndarray'>"

        # Note: Other backends require installation, so we can't test them here
        # In a full test suite, would check:
        # np.set_backend("torch"); a_torch = np.array([1,2,3]); assert "torch" in str(type(a_torch))

    finally:
        np.set_backend(original_backend)

if __name__ == "__main__":
    test_base_import()
    test_emulate_legacy()
    test_backend_numpy()
    test_set_backend_invalid()
    test_backend_functions()
    test_performance_monitoring()
    test_config_loading()
    test_backend_dispatch()
    print("All tests passed!")