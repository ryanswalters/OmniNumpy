import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import omninumpy as np
import numpy as _np

def test_torch_backend():
    """Test PyTorch backend functionality"""
    try:
        import torch
        original_backend = np.BACKEND

        np.set_backend("torch")

        # Test array creation
        a = np.array([1, 2, 3])
        assert isinstance(a, torch.Tensor), f"Expected torch.Tensor, got {type(a)}"

        # Test basic operations
        b = np.array([4, 5, 6])
        c = np.dot(a, b)
        assert isinstance(c, torch.Tensor), f"Expected torch.Tensor, got {type(c)}"
        assert c.item() == 32, f"Expected 32, got {c.item()}"

        # Test mean
        d = np.mean(a)
        assert isinstance(d, torch.Tensor), f"Expected torch.Tensor, got {type(d)}"

        # Test stack
        e = np.stack([a, b])
        assert isinstance(e, torch.Tensor), f"Expected torch.Tensor, got {type(e)}"
        assert e.shape == (2, 3), f"Expected shape (2, 3), got {e.shape}"

        # Test linalg.inv
        f = np.eye(3)
        g = np.linalg.inv(f)
        assert isinstance(g, torch.Tensor), f"Expected torch.Tensor, got {type(g)}"

        np.set_backend(original_backend)
        return True
    except ImportError:
        print("PyTorch not available, skipping torch tests")
        return True
    except Exception as e:
        print(f"PyTorch test failed: {e}")
        return False

def test_cupy_backend():
    """Test CuPy backend functionality"""
    try:
        import cupy as cp
        original_backend = np.BACKEND

        np.set_backend("cupy")

        # Test array creation
        a = np.array([1, 2, 3])
        assert isinstance(a, cp.ndarray), f"Expected cupy.ndarray, got {type(a)}"

        # Test basic operations
        b = np.array([4, 5, 6])
        c = np.dot(a, b)
        assert isinstance(c, cp.ndarray), f"Expected cupy.ndarray, got {type(c)}"
        assert c.item() == 32, f"Expected 32, got {c.item()}"

        # Test mean
        d = np.mean(a)
        assert isinstance(d, cp.ndarray), f"Expected cupy.ndarray, got {type(d)}"

        # Test stack
        e = np.stack([a, b])
        assert isinstance(e, cp.ndarray), f"Expected cupy.ndarray, got {type(e)}"
        assert e.shape == (2, 3), f"Expected shape (2, 3), got {e.shape}"

        # Test linalg.inv
        f = np.eye(3)
        g = np.linalg.inv(f)
        assert isinstance(g, cp.ndarray), f"Expected cupy.ndarray, got {type(g)}"

        np.set_backend(original_backend)
        return True
    except ImportError:
        print("CuPy not available, skipping cupy tests")
        return True
    except Exception as e:
        print(f"CuPy test failed: {e}")
        return False

def test_jax_backend():
    """Test JAX backend functionality"""
    try:
        import jax.numpy as jnp
        original_backend = np.BACKEND

        np.set_backend("jax")

        # Test array creation
        a = np.array([1, 2, 3])
        assert isinstance(a, jnp.ndarray), f"Expected jax.Array, got {type(a)}"

        # Test basic operations
        b = np.array([4, 5, 6])
        c = np.dot(a, b)
        assert isinstance(c, jnp.ndarray), f"Expected jax.Array, got {type(c)}"
        assert c.item() == 32, f"Expected 32, got {c.item()}"

        # Test mean
        d = np.mean(a)
        assert isinstance(d, jnp.ndarray), f"Expected jax.Array, got {type(d)}"

        # Test stack
        e = np.stack([a, b])
        assert isinstance(e, jnp.ndarray), f"Expected jax.Array, got {type(e)}"
        assert e.shape == (2, 3), f"Expected shape (2, 3), got {e.shape}"

        # Test linalg.inv
        f = np.eye(3)
        g = np.linalg.inv(f)
        assert isinstance(g, jnp.ndarray), f"Expected jax.Array, got {type(g)}"

        np.set_backend(original_backend)
        return True
    except ImportError:
        print("JAX not available, skipping jax tests")
        return True
    except Exception as e:
        print(f"JAX test failed: {e}")
        return False

def test_numerical_correctness():
    """Test that backends produce numerically correct results"""
    test_matrix = [[1, 2], [3, 4]]
    original_backend = np.BACKEND

    # Get reference results from NumPy
    np.set_backend("numpy")
    ref_a = np.array(test_matrix)
    ref_dot = np.dot(ref_a, ref_a)
    ref_mean = np.mean(ref_a)
    ref_inv = np.linalg.inv(ref_a)

    backends_to_test = ["torch", "cupy", "jax"]

    for backend in backends_to_test:
        try:
            np.set_backend(backend)

            a = np.array(test_matrix)
            dot_result = np.dot(a, a)
            mean_result = np.mean(a)
            inv_result = np.linalg.inv(a)

            # Convert to numpy for comparison
            dot_numpy = np.to_numpy(dot_result)
            mean_numpy = np.to_numpy(mean_result)
            inv_numpy = np.to_numpy(inv_result)

            # Check numerical accuracy
            assert _np.allclose(dot_numpy, ref_dot), f"Dot product mismatch for {backend}"
            assert _np.allclose(mean_numpy, ref_mean), f"Mean mismatch for {backend}"
            assert _np.allclose(inv_numpy, ref_inv), f"Inv mismatch for {backend}"

        except ImportError:
            continue  # Skip if backend not available
        except Exception as e:
            print(f"Numerical test failed for {backend}: {e}")
            return False

    np.set_backend(original_backend)
    return True

if __name__ == "__main__":
    results = []
    results.append(("torch", test_torch_backend()))
    results.append(("cupy", test_cupy_backend()))
    results.append(("jax", test_jax_backend()))
    results.append(("numerical", test_numerical_correctness()))

    passed = 0
    total = len(results)

    for name, success in results:
        if success:
            print(f"✓ {name} tests passed")
            passed += 1
        else:
            print(f"✗ {name} tests failed")

    print(f"\n{passed}/{total} test suites passed")
    if passed == total:
        print("All backend tests passed!")
    else:
        print("Some backend tests failed")
        sys.exit(1)