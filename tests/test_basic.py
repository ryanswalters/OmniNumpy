"""Basic tests for OmniNumpy functionality."""

import pytest
import sys
import os

# Add the parent directory to the path so we can import omninumpy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import omninumpy as onp


class TestBasicFunctionality:
    """Test basic NumPy-like functionality."""
    
    def test_import(self):
        """Test that omninumpy can be imported."""
        assert onp is not None
        assert hasattr(onp, '__version__')
    
    def test_backend_management(self):
        """Test backend management functions."""
        # Test that we can get the current backend
        backend = onp.get_backend()
        assert isinstance(backend, str)
        
        # Test that we can list backends
        backends = onp.list_backends()
        assert isinstance(backends, list)
        assert len(backends) > 0
        assert 'numpy' in backends  # NumPy should always be available
    
    def test_array_creation(self):
        """Test basic array creation functions."""
        # Test array()
        arr = onp.array([1, 2, 3])
        assert arr.shape == (3,)
        
        # Test zeros()
        zeros = onp.zeros((2, 3))
        assert zeros.shape == (2, 3)
        
        # Test ones()
        ones = onp.ones(5)
        assert ones.shape == (5,)
        
        # Test empty()
        empty = onp.empty((2, 2))
        assert empty.shape == (2, 2)
    
    def test_mathematical_operations(self):
        """Test basic mathematical operations."""
        a = onp.array([1, 2, 3])
        b = onp.array([4, 5, 6])
        
        # Test element-wise operations
        result_add = onp.add(a, b)
        assert result_add.shape == (3,)
        
        result_sub = onp.subtract(a, b)
        assert result_sub.shape == (3,)
        
        result_mul = onp.multiply(a, b)
        assert result_mul.shape == (3,)
        
        result_div = onp.divide(a, b)
        assert result_div.shape == (3,)
    
    def test_reduction_operations(self):
        """Test reduction operations."""
        arr = onp.array([[1, 2], [3, 4]])
        
        # Test sum
        total = onp.sum(arr)
        assert total == 10 or float(total) == 10.0
        
        # Test mean
        avg = onp.mean(arr)
        assert avg == 2.5 or float(avg) == 2.5
    
    def test_shape_operations(self):
        """Test shape manipulation."""
        arr = onp.array([[1, 2], [3, 4]])
        
        # Test reshape
        reshaped = onp.reshape(arr, (4,))
        assert reshaped.shape == (4,)
        
        # Test transpose
        transposed = onp.transpose(arr)
        assert transposed.shape == (2, 2)
    
    def test_dtypes_available(self):
        """Test that common dtypes are available."""
        # These should not raise AttributeError
        assert hasattr(onp, 'int32')
        assert hasattr(onp, 'float64')
        assert hasattr(onp, 'bool_')
    
    def test_constants_available(self):
        """Test that mathematical constants are available."""
        assert hasattr(onp, 'pi')
        # pi should be approximately 3.14159
        pi_val = float(onp.pi)
        assert 3.14 < pi_val < 3.15


class TestCompatibility:
    """Test NumPy compatibility features."""
    
    def test_numpy_like_interface(self):
        """Test that the interface behaves like NumPy."""
        # Should be able to access many NumPy functions
        assert hasattr(onp, 'array')
        assert hasattr(onp, 'zeros')
        assert hasattr(onp, 'ones')
        assert hasattr(onp, 'sum')
        assert hasattr(onp, 'mean')
        
        # Should handle common NumPy patterns
        arr = onp.array([1, 2, 3, 4, 5])
        result = onp.sum(arr)
        assert result == 15 or float(result) == 15.0
    
    def test_backend_switching(self):
        """Test switching between backends."""
        original_backend = onp.get_backend()
        
        # Try to set to numpy (should always be available)
        onp.set_backend('numpy')
        assert onp.get_backend() == 'numpy'
        
        # Switch back to original
        onp.set_backend(original_backend)
        assert onp.get_backend() == original_backend
    
    def test_invalid_backend(self):
        """Test handling of invalid backend names."""
        with pytest.raises(ValueError):
            onp.set_backend('nonexistent_backend')


if __name__ == '__main__':
    pytest.main([__file__])