"""Tests for NumPy 2.x compatibility features."""

import pytest
import sys
import os

# Add the parent directory to the path so we can import omninumpy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import omninumpy as onp
import numpy as np


class TestNumPy2xCompatibility:
    """Test NumPy 2.x specific compatibility features."""
    
    def test_array_copy_parameter(self):
        """Test that the copy parameter works correctly."""
        original = [1, 2, 3]
        
        # Test copy=True (should work in both NumPy 1.x and 2.x)
        arr1 = onp.array(original, copy=True)
        assert arr1.tolist() == [1, 2, 3]
        
        # Test copy=False
        arr2 = onp.array(original, copy=False)
        assert arr2.tolist() == [1, 2, 3]
        
        # Test copy=None (NumPy 2.x default)
        arr3 = onp.array(original, copy=None)
        assert arr3.tolist() == [1, 2, 3]
    
    def test_dtype_compatibility(self):
        """Test that dtype handling works across NumPy versions."""
        # Test with different dtypes
        arr_int = onp.array([1, 2, 3], dtype=onp.int32)
        assert arr_int.dtype == np.int32
        
        arr_float = onp.array([1.0, 2.0, 3.0], dtype=onp.float64)
        assert arr_float.dtype == np.float64
        
        arr_bool = onp.array([True, False, True], dtype=onp.bool_)
        assert arr_bool.dtype == np.bool_
    
    def test_keepdims_parameter(self):
        """Test keepdims parameter in reduction functions."""
        arr = onp.array([[1, 2], [3, 4]])
        
        # Test sum with keepdims
        result = onp.sum(arr, keepdims=True)
        assert result.shape == (1, 1)
        
        result_axis = onp.sum(arr, axis=0, keepdims=True)
        assert result_axis.shape == (1, 2)
    
    def test_parameter_filtering(self):
        """Test that incompatible parameters are filtered correctly."""
        arr = onp.array([1, 2, 3, 4, 5])
        
        # These should work even if the backend doesn't support all parameters
        result = onp.sum(arr, axis=None, dtype=None, keepdims=False)
        assert result == 15 or float(result) == 15.0
    
    def test_backend_switching_preserves_functionality(self):
        """Test that functionality is preserved when switching backends."""
        original_backend = onp.get_backend()
        
        # Test basic operations
        arr = onp.array([1, 2, 3])
        result1 = onp.sum(arr)
        
        # Switch to numpy explicitly (should always be available)
        onp.set_backend('numpy')
        arr2 = onp.array([1, 2, 3])
        result2 = onp.sum(arr2)
        
        # Results should be the same
        assert result1 == result2 or float(result1) == float(result2)
        
        # Switch back
        onp.set_backend(original_backend)
    
    def test_array_function_protocol(self):
        """Test that arrays work with standard operations."""
        arr1 = onp.array([1, 2, 3])
        arr2 = onp.array([4, 5, 6])
        
        # Test that basic Python operations work
        result = arr1 + arr2  # Should use NumPy's __add__
        expected = onp.array([5, 7, 9])
        
        # Compare element-wise
        assert (result == expected).all()
    
    def test_numpy_version_detection(self):
        """Test that we can detect and work with the underlying NumPy version."""
        # Should have access to version information
        version_info = onp.version
        assert version_info is not None
        
        # Should be able to create arrays regardless of version
        arr = onp.array([1.0, 2.0, 3.0])
        assert arr.shape == (3,)
        assert arr.dtype in [np.float64, np.float32]  # Depends on system default


class TestAdvancedCompatibility:
    """Test advanced compatibility features."""
    
    def test_matrix_operations(self):
        """Test matrix operations work correctly."""
        a = onp.array([[1, 2], [3, 4]])
        b = onp.array([[5, 6], [7, 8]])
        
        # Test dot product
        result = onp.dot(a, b)
        expected = [[19, 22], [43, 50]]
        
        # Check result shape and values
        assert result.shape == (2, 2)
        for i in range(2):
            for j in range(2):
                assert result[i, j] == expected[i][j]
    
    def test_shape_manipulation(self):
        """Test that shape manipulation functions work."""
        arr = onp.array([1, 2, 3, 4, 5, 6])
        
        # Test reshape
        reshaped = onp.reshape(arr, (2, 3))
        assert reshaped.shape == (2, 3)
        
        # Test transpose
        transposed = onp.transpose(reshaped)
        assert transposed.shape == (3, 2)
    
    def test_mathematical_functions(self):
        """Test mathematical functions are available."""
        arr = onp.array([1, 4, 9, 16])
        
        # Test square root (should be available from backend)
        sqrt_result = onp.sqrt(arr)
        expected = [1.0, 2.0, 3.0, 4.0]
        
        for i, val in enumerate(expected):
            assert abs(float(sqrt_result[i]) - val) < 1e-10
    
    def test_error_handling(self):
        """Test proper error handling."""
        # Test invalid backend
        with pytest.raises(ValueError):
            onp.set_backend('nonexistent_backend')
        
        # Test invalid attribute access
        with pytest.raises(AttributeError):
            getattr(onp, 'nonexistent_function_that_does_not_exist')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])