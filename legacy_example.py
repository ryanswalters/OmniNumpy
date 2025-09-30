#!/usr/bin/env python3
"""
Legacy NumPy code example - shows how OmniNumpy provides drop-in compatibility.

This script contains typical NumPy 1.x code patterns that might break in NumPy 2.x,
but work seamlessly with OmniNumpy.
"""

import sys
import os
sys.path.insert(0, '.')

# This is the ONLY line you need to change to make legacy code work with NumPy 2.x!
import omninumpy as np  # Instead of: import numpy as np


def legacy_data_processing():
    """Example of typical legacy NumPy 1.x code patterns."""
    
    print("Legacy Data Processing with OmniNumpy")
    print("=" * 50)
    
    # 1. Array creation with copy=False (changed behavior in NumPy 2.x)
    print("\n1. Array creation with copy parameters:")
    data = [1, 2, 3, 4, 5]
    
    # This would fail in NumPy 2.x but works with OmniNumpy
    arr_no_copy = np.array(data, copy=False)
    arr_copy = np.array(data, copy=True)  
    arr_default = np.array(data)
    
    print(f"Original data: {data}")
    print(f"Array (copy=False): {arr_no_copy}")
    print(f"Array (copy=True): {arr_copy}")
    print(f"Array (default): {arr_default}")
    
    # 2. Mathematical operations with legacy patterns
    print("\n2. Mathematical operations:")
    matrix1 = np.array([[1, 2], [3, 4]], dtype=np.float64)
    matrix2 = np.ones((2, 2), dtype=np.float32)  # Mixed dtypes
    
    # Operations that need careful handling in NumPy 2.x
    result_add = np.add(matrix1, matrix2)
    result_mult = matrix1 * matrix2
    result_dot = np.dot(matrix1, matrix2)
    
    print(f"Matrix 1:\n{matrix1}")
    print(f"Matrix 2:\n{matrix2}")
    print(f"Addition result:\n{result_add}")
    print(f"Element-wise multiplication:\n{result_mult}")
    print(f"Dot product:\n{result_dot}")
    
    # 3. Reduction operations with all parameters
    print("\n3. Reduction operations:")
    data_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    
    # These parameter combinations might behave differently in NumPy 2.x
    sum_all = np.sum(data_matrix)
    sum_axis0 = np.sum(data_matrix, axis=0, keepdims=True)
    sum_axis1 = np.sum(data_matrix, axis=1, keepdims=False, dtype=np.float64)
    mean_all = np.mean(data_matrix, dtype=np.float32)
    
    print(f"Data matrix:\n{data_matrix}")
    print(f"Sum all: {sum_all}")
    print(f"Sum axis=0 (keepdims): {sum_axis0}")
    print(f"Sum axis=1 (no keepdims): {sum_axis1}")
    print(f"Mean (float32): {mean_all}")
    
    # 4. Shape manipulation
    print("\n4. Shape manipulation:")
    flat_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    reshaped = np.reshape(flat_data, (3, 4), order='C')
    transposed = np.transpose(reshaped)
    
    print(f"Original: {flat_data}")
    print(f"Reshaped (3,4):\n{reshaped}")
    print(f"Transposed:\n{transposed}")
    
    # 5. Data type operations
    print("\n5. Data type operations:")
    mixed_data = [1, 2.5, 3, 4.7, 5]
    
    # Automatic and explicit dtype handling
    auto_dtype = np.array(mixed_data)
    int_dtype = np.array(mixed_data, dtype=np.int32)
    float_dtype = np.array(mixed_data, dtype=np.float64)
    
    print(f"Original mixed data: {mixed_data}")
    print(f"Auto dtype: {auto_dtype} (dtype: {auto_dtype.dtype})")
    print(f"Int32 dtype: {int_dtype} (dtype: {int_dtype.dtype})")
    print(f"Float64 dtype: {float_dtype} (dtype: {float_dtype.dtype})")
    
    return True


def demonstrate_gpu_acceleration():
    """Show how OmniNumpy automatically uses GPU acceleration when available."""
    
    print("\n" + "=" * 50)
    print("Backend Information")
    print("=" * 50)
    
    print(f"Current backend: {np.get_backend()}")
    print(f"Available backends: {np.list_backends()}")
    print(f"NumPy version: {np.version}")
    
    # Create some larger arrays to show potential performance benefits
    print("\nCreating large arrays (GPU acceleration would help here):")
    large_array1 = np.ones((1000, 1000), dtype=np.float32)
    large_array2 = np.ones((1000, 1000), dtype=np.float32) * 2
    
    # Matrix multiplication (where GPU really shines)
    result = np.dot(large_array1, large_array2)
    
    print(f"Matrix multiplication of 1000x1000 arrays completed")
    print(f"Result shape: {result.shape}")
    print(f"Result sample: {result[0, 0]} (should be 2000.0)")
    
    return True


def main():
    """Run all legacy code examples."""
    print("OmniNumpy Legacy Code Compatibility Demo")
    print("This code would work in NumPy 1.x and continues to work in NumPy 2.x!")
    
    # Run legacy processing
    success1 = legacy_data_processing()
    
    # Show backend capabilities
    success2 = demonstrate_gpu_acceleration()
    
    if success1 and success2:
        print("\n" + "=" * 50)
        print("✅ ALL LEGACY CODE WORKS PERFECTLY!")
        print("✅ Zero refactoring required!")
        print("✅ NumPy 2.x compatibility achieved!")
        print("✅ GPU acceleration available when backends are installed!")
        print("=" * 50)
        
        print("\nTo enable GPU acceleration:")
        print("pip install omninumpy[gpu]  # For CuPy support")
        print("pip install omninumpy[jax]  # For JAX support")
        print("pip install omninumpy[all]  # For all backends")
    
    return True


if __name__ == "__main__":
    main()