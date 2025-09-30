#!/usr/bin/env python3
"""
Examples demonstrating OmniNumpy functionality.

This script shows how OmniNumpy provides a drop-in replacement for NumPy
with automatic backend detection and GPU acceleration support.
"""

import sys
import os
sys.path.insert(0, '.')

import omninumpy as np


def main():
    print("=" * 60)
    print("OmniNumpy Examples")
    print("=" * 60)
    
    # Show backend information
    print(f"OmniNumpy version: {np.__version__}")
    print(f"Current backend: {np.get_backend()}")
    print(f"Available backends: {np.list_backends()}")
    print()
    
    # Basic array operations
    print("Basic Array Operations:")
    print("-" * 30)
    
    # Array creation
    arr1 = np.array([1, 2, 3, 4, 5])
    arr2 = np.array([2, 4, 6, 8, 10])
    print(f"arr1 = {arr1}")
    print(f"arr2 = {arr2}")
    
    # Mathematical operations
    print(f"arr1 + arr2 = {np.add(arr1, arr2)}")
    print(f"arr1 * arr2 = {np.multiply(arr1, arr2)}")
    print(f"sum(arr1) = {np.sum(arr1)}")
    print(f"mean(arr1) = {np.mean(arr1)}")
    print()
    
    # Matrix operations
    print("Matrix Operations:")
    print("-" * 30)
    
    # Create matrices
    matrix1 = np.array([[1, 2], [3, 4]])
    matrix2 = np.array([[5, 6], [7, 8]])
    print(f"matrix1 =\n{matrix1}")
    print(f"matrix2 =\n{matrix2}")
    
    # Matrix multiplication
    result = np.dot(matrix1, matrix2)
    print(f"matrix1 @ matrix2 =\n{result}")
    print()
    
    # Array creation functions
    print("Array Creation Functions:")
    print("-" * 30)
    
    zeros = np.zeros((3, 3))
    ones = np.ones((2, 4))
    print(f"zeros(3,3) =\n{zeros}")
    print(f"ones(2,4) =\n{ones}")
    print()
    
    # NumPy 2.x compatibility demonstration
    print("NumPy 2.x Compatibility:")
    print("-" * 30)
    
    # These operations work the same way regardless of NumPy version
    data = np.array([1.5, 2.7, 3.2, 4.8, 5.1])
    print(f"data = {data}")
    print(f"sum with keepdims = {np.sum(data, keepdims=True)}")
    
    # Shape operations
    reshaped = np.reshape(data, (5, 1))
    print(f"reshaped to (5,1) =\n{reshaped}")
    print()
    
    # Constants and dtypes
    print("Constants and Data Types:")
    print("-" * 30)
    print(f"pi = {np.pi}")
    print(f"Available dtypes: int32, float64, etc.")
    
    # Create arrays with specific dtypes
    int_array = np.array([1, 2, 3], dtype=np.int32)
    float_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    print(f"int32 array: {int_array} (dtype: {int_array.dtype})")
    print(f"float64 array: {float_array} (dtype: {float_array.dtype})")
    print()
    
    print("=" * 60)
    print("OmniNumpy: Drop-in NumPy replacement with GPU acceleration!")
    print("Your existing NumPy code works without any changes.")
    print("=" * 60)


if __name__ == "__main__":
    main()