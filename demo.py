#!/usr/bin/env python3
"""
OmniNumpy Demo - Complete showcase of features and value proposition.

This demo shows how OmniNumpy solves the "NumPy version hell" problem
while adding GPU acceleration capabilities.
"""

import sys
import os
sys.path.insert(0, '.')

import omninumpy as np
import time


def show_header():
    """Display the demo header."""
    print("*" * 70)
    print("            🚀 OMNINUMPY DEMO 🚀")
    print("     Stop fighting NumPy version hell!")
    print("*" * 70)
    print()


def show_basic_info():
    """Show basic information about OmniNumpy."""
    print("📋 BASIC INFORMATION")
    print("-" * 40)
    print(f"OmniNumpy version: {np.__version__}")
    print(f"Current backend: {np.get_backend()}")
    print(f"Available backends: {np.list_backends()}")
    print(f"Backend version: {np.version}")
    print()


def demonstrate_drop_in_replacement():
    """Show that OmniNumpy is a true drop-in replacement."""
    print("🔄 DROP-IN REPLACEMENT DEMO")
    print("-" * 40)
    print("The ONLY change needed in your code:")
    print("  OLD: import numpy as np")
    print("  NEW: import omninumpy as np")
    print()
    
    print("All your existing NumPy code works unchanged:")
    
    # Standard NumPy operations
    arr = np.array([1, 2, 3, 4, 5])
    print(f"  np.array([1,2,3,4,5]) = {arr}")
    
    matrix = np.zeros((3, 3))
    print(f"  np.zeros((3,3)) shape = {matrix.shape}")
    
    result = np.sum(arr) * np.pi
    print(f"  np.sum(arr) * np.pi = {result:.4f}")
    
    print("✅ Zero refactoring required!")
    print()


def demonstrate_numpy2_compatibility():
    """Show NumPy 2.x compatibility features."""
    print("🔧 NUMPY 2.x COMPATIBILITY")
    print("-" * 40)
    
    print("These operations work in both NumPy 1.x and 2.x:")
    
    # NumPy 2.x copy parameter handling
    data = [1, 2, 3, 4, 5]
    
    print("  Array creation with different copy parameters:")
    arr1 = np.array(data, copy=True)
    print(f"    copy=True: {arr1}")
    
    arr2 = np.array(data, copy=False)  # This fails in raw NumPy 2.x
    print(f"    copy=False: {arr2}")
    
    arr3 = np.array(data, copy=None)
    print(f"    copy=None: {arr3}")
    
    # Reduction operations with various parameters
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    sum_keepdims = np.sum(matrix, axis=1, keepdims=True)
    print(f"  Sum with keepdims: {sum_keepdims.shape}")
    
    print("✅ Handles NumPy 2.x breaking changes automatically!")
    print()


def demonstrate_performance():
    """Show performance characteristics."""
    print("⚡ PERFORMANCE DEMO")
    print("-" * 40)
    
    # Create moderately large arrays
    size = 1000
    print(f"Creating {size}x{size} matrices for performance test...")
    
    start_time = time.time()
    
    # Matrix operations that would benefit from GPU acceleration
    a = np.random.rand(size, size).astype(np.float32)
    b = np.random.rand(size, size).astype(np.float32)
    
    # Matrix multiplication
    c = np.dot(a, b)
    
    # Some reductions
    sum_result = np.sum(c)
    mean_result = np.mean(c)
    
    end_time = time.time()
    
    print(f"  Matrix multiplication: {size}x{size} @ {size}x{size}")
    print(f"  Result sum: {sum_result:.2f}")
    print(f"  Result mean: {mean_result:.4f}")
    print(f"  Time taken: {end_time - start_time:.3f} seconds")
    print(f"  Backend used: {np.get_backend()}")
    
    if np.get_backend() == 'numpy':
        print("  💡 Install CuPy or JAX for GPU acceleration!")
    else:
        print("  🚀 GPU acceleration active!")
    
    print()


def demonstrate_backend_management():
    """Show backend management capabilities."""
    print("🔧 BACKEND MANAGEMENT")
    print("-" * 40)
    
    current = np.get_backend()
    available = np.list_backends()
    
    print(f"Current backend: {current}")
    print(f"Available backends: {available}")
    
    # Try switching backends
    if len(available) > 1:
        for backend in available:
            if backend != current:
                print(f"  Switching to {backend}...")
                np.set_backend(backend)
                test_arr = np.array([1, 2, 3])
                print(f"  Test array: {test_arr} (backend: {np.get_backend()})")
                break
    
    # Switch back to original
    np.set_backend(current)
    print(f"  Switched back to: {np.get_backend()}")
    print()


def demonstrate_real_world_use_case():
    """Show a real-world machine learning-like use case."""
    print("🧠 REAL-WORLD USE CASE: ML-style Operations")
    print("-" * 40)
    
    print("Simulating typical machine learning operations...")
    
    # Simulate some data
    batch_size, features = 100, 50
    X = np.random.randn(batch_size, features).astype(np.float32)
    y = np.random.randn(batch_size, 1).astype(np.float32)
    W = np.random.randn(features, 1).astype(np.float32)
    
    print(f"  Data shapes: X{X.shape}, y{y.shape}, W{W.shape}")
    
    # Forward pass
    predictions = np.dot(X, W)
    
    # Loss computation (MSE)
    loss = np.mean((predictions - y) ** 2)
    
    # Some statistics
    X_mean = np.mean(X, axis=0, keepdims=True)
    X_std = np.std(X, axis=0, keepdims=True)
    
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  MSE Loss: {loss:.6f}")
    print(f"  Input mean range: [{X_mean.min():.3f}, {X_mean.max():.3f}]")
    print(f"  Input std range: [{X_std.min():.3f}, {X_std.max():.3f}]")
    
    print("✅ All operations completed successfully!")
    print("🚀 This would run much faster on GPU with CuPy/JAX!")
    print()


def show_installation_guide():
    """Show installation options."""
    print("📦 INSTALLATION OPTIONS")
    print("-" * 40)
    print("Basic installation:")
    print("  pip install omninumpy")
    print()
    print("With GPU support (NVIDIA):")
    print("  pip install omninumpy[gpu]")
    print()
    print("With JAX support (CPU/GPU/TPU):")
    print("  pip install omninumpy[jax]")
    print()
    print("With all backends:")
    print("  pip install omninumpy[all]")
    print()


def show_value_proposition():
    """Show the key value propositions."""
    print("💎 VALUE PROPOSITION")
    print("-" * 40)
    print("✅ ZERO REFACTORING: Drop-in NumPy replacement")
    print("✅ NUMPY 2.x COMPATIBILITY: Handles breaking changes automatically")
    print("✅ GPU ACCELERATION: Automatic CuPy/JAX backend selection")
    print("✅ BACKEND AGNOSTIC: Works with NumPy, CuPy, JAX, and more")
    print("✅ LEGACY SUPPORT: Your old code keeps working")
    print("✅ FUTURE PROOF: Ready for new NumPy versions")
    print("✅ PERFORMANCE: GPU acceleration without code changes")
    print("✅ SIMPLE: Just change your import statement")
    print()


def main():
    """Run the complete demo."""
    show_header()
    show_basic_info()
    demonstrate_drop_in_replacement()
    demonstrate_numpy2_compatibility()
    demonstrate_performance()
    demonstrate_backend_management()
    demonstrate_real_world_use_case()
    show_installation_guide()
    show_value_proposition()
    
    print("*" * 70)
    print("  🎉 OMNINUMPY: THE NUMPY COMPATIBILITY SOLUTION 🎉")
    print("     Stop fighting version hell. Start accelerating.")
    print("*" * 70)


if __name__ == "__main__":
    main()