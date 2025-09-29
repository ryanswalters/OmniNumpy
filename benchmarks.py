#!/usr/bin/env python3
"""
Benchmark suite for Omninumpy backends
"""
import time
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import omninumpy as np

def benchmark_array_creation(size=1000000, iterations=10):
    """Benchmark array creation across backends"""
    results = {}

    for backend in ["numpy", "torch", "cupy", "jax"]:
        try:
            np.set_backend(backend)
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                a = np.array(list(range(size)))
                _ = a + 1  # Simple operation
                end = time.perf_counter()
                times.append(end - start)
            results[backend] = {
                'mean': sum(times) / len(times),
                'min': min(times),
                'max': max(times)
            }
        except ImportError:
            results[backend] = "Not available"
        except Exception as e:
            results[backend] = f"Error: {e}"

    return results

def benchmark_matrix_multiplication(size=1000, iterations=5):
    """Benchmark matrix multiplication"""
    results = {}

    for backend in ["numpy", "torch", "cupy", "jax"]:
        try:
            np.set_backend(backend)
            # Create matrices
            a = np.random.random((size, size))
            b = np.random.random((size, size))

            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                c = np.dot(a, b)
                end = time.perf_counter()
                times.append(end - start)

            results[backend] = {
                'mean': sum(times) / len(times),
                'min': min(times),
                'max': max(times)
            }
        except ImportError:
            results[backend] = "Not available"
        except Exception as e:
            results[backend] = f"Error: {e}"

    return results

if __name__ == "__main__":
    print("Omninumpy Benchmark Suite")
    print("=" * 40)

    print("\nArray Creation (1M elements):")
    results = benchmark_array_creation()
    for backend, data in results.items():
        if isinstance(data, dict):
            print(f"  {backend}: {data['mean']:.4f}s (min: {data['min']:.4f}s)")
        else:
            print(f"  {backend}: {data}")

    print("\nMatrix Multiplication (1000x1000):")
    results = benchmark_matrix_multiplication()
    for backend, data in results.items():
        if isinstance(data, dict):
            print(f"  {backend}: {data['mean']:.4f}s (min: {data['min']:.4f}s)")
        else:
            print(f"  {backend}: {data}")

    print("\nPerformance monitoring timings:")
    timings = np.get_timings()
    for func, total_time in timings.items():
        print(f"  {func}: {total_time:.6f}s total")