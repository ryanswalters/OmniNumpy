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
                if backend == "numpy":
                    a = np.arange(size, dtype=np.float32)
                elif backend == "torch":
                    a = np.arange(size, dtype=np.float32)
                elif backend == "cupy":
                    a = np.arange(size, dtype=np.float32)
                elif backend == "jax":
                    a = np.arange(size, dtype=np.float32)
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
            # Create matrices using backend-native random
            a, b = _create_random_matrices(backend, size)

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

def benchmark_linear_algebra(size=500, iterations=3):
    """Benchmark linear algebra operations"""
    results = {}

    for backend in ["numpy", "torch", "cupy", "jax"]:
        try:
            np.set_backend(backend)
            # Create matrix using backend-native random
            a, _ = _create_random_matrices(backend, size)
            # Make it positive definite for cholesky
            a = np.dot(a, a.T) + np.eye(size)
            b = _create_random_vector(backend, size)

            operations = ['inv', 'svd', 'eig', 'cholesky', 'qr', 'det', 'norm', 'solve']
            op_results = {}

            for op in operations:
                times = []
                for _ in range(iterations):
                    start = time.perf_counter()
                    if op == 'inv':
                        result = np.linalg.inv(a)
                    elif op == 'svd':
                        result = np.linalg.svd(a)
                    elif op == 'eig':
                        result = np.linalg.eig(a)
                    elif op == 'cholesky':
                        result = np.linalg.cholesky(a)
                    elif op == 'qr':
                        result = np.linalg.qr(a)
                    elif op == 'det':
                        result = np.linalg.det(a)
                    elif op == 'norm':
                        result = np.linalg.norm(a)
                    elif op == 'solve':
                        result = np.linalg.solve(a, b)
                    end = time.perf_counter()
                    times.append(end - start)

                op_results[op] = {
                    'mean': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times)
                }

            results[backend] = op_results
        except ImportError:
            results[backend] = "Not available"
        except Exception as e:
            results[backend] = f"Error: {e}"

    return results

def benchmark_mixed_workload(size=500, iterations=3):
    """Benchmark mixed computational workloads"""
    results = {}

    for backend in ["numpy", "torch", "cupy", "jax"]:
        try:
            np.set_backend(backend)
            times = []

            for _ in range(iterations):
                start = time.perf_counter()

                # Mixed workload: creation, arithmetic, reduction, linalg
                a, b = _create_random_matrices(backend, size)

                # Arithmetic operations
                c = np.dot(a, b)
                d = np.sum(c, axis=0)
                e = np.mean(d)

                # Linear algebra
                f = np.eye(size // 2)
                g = np.linalg.inv(f)

                # Array manipulation
                h = np.stack([a[:size//2], b[:size//2]])
                i = np.concatenate([a, b], axis=1)

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

def _create_random_matrices(backend, size):
    """Helper to create random matrices using backend-native methods"""
    if backend == "numpy":
        a = np.random.random((size, size)).astype(np.float32)
        b = np.random.random((size, size)).astype(np.float32)
    elif backend == "torch":
        try:
            torch = np._get_torch()
            a = torch.randn(size, size, dtype=torch.float32)
            b = torch.randn(size, size, dtype=torch.float32)
        except:
            a = np.random.random((size, size)).astype(np.float32)
            b = np.random.random((size, size)).astype(np.float32)
    elif backend == "cupy":
        try:
            cp = np._get_cupy()
            a = cp.random.random((size, size), dtype=cp.float32)
            b = cp.random.random((size, size), dtype=cp.float32)
        except:
            a = np.random.random((size, size)).astype(np.float32)
            b = np.random.random((size, size)).astype(np.float32)
    elif backend == "jax":
        try:
            jr = np._get_jax_random()
            key = jr.PRNGKey(42)
            key1, key2 = jr.split(key)
            a = jr.normal(key1, (size, size), dtype=np._get_jax_numpy().float32)
            b = jr.normal(key2, (size, size), dtype=np._get_jax_numpy().float32)
        except:
            a = np.random.random((size, size)).astype(np.float32)
            b = np.random.random((size, size)).astype(np.float32)
    else:
        a = np.random.random((size, size)).astype(np.float32)
        b = np.random.random((size, size)).astype(np.float32)

    return a, b

def _create_random_vector(backend, size):
    """Helper to create random vectors using backend-native methods"""
    if backend == "numpy":
        return np.random.random(size).astype(np.float32)
    elif backend == "torch":
        try:
            torch = np._get_torch()
            return torch.randn(size, dtype=torch.float32)
        except:
            return np.random.random(size).astype(np.float32)
    elif backend == "cupy":
        try:
            cp = np._get_cupy()
            return cp.random.random(size, dtype=cp.float32)
        except:
            return np.random.random(size).astype(np.float32)
    elif backend == "jax":
        try:
            jr = np._get_jax_random()
            key = jr.PRNGKey(42)
            return jr.normal(key, (size,), dtype=np._get_jax_numpy().float32)
        except:
            return np.random.random(size).astype(np.float32)
    else:
        return np.random.random(size).astype(np.float32)

if __name__ == "__main__":
    print("Omninumpy Comprehensive Benchmark Suite")
    print("=" * 50)

    print("\n1. Array Creation (1M elements):")
    results = benchmark_array_creation()
    for backend, data in results.items():
        if isinstance(data, dict):
            print(f"  {backend}: {data['mean']:.4f}s (min: {data['min']:.4f}s)")
        else:
            print(f"  {backend}: {data}")

    print("\n2. Matrix Multiplication (1000x1000):")
    results = benchmark_matrix_multiplication()
    for backend, data in results.items():
        if isinstance(data, dict):
            print(f"  {backend}: {data['mean']:.4f}s (min: {data['min']:.4f}s)")
        else:
            print(f"  {backend}: {data}")

    print("\n3. Linear Algebra Operations (500x500):")
    results = benchmark_linear_algebra()
    for backend, ops in results.items():
        if isinstance(ops, dict):
            print(f"  {backend}:")
            for op, data in ops.items():
                if isinstance(data, dict):
                    print(f"    {op}: {data['mean']:.4f}s")
                else:
                    print(f"    {op}: {data}")
        else:
            print(f"  {backend}: {ops}")

    print("\n4. Mixed Workload (500x500):")
    results = benchmark_mixed_workload()
    for backend, data in results.items():
        if isinstance(data, dict):
            print(f"  {backend}: {data['mean']:.4f}s (min: {data['min']:.4f}s)")
        else:
            print(f"  {backend}: {data}")

    print("\n5. Performance monitoring timings:")
    timings = np.get_timings()
    for func, total_time in timings.items():
        print(f"  {func}: {total_time:.6f}s total")