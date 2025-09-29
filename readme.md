🎯 Goal

A single NumPy wrapper that:

Runs on top of the latest stable NumPy (≥2.0).

Provides backward compatibility for older NumPy APIs (1.x, 1.21, etc.).

Optionally swaps backends (CuPy, Torch, JAX) without code changes.

Minimizes breakage in AI/ML libraries that hard-pin weird NumPy versions.

🛠️ Core Components
1. Base Layer (NumPy ≥2.x)

Import the newest NumPy internally:

import numpy as _np


Expose everything by default:

from numpy import *


This ensures modern code works natively.

2. Legacy Compatibility Layer

A shim that re-creates removed aliases and functions:

Aliases:

_np.int = int
_np.float = float
_np.bool = bool


Functions:

def asscalar(a): return a.item()
def matrix(data, *args, **kwargs): return _np.array(data, *args, **kwargs)


Catch-all for missing attributes:

class OmniNP:
    def __getattr__(self, name):
        if name in ("int", "float", "bool"):
            return getattr(_np, "int64")
        if name == "asscalar":
            return lambda a: a.item()
        raise AttributeError(name)
np = OmniNP()

3. Backend Abstraction Layer

A toggle system:

BACKEND = "numpy"

def set_backend(name):
    global BACKEND
    if name in ("numpy", "torch", "cupy", "jax"):
        BACKEND = name
    else:
        raise ValueError("Unsupported backend")


Wrapper functions:

def array(x, *args, **kwargs):
    if BACKEND == "torch":
        import torch
        return torch.tensor(x, *args, **kwargs)
    if BACKEND == "cupy":
        import cupy
        return cupy.array(x, *args, **kwargs)
    return _np.array(x, *args, **kwargs)

4. Version Emulation Profiles

Profiles like "1.21", "1.19", "2.x".

Each profile enables/patches features:

def emulate(version="1.21"):
    if version.startswith("1."):
        _np.int = int
        _np.float = float
        _np.bool = bool
        _np.asscalar = lambda a: a.item()

5. Testing Matrix

Run CI against:

NumPy 1.19, 1.21, 2.x

Libraries: TensorFlow, PyTorch, Pandas

Verify import omninumpy as np doesn’t break.

6. Optional Extensions

Auto-detect when a library is breaking and enable the right profile.

Config file (~/.omninumpy.json) for defaults.

Logging/warnings when old APIs are used.

🚀 Steps to Build

Scaffold repo → omninumpy/ with __init__.py.

Base import → wrap NumPy ≥2.x.

Legacy shim → implement aliases + removed functions.

Backend toggle → add set_backend().

Profiles → add emulate("1.21").

Testing → scripts that run old + new libraries.

Release → package on PyPI as omninumpy.


Here’s a **visual-style architecture overview** for **Omninumpy (or Omninp)** to show how we’d structure it:

---

# 🌐 Omninumpy High-Level Architecture

```
 ┌───────────────────────────────────────────────┐
 │                 User Code                     │
 │   import omninumpy as np                      │
 │                                               │
 │   # Works across old + new versions           │
 │   np.int, np.asscalar, np.array([...])        │
 └───────────────────────────────────────────────┘
                         │
                         ▼
 ┌───────────────────────────────────────────────┐
 │         Omninumpy Compatibility Layer         │
 │                                               │
 │  - Re-add deprecated APIs                     │
 │    (np.int, np.float, np.bool, asscalar)      │
 │  - Shims for 1.x → 2.x changes                │
 │  - Error catching & redirecting               │
 └───────────────────────────────────────────────┘
                         │
                         ▼
 ┌───────────────────────────────────────────────┐
 │          Backend Abstraction Layer            │
 │                                               │
 │  set_backend("numpy" | "cupy" | "torch" | "jax")  │
 │                                               │
 │  array(), dot(), matmul() wrappers            │
 │  Dispatches to correct backend implementation │
 └───────────────────────────────────────────────┘
                         │
                         ▼
 ┌───────────────────────────────────────────────┐
 │           Version Emulation Profiles          │
 │                                               │
 │  emulate("1.19") → enable legacy functions    │
 │  emulate("1.21") → align dtype behaviors      │
 │  emulate("2.x")  → strict new NumPy rules     │
 │                                               │
 │  Config-driven: ~/.omninp.json or ENV vars    │
 └───────────────────────────────────────────────┘
                         │
                         ▼
 ┌───────────────────────────────────────────────┐
 │            Base NumPy ≥ 2.x Core              │
 │                                               │
 │  Always imports latest NumPy (≥2.x)           │
 │  Uses as foundation for universal API         │
 └───────────────────────────────────────────────┘
```

---

## 🔑 Key Ideas

1. **Top-down flow**:
   User calls `omninumpy` → hits **compat layer** → passes through backend toggle → applies profile rules → executes on NumPy ≥2.x (or another backend).

2. **Legacy support**:

   * Shims for **1.x removed functions/aliases**.
   * Profiles simulate older behaviors (like `matrix`, `asscalar`, or dtype coercions).

3. **Future-proofing**:

   * Base always tracks **latest NumPy release**.
   * Profiles + shims handle breakage automatically.

4. **Extendable**:

   * Can plug in JAX, Torch, CuPy just by implementing the same wrapper API.

---

⚡ This way, you end version hell:

* Old libraries written for NumPy 1.19 → still work.
* New ML frameworks requiring NumPy 2.x → also work.
* Users can flip backends (CPU/GPU) without changing code.

---
Do you want me to also sketch out **the first-pass API design** (what functions we implement in v1)? That way we can start small and grow.

Mapped out a full API design: import stays dead simple, legacy shims live behind `emulate()`, backend switching works with `set_backend()`, and we can optionally wire in WebSockets for remote execution if you want to get spicy.


# 🔌 Omninumpy API Design

Here’s a breakdown of how the API should look and behave, including whether WebSockets make sense.

---

## 🎯 Core Import Behavior

Users always import the same way:

```python
import omninumpy as np
```

That’s it. Omninumpy decides under the hood whether it needs to shim old APIs, switch backends, or run in strict NumPy ≥2.x mode.

---

## 🛠️ API Layers

### 1. Base NumPy API

* Exposes **all current NumPy functions** directly.
* Behaves like modern NumPy (≥2.0).

Example:

```python
np.array([1, 2, 3])
np.dot(a, b)
```

---

### 2. Legacy Compatibility

* Provides shims for **removed/renamed functions**.
* Behaves like older NumPy when `emulate("1.x")` is called.

Example:

```python
np.int, np.float, np.bool    # Restored
np.asscalar(x)               # Works again
np.matrix([[1,2],[3,4]])     # Wraps np.array()
```

---

### 3. Backend Abstraction

Users can change computation backend with a single call:

```python
np.set_backend("torch")
np.set_backend("cupy")
np.set_backend("jax")
```

Example:

```python
x = np.array([1, 2, 3])   # torch.tensor if backend=torch
```

Wrapper functions mirror NumPy's API for implemented functions:

```python
np.array, np.dot, np.matmul, np.linalg.inv  # These are backend-aware
# Other functions fall back to NumPy (see limitations below)
```

---

### 4. Version Emulation

Switch profiles to restore old semantics:

```python
np.emulate("1.19")
np.emulate("1.21")
np.emulate("2.x")
```

This changes:

* dtype aliasing
* deprecated function support
* matrix/asscalar availability

---

## ⚙️ Configuration

* Defaults can be stored in `~/.omninumpy.json`:

```json
{
  "backend": "jax",
  "device": "gpu",
  "jit": true,
  "jit_threshold": 1000,
  "profile": "1.21",
  "auto_select_backend": false
}
```

* Environment variables override config:

```bash
OMNINP_BACKEND=jax:gpu
OMNINP_JIT=true
OMNINP_JIT_THRESHOLD=1000
OMNINP_PROFILE=1.19
OMNINP_AUTO_BACKEND=true
```

## 🚀 Installation

```bash
pip install omninumpy
```

Optional backends:
```bash
pip install omninumpy[torch]  # For PyTorch backend
pip install omninumpy[cupy]  # For CuPy backend
pip install omninumpy[jax]   # For JAX backend
```

## 📖 Usage

### Basic Usage

```python
import omninumpy as np

# Works like regular NumPy
a = np.array([1, 2, 3])
b = np.dot(a, a)
```

### Legacy Compatibility

```python
import omninumpy as np

# Enable legacy APIs
np.emulate("1.21")

# Now old APIs work
print(np.int)  # <class 'numpy.int64'>
scalar = np.asscalar(np.array([42]))  # 42
```

### Backend Switching

```python
import omninumpy as np

# Switch to PyTorch
np.set_backend("torch")
a = np.array([1, 2, 3])  # Creates torch.Tensor

# Switch to CuPy
np.set_backend("cupy")
b = np.array([4, 5, 6])  # Creates cupy.ndarray

# Advanced JAX with device and JIT
np.set_backend("jax:gpu")  # GPU acceleration
c = np.array([7, 8, 9])    # JAX array on GPU
d = np.dot(c, c)           # JIT compiled for speed
```

### JAX Advanced Features

```python
import omninumpy as np

# Device-aware backends
np.set_backend("jax:cpu")   # CPU only
np.set_backend("jax:gpu")   # GPU acceleration
np.set_backend("jax:tpu")   # TPU acceleration

# JIT compilation with smart thresholds
# Large arrays automatically use JIT for performance
a = np.random.random((2000, 2000))  # Large array → JIT compiled
b = np.random.random((100, 100))    # Small array → eager execution
c = np.dot(a, b)  # Fast JIT execution

# Interoperability
numpy_array = np.to_numpy(jax_array)    # JAX → NumPy
jax_array = np.to_backend(numpy_array)  # NumPy → JAX (on current device)
```

### Auto Backend Selection

```python
import omninumpy as np

# Automatically pick fastest available backend
backend = np.auto_backend()
np.set_backend(backend)
```

Or via config:

```json
{
  "auto_select_backend": true
}
```

### Performance Monitoring

```python
import omninumpy as np

# Use functions
a = np.array([1, 2, 3])

# Get timing data
timings = np.get_timings()
print(timings)  # {'_array': 0.000123}
```

### Validation

```python
import omninumpy as np

a = np.array([1, 2, 3])
np.validate_array(a, shape=(3,), dtype=np.int64)
```

## 🔄 Migration Guide

### From NumPy 1.x

If your code uses deprecated NumPy 1.x APIs:

1. Install omninumpy
2. Change `import numpy as np` to `import omninumpy as np`
3. Add `np.emulate("1.21")` at the top of your script
4. Your code should work unchanged

### Backend Migration

To use GPU acceleration without code changes:

1. Install backend: `pip install omninumpy[torch]`
2. Add `np.set_backend("torch")` before computations
3. Arrays automatically become torch tensors

### Configuration

Create `~/.omninumpy.json`:

```json
{
  "backend": "torch",
  "profile": "1.21",
  "auto_select_backend": false
}
```

This sets defaults for all imports.

---

## 📡 WebSocket Integration (Optional)

### Why?

* Could allow **remote backend execution** (e.g. run arrays on a GPU server while client runs lightweight code).
* Lets multiple processes share one backend session.

### How?

* Provide `set_backend("ws://server:port")`.
* Under the hood, Omninumpy serializes arrays (using pickle or Arrow), sends over WebSocket.
* Remote worker (server) executes NumPy/Torch/CuPy and returns result.

Example:

```python
np.set_backend("ws://localhost:9000")
np.array([1, 2, 3])   # Executed remotely
```

### API Impact

* Transparent to user.
* Extra dependency: `websockets` or `socket.io`.
* Useful for distributed/remote compute scenarios.

---

## 🚀 API Summary

* **Core:** Works like modern NumPy.
* **Compat:** Old APIs restored if needed.
* **Backend toggle:** Local CPU/GPU or Torch/JAX.
* **Profiles:** Emulate specific versions.
* **Optional WebSocket:** Offload execution to remote workers.

This keeps the surface familiar (NumPy-like) while hiding all the messy backend switching and compatibility logic.

## ✅ Implemented Features

### Backend-Aware Functions
| Function | NumPy | PyTorch | CuPy | JAX |
|----------|-------|---------|------|-----|
| `array` | ✅ | ✅ | ✅ | ✅ |
| `dot` | ✅ | ✅ | ✅ | ✅ (JIT) |
| `matmul` | ✅ | ✅ | ✅ | ✅ (JIT) |
| `mean` | ✅ | ✅ | ✅ | ✅ (JIT) |
| `sum` | ✅ | ✅ | ✅ | ✅ (JIT) |
| `stack` | ✅ | ✅ | ✅ | ✅ |
| `concatenate` | ✅ | ✅ | ✅ | ✅ |
| `eye` | ✅ | ✅ | ✅ | ✅ |
| `zeros` | ✅ | ✅ | ✅ | ✅ |
| `ones` | ✅ | ✅ | ✅ | ✅ |
| `reshape` | ✅ | ✅ | ✅ | ✅ (JIT) |
| `transpose` | ✅ | ✅ | ✅ | ✅ (JIT) |
| `astype` | ✅ | ✅ | ✅ | ✅ |
| `clip` | ✅ | ✅ | ✅ | ✅ (JIT) |
| `where` | ✅ | ✅ | ✅ | ✅ |

### Linear Algebra Functions
| Function | NumPy | PyTorch | CuPy | JAX |
|----------|-------|---------|------|-----|
| `linalg.inv` | ✅ | ✅ | ✅ | ✅ (JIT) |
| `linalg.svd` | ✅ | ✅ | ✅ | ✅ (JIT) |
| `linalg.eig` | ✅ | ✅ | ✅ | ✅ (JIT) |
| `linalg.cholesky` | ✅ | ✅ | ✅ | ✅ (JIT) |
| `linalg.qr` | ✅ | ✅ | ✅ | ✅ (JIT) |

## ⚠️ Current Limitations

- **Partial Coverage**: ~16 core functions are backend-aware. All others fall back to NumPy or raise `AttributeError` in non-NumPy backends.
- **JAX JIT**: Only applied to performance-critical functions (matrix ops, reductions). Other operations use eager execution.
- **Device Support**: JAX supports CPU/GPU/TPU placement. PyTorch/CuPy use their native device management.
- **Error Handling**: Unimplemented functions raise clear errors instead of silent NumPy fallback.
- **Testing**: Comprehensive backend-specific tests ensure type correctness and numerical accuracy.
- **Benchmarks**: Use backend-native random generation for fair performance comparison.

## 🚧 Roadmap

- Complete backend wrapping for all common NumPy functions
- Add strict backend isolation with error raising
- Expand linear algebra coverage
- Add comprehensive cross-backend tests
- Optimize backend switching performance
