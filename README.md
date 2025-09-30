

# OmniNumPy (Experimental)

> Stop fighting NumPy version hell. OmniNumPy is a compatibility layer that lets legacy code run on NumPy 2.x while unlocking GPU acceleration with zero refactoring. Drop-in replacement. Backend agnostic. Just works.

---

## 📜 Credits

This project uses [NumPy](https://numpy.org/), licensed under the BSD 3-Clause License.  
NumPy © 2005–2025 NumPy Developers. See [LICENSE.txt](https://github.com/numpy/numpy/blob/main/LICENSE.txt) for details.

---

## ⚠️ Status

- **Experimental**: APIs, wrappers, and behaviors will change often.  
- **Partial coverage**: Only ~20 functions are backend-aware today. Thousands more are untouched.  
- **Testing**: Cross-backend correctness checks exist but aren’t exhaustive.  
- **Performance**: Benchmarks highlight real speedups, but not every op is optimized.  

---

## 🚀 Why This Exists

Scientific computing shouldn’t force you into one backend forever. You should be able to:

- Write NumPy-style code.  
- Switch to GPU with Torch or CuPy.  
- Explore JAX with JIT, TPU, and auto-device placement.  
- Keep old libraries alive by restoring missing APIs.  

OmniNumPy proves this vision works — even if it’s only partial today.

---

## 📦 Installation

```bash
pip install omninumpy


Optional backends:

pip install omninumpy[torch]   # For PyTorch backend
pip install omninumpy[cupy]    # For CuPy backend
pip install omninumpy[jax]     # For JAX backend

📖 Usage
Basic
import omninumpy as np

a = np.array([1, 2, 3])
b = np.dot(a, a)

Legacy APIs
import omninumpy as np
np.emulate("1.21")

print(np.int)  
scalar = np.asscalar(np.array([42]))

Backend Switching
np.set_backend("torch")
a = np.array([1, 2, 3])   # torch.Tensor

np.set_backend("cupy")
b = np.array([4, 5, 6])   # cupy.ndarray

np.set_backend("jax:gpu")
c = np.array([7, 8, 9])   # JAX array on GPU

🎯 Goals

Run on top of the latest stable NumPy (≥2.0).

Provide backward compatibility for older NumPy APIs (1.x, 1.21, etc.).

Allow backend swaps (CuPy, Torch, JAX) with zero refactoring.

Minimize breakage in AI/ML libraries pinned to outdated NumPy.

🛠️ Core Components

Base Layer (NumPy ≥2.x) – Import and expose modern NumPy.

Legacy Compatibility Layer – Restore np.int, np.float, np.asscalar, etc.

Backend Abstraction Layer – set_backend("numpy" | "torch" | "cupy" | "jax").

Version Emulation Profiles – emulate("1.19"), emulate("1.21"), etc.

Testing Matrix – CI with NumPy 1.19 → 2.x, Torch, Pandas.

Optional Extensions – Auto-detect breaking libs, config file, warnings.

🌐 High-Level Architecture
 ┌───────────────────────────┐
 │        User Code          │
 │   import omninumpy as np  │
 └─────────────┬─────────────┘
               ▼
 ┌───────────────────────────┐
 │  Compatibility Layer       │
 │  (shims for old APIs)      │
 └─────────────┬─────────────┘
               ▼
 ┌───────────────────────────┐
 │ Backend Abstraction Layer │
 │ set_backend("numpy"/...)  │
 └─────────────┬─────────────┘
               ▼
 ┌───────────────────────────┐
 │ Version Emulation Profiles│
 │ emulate("1.19"/"2.x")     │
 └─────────────┬─────────────┘
               ▼
 ┌───────────────────────────┐
 │     Base NumPy ≥ 2.x      │
 └───────────────────────────┘

✅ Implemented Functions (Preview)
Function	NumPy	Torch	CuPy	JAX
array	✅	✅	✅	✅
dot	✅	✅	✅	✅ (JIT)
matmul	✅	✅	✅	✅ (JIT)
mean, sum	✅	✅	✅	✅
linalg.inv	✅	✅	✅	✅ (JIT)
linalg.svd	✅	✅	✅	✅ (JIT)
…	…	…	…	…
⚠️ Limitations

Only ~20 core functions backend-aware today.

Most others fall back to NumPy.

JAX JIT applied only to critical ops.

Error handling = explicit (no silent fallback).

## 🔧 Troubleshooting (a.k.a. Install for Humans)

1. Smash the big blue **Code** button at the top.  
2. Download ZIP.  
3. Unzip like it’s 2007.  
4. Open VS Code.  
5. Install Roo or Kilo extension.  
6. Type your incantation, profit.  


🗺️ Roadmap

Wrap more functions across all backends.

Add strict backend isolation.

Expand linear algebra coverage.

Improve cross-backend tests + benchmarks.

📜 License

MIT — take any piece, fork it, or bolt it into your own project.

