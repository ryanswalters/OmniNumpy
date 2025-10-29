

---

# **OmniNumPy (Unfinished Experiment)**

> *Stop fighting NumPy version hell.*
> OmniNumPy is an **attempt** at a universal compatibility layer — run legacy NumPy code on NumPy 2.x and even swap backends (Torch, CuPy, JAX) with zero refactoring.
> It mostly works. Sometimes it explodes. That’s research.

---

## ⚠️ Status: Prototype, not Production

* **Unfinished** — several backends half-wired; some ops crash outright.
* **Partial coverage** — about 20 functions dispatch correctly; thousands don’t.
* **Fragile** — error handling is explicit, but not graceful.
* **Performance** — benchmarks are interesting, not reliable.

If you can make it stable, please do. PRs welcome.

---

## 🚀 Why This Exists

Scientific computing shouldn’t be a walled garden. You should be able to:

* Write once, run on CPU or GPU.
* Keep ancient NumPy 1.x code alive under NumPy 2.x.
* Experiment with Torch, CuPy, or JAX without rewriting everything.

OmniNumPy proves that idea *mostly* works — for now.

---


---

## 📖 Usage (When It Does Work)

```python
import omninumpy as np

np.set_backend("torch")
a = np.array([1, 2, 3])
b = np.dot(a, a)
```

You can also emulate old NumPy behavior:

```python
np.emulate("1.21")
print(np.int)
```

---

## 🧠 Architecture Sketch

```
User Code → Compatibility Layer → Backend Abstraction → Version Emulator → NumPy ≥ 2.x
```

The idea: same API, different engines underneath.

---

## ✅ Currently Functional

| Function   | NumPy | Torch | CuPy |    JAX   |
| ---------- | :---: | :---: | :--: | :------: |
| array      |   ✅   |   ✅   |   ✅  |     ✅    |
| dot        |   ✅   |   ✅   |   ✅  | ⚠️ (JIT) |
| matmul     |   ✅   |   ✅   |   ✅  | ⚠️ (JIT) |
| mean, sum  |   ✅   |   ✅   |   ✅  |     ✅    |
| linalg.inv |   ✅   |   ✅   |   ✅  | ⚠️ (JIT) |
| linalg.svd |   ✅   |   ✅   |   ✅  | ⚠️ (JIT) |

Everything else → NumPy fallback → hope.

---

## 🗺️ Roadmap

* Wire up more functions across all backends
* Add isolation and smarter fallbacks
* Expand linear algebra suite
* Harden tests and CI

---

## 📜 License & Credits

MIT License.
Built on [NumPy (© NumPy Developers, BSD 3-Clause)](https://numpy.org/).
Torch, CuPy, and JAX belong to their respective developers.

---

**TL;DR:**
OmniNumPy doesn’t *work* reliably — yet. But the architecture’s there.
If you enjoy fighting the laws of tensor physics, fork it and evolve it.

---

