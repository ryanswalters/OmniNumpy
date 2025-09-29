# OmniNumpy

Stop fighting NumPy version hell. OmniNumpy is the compatibility layer that lets legacy code run on NumPy 2.x while unlocking GPU acceleration with zero refactoring. Drop-in replacement. Backend agnostic. Just works.

## Installation

```bash
pip install omninumpy
```

## Usage

Simply replace your numpy import:

```python
# Before
import numpy as np

# After
import omninumpy as np
```

That's it! Your existing code will work with NumPy 2.x and automatically gain GPU acceleration when available.

## Features

- **Zero Refactoring**: Drop-in replacement for NumPy
- **NumPy 2.x Compatibility**: Handles breaking changes automatically
- **GPU Acceleration**: Automatically uses CuPy, JAX, or other backends when available
- **Backend Agnostic**: Works with NumPy, CuPy, JAX, and more
- **Backwards Compatible**: Supports legacy NumPy 1.x APIs

## Supported Backends

- NumPy (CPU)
- CuPy (NVIDIA GPUs)
- JAX (CPU/GPU/TPU)
- More backends coming soon...
