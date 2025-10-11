def dot(a, b):
    return get_backend_module().dot(a, b)

def matmul(a, b):
    backend = get_backend_module()
    return backend.matmul(a, b) if hasattr(backend, "matmul") else backend.dot(a, b)
