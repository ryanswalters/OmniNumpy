def sum(a, axis=None, dtype=None, keepdims=False):
    return get_backend_module().sum(a, axis=axis, dtype=dtype, keepdims=keepdims)

def mean(a, axis=None, dtype=None, keepdims=False):
    return get_backend_module().mean(a, axis=axis, dtype=dtype, keepdims=keepdims)
