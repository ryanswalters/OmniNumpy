def array(object, dtype=None, *, copy=None, order='K', subok=False, ndmin=0, like=None):
    backend = get_backend_module()
    # handle kwargs compatibly
    return backend.array(object, **kwargs)

def zeros(shape, dtype=float, order='C', *, like=None):
    return get_backend_module().zeros(shape, dtype=dtype)

def ones(shape, dtype=None, order='C', *, like=None):
    return get_backend_module().ones(shape, dtype=dtype)
