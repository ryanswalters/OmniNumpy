def random(shape):
    return get_backend_module().random.random(shape)

def randn(shape):
    return get_backend_module().random.randn(shape)
