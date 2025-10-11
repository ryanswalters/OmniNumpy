def reshape(a, newshape):
    return get_backend_module().reshape(a, newshape)

def transpose(a, axes=None):
    return get_backend_module().transpose(a, axes)
