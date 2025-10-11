def emulate(version="2.x"):
    if version.startswith("1."):
        legacy_attrs["int"] = np.int64
        legacy_attrs["bool"] = np.bool_
    else:
        legacy_attrs.clear()
