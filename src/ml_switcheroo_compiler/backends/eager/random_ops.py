"""Random operations for eager backends."""

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("PRNGKey")
def prng_key(backend_module: object, seed: int) -> object:
    """Create a PRNGKey."""
    if hasattr(backend_module, "array"):
        return backend_module.array([0, seed], dtype="uint32")
    return [0, seed]


@global_eager_registry.register("RandomSplit")
def random_split(backend_module: object, key: object, num: int = 2) -> object:
    """Split a PRNGKey."""
    import random

    data_list = [[random.randint(0, 0xFFFFFFFF), random.randint(0, 0xFFFFFFFF)] for _ in range(num)]
    if hasattr(backend_module, "array"):
        return backend_module.array(data_list, dtype="uint32")
    return data_list


@global_eager_registry.register("RandomFoldIn")
def random_fold_in(backend_module: object, key: object, data: int) -> object:
    """Fold in data to a PRNGKey."""
    val0 = int(key[0]) if hasattr(key, "__getitem__") else 0
    val1 = int(key[1]) if hasattr(key, "__getitem__") else 0
    data_list = [val0 + data, val1]
    if hasattr(backend_module, "array"):
        return backend_module.array(data_list, dtype="uint32")
    return data_list


@global_eager_registry.register("Rand")
def rand(backend_module: object, *args: object, **kwargs: object) -> object:
    """Generate uniform random values."""
    shape = kwargs.get("shape", args if args else ())
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    if hasattr(backend_module, "random"):
        rand_mod = backend_module.random
        if hasattr(rand_mod, "uniform"):
            return rand_mod.uniform(size=shape)
        if hasattr(rand_mod, "rand"):
            return rand_mod.rand(*shape)
    if hasattr(backend_module, "rand"):
        return backend_module.rand(*shape)
    if hasattr(backend_module, "uniform"):
        return backend_module.uniform(size=shape)

    return backend_module.random.uniform(size=shape)


@global_eager_registry.register("Randn")
def randn(backend_module: object, *args: object, **kwargs: object) -> object:
    """Generate normal random values."""
    shape = kwargs.get("shape", args if args else ())
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    if hasattr(backend_module, "random"):
        rand_mod = backend_module.random
        if hasattr(rand_mod, "normal"):
            return rand_mod.normal(size=shape)
        if hasattr(rand_mod, "randn"):
            return rand_mod.randn(*shape)
    if hasattr(backend_module, "randn"):
        return backend_module.randn(*shape)
    if hasattr(backend_module, "normal"):
        return backend_module.normal(size=shape)

    return backend_module.random.normal(size=shape)


@global_eager_registry.register("Randint")
def randint(backend_module: object, *args: object, **kwargs: object) -> object:
    """Generate integer random values."""
    low = kwargs.get("low", args[0] if len(args) > 0 else 0)
    high = kwargs.get("high", args[1] if len(args) > 1 else 10)
    shape = kwargs.get("shape", args[2] if len(args) > 2 else ())
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "randint"):
        return backend_module.random.randint(low, high, size=shape)

    return backend_module.random.randint(low, high, size=shape)
