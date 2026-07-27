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
    shape = kwargs.get("shape", args[0] if args else ())
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "rand"):
        return backend_module.random.rand(*shape)
    elif hasattr(backend_module, "rand"):
        return backend_module.rand(*shape)
    import numpy as np

    return np.random.rand(*shape)


@global_eager_registry.register("Randn")
def randn(backend_module: object, *args: object, **kwargs: object) -> object:
    """Generate normal random values."""
    shape = kwargs.get("shape", args[0] if args else ())
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "randn"):
        return backend_module.random.randn(*shape)
    elif hasattr(backend_module, "randn"):
        return backend_module.randn(*shape)
    import numpy as np

    return np.random.randn(*shape)


@global_eager_registry.register("Randint")
def randint(backend_module: object, *args: object, **kwargs: object) -> object:
    """Generate integer random values."""
    low = kwargs.get("low", args[0] if len(args) > 0 else 0)
    high = kwargs.get("high", args[1] if len(args) > 1 else 10)
    shape = kwargs.get("shape", args[2] if len(args) > 2 else ())
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "randint"):
        return backend_module.random.randint(low, high, size=shape)
    import numpy as np

    return np.random.randint(low, high, size=shape)
