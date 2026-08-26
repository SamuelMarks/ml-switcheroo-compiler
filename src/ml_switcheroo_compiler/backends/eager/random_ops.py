# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Generate random operations for eager backends."""

from typing import Any

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("PRNGKey")
def prng_key(backend_module: Any, seed: int) -> Any:
    """Create a PRNGKey.

    Args:
        backend_module (Any): The backend_module parameter.
        seed (int): The seed parameter.

    Returns:
            Any: Result.
    """
    if hasattr(backend_module, "array"):
        return backend_module.array([0, seed], dtype="uint32")
    return [0, seed]


@global_eager_registry.register("RandomSplit")
def random_split(backend_module: Any, key: Any, num: int = 2) -> Any:
    """Split a PRNGKey.

    Args:
        backend_module (Any): The backend_module parameter.
        key (Any): The key parameter.
        num (int): The num parameter.

    Returns:
            Any: Result.
    """
    import random

    data_list = [[random.randint(0, 0xFFFFFFFF), random.randint(0, 0xFFFFFFFF)] for _ in range(num)]
    if hasattr(backend_module, "array"):
        return backend_module.array(data_list, dtype="uint32")
    return data_list


@global_eager_registry.register("RandomFoldIn")
def random_fold_in(backend_module: Any, key: Any, data: int) -> Any:
    """Fold in data to a PRNGKey.

    Args:
        backend_module (Any): The backend_module parameter.
        key (Any): The key parameter.
        data (int): The data parameter.

    Returns:
            Any: Result.
    """
    val0 = int(key[0]) if hasattr(key, "__getitem__") else 0
    val1 = int(key[1]) if hasattr(key, "__getitem__") else 0
    data_list = [val0 + data, val1]
    if hasattr(backend_module, "array"):
        return backend_module.array(data_list, dtype="uint32")
    return data_list


@global_eager_registry.register("Rand")
def rand(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Generate uniform random values.

    Args:
        backend_module (Any): The backend_module parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            Any: Result.
    """
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
def randn(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Generate normal random values.

    Args:
        backend_module (Any): The backend_module parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            Any: Result.
    """
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
def randint(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Generate integer random values.

    Args:
        backend_module (Any): The backend_module parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            Any: Result.
    """
    low = kwargs.get("low", args[0] if len(args) > 0 else 0)
    high = kwargs.get("high", args[1] if len(args) > 1 else 10)
    shape = kwargs.get("shape", args[2] if len(args) > 2 else ())
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "randint"):
        return backend_module.random.randint(low, high, size=shape)

    return backend_module.random.randint(low, high, size=shape)
