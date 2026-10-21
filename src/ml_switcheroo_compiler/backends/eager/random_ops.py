"""Generate random operations for eager backends."""

from __future__ import annotations

import typing  # noqa: F401

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, as_eager_list, as_int_list, as_tensor, global_eager_registry

RandVal = EagerValue


@global_eager_registry.register("PRNGKey")
def prng_key(backend_module: EagerValue, seed: int) -> RandVal:
    """Create a PRNGKey.

    Args:
        backend_module: The backend_module parameter.
        seed (int): The seed parameter.

    Returns:
            RandVal: Result.
    """
    if hasattr(backend_module, "array"):
        return as_tensor(getattr(backend_module, "array")([0, seed], dtype="uint32"))
    return [0, seed]


@global_eager_registry.register("RandomSplit")
def random_split(backend_module: EagerValue, key: EagerValue, num: int = 2) -> RandVal:
    """Split a PRNGKey.

    Args:
        backend_module: The backend_module parameter.
        key: The key parameter.
        num (int): The num parameter.

    Returns:
            RandVal: Result.
    """
    import random

    data_list = [[random.randint(0, 0xFFFFFFFF), random.randint(0, 0xFFFFFFFF)] for _ in range(num)]
    if hasattr(backend_module, "array"):
        return as_tensor(getattr(backend_module, "array")(data_list, dtype="uint32"))
    return data_list


@global_eager_registry.register("RandomFoldIn")
def random_fold_in(backend_module: EagerValue, key: EagerValue, data: int) -> RandVal:
    """Fold in data to a PRNGKey.

    Args:
        backend_module: The backend_module parameter.
        key: The key parameter.
        data (int): The data parameter.

    Returns:
            RandVal: Result.
    """
    val0 = int(as_int_list(key)[0]) if hasattr(key, "__getitem__") else 0
    val1 = int(as_int_list(key)[1]) if hasattr(key, "__getitem__") else 0
    data_list = [val0 + data, val1]
    if hasattr(backend_module, "array"):
        return as_tensor(getattr(backend_module, "array")(data_list, dtype="uint32"))
    return data_list


@global_eager_registry.register("Rand")
def rand(backend_module: EagerValue, *args: RandVal, **kwargs: RandVal) -> RandVal:
    """Generate uniform random values.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            RandVal: Result.
    """
    shape = as_eager_list(kwargs.get("shape", args if args else ()) if hasattr(kwargs, "get") else ())
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = list(shape[0])
    if hasattr(backend_module, "random"):
        rand_mod = getattr(backend_module, "random")
        if hasattr(rand_mod, "uniform"):
            return as_tensor(rand_mod.uniform(size=shape))
        if hasattr(rand_mod, "rand"):
            return as_tensor(rand_mod.rand(*shape))
    if hasattr(backend_module, "rand"):
        return as_tensor(getattr(backend_module, "rand")(*shape))
    if hasattr(backend_module, "uniform"):
        return as_tensor(getattr(backend_module, "uniform")(size=shape))

    return as_tensor(getattr(backend_module, "random").uniform(size=shape))


@global_eager_registry.register("Randn")
def randn(backend_module: EagerValue, *args: RandVal, **kwargs: RandVal) -> RandVal:
    """Generate normal random values.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            RandVal: Result.
    """
    shape = as_eager_list(kwargs.get("shape", args if args else ()) if hasattr(kwargs, "get") else ())
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = list(shape[0])
    if hasattr(backend_module, "random"):
        rand_mod = getattr(backend_module, "random")
        if hasattr(rand_mod, "normal"):
            return as_tensor(rand_mod.normal(size=shape))
        if hasattr(rand_mod, "randn"):
            return as_tensor(rand_mod.randn(*shape))
    if hasattr(backend_module, "randn"):
        return as_tensor(getattr(backend_module, "randn")(*shape))
    if hasattr(backend_module, "normal"):
        return as_tensor(getattr(backend_module, "normal")(size=shape))

    return as_tensor(getattr(backend_module, "random").normal(size=shape))


@global_eager_registry.register("Randint")
def randint(backend_module: EagerValue, *args: RandVal, **kwargs: RandVal) -> RandVal:
    """Generate integer random values.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            RandVal: Result.
    """
    low = kwargs.get("low", args[0] if len(args) > 0 else 0) if hasattr(kwargs, "get") else 0
    high = kwargs.get("high", args[1] if len(args) > 1 else 10) if hasattr(kwargs, "get") else 10
    shape = kwargs.get("shape", args[2] if len(args) > 2 else ()) if hasattr(kwargs, "get") else ()
    if hasattr(backend_module, "random") and hasattr(getattr(backend_module, "random"), "randint"):
        return as_tensor(getattr(backend_module, "random").randint(low, high, size=shape))

    return as_tensor(getattr(backend_module, "random").randint(low, high, size=shape))
