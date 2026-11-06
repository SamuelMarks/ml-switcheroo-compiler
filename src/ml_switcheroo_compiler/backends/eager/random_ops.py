"""Generate random operations for eager backends."""

from __future__ import annotations

import typing  # noqa: F401

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, ensure_eager_list, ensure_eager_tensor, ensure_int_list, global_eager_registry
from ml_switcheroo_compiler.backends.protocols import get_backend_member, has_backend_member

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
    if has_backend_member(backend_module, "array"):
        return ensure_eager_tensor(get_backend_member(backend_module, "array")([0, seed], dtype="uint32"))
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
    if has_backend_member(backend_module, "array"):
        return ensure_eager_tensor(get_backend_member(backend_module, "array")(data_list, dtype="uint32"))
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
    has_idx = isinstance(key, (list, tuple)) or has_backend_member(key, "shape")
    val0 = int(ensure_int_list(key)[0]) if has_idx else 0
    val1 = int(ensure_int_list(key)[1]) if has_idx else 0
    data_list = [val0 + data, val1]
    if has_backend_member(backend_module, "array"):
        return ensure_eager_tensor(get_backend_member(backend_module, "array")(data_list, dtype="uint32"))
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
    shape = ensure_eager_list(kwargs.get("shape", args if args else ()) if has_backend_member(kwargs, "get") else ())
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = list(shape[0])
    if has_backend_member(backend_module, "random"):
        rand_mod = get_backend_member(backend_module, "random")
        if has_backend_member(rand_mod, "uniform"):
            return ensure_eager_tensor(rand_mod.uniform(size=shape))
        if has_backend_member(rand_mod, "rand"):
            return ensure_eager_tensor(rand_mod.rand(*shape))
    if has_backend_member(backend_module, "rand"):
        return ensure_eager_tensor(get_backend_member(backend_module, "rand")(*shape))
    if has_backend_member(backend_module, "uniform"):
        return ensure_eager_tensor(get_backend_member(backend_module, "uniform")(size=shape))

    return ensure_eager_tensor(get_backend_member(backend_module, "random").uniform(size=shape))


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
    shape = ensure_eager_list(kwargs.get("shape", args if args else ()) if has_backend_member(kwargs, "get") else ())
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = list(shape[0])
    if has_backend_member(backend_module, "random"):
        rand_mod = get_backend_member(backend_module, "random")
        if has_backend_member(rand_mod, "normal"):
            return ensure_eager_tensor(rand_mod.normal(size=shape))
        if has_backend_member(rand_mod, "randn"):
            return ensure_eager_tensor(rand_mod.randn(*shape))
    if has_backend_member(backend_module, "randn"):
        return ensure_eager_tensor(get_backend_member(backend_module, "randn")(*shape))
    if has_backend_member(backend_module, "normal"):
        return ensure_eager_tensor(get_backend_member(backend_module, "normal")(size=shape))

    return ensure_eager_tensor(get_backend_member(backend_module, "random").normal(size=shape))


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
    low = kwargs.get("low", args[0] if len(args) > 0 else 0) if has_backend_member(kwargs, "get") else 0
    high = kwargs.get("high", args[1] if len(args) > 1 else 10) if has_backend_member(kwargs, "get") else 10
    shape = kwargs.get("shape", args[2] if len(args) > 2 else ()) if has_backend_member(kwargs, "get") else ()
    if has_backend_member(backend_module, "random") and has_backend_member(get_backend_member(backend_module, "random"), "randint"):
        return ensure_eager_tensor(get_backend_member(backend_module, "random").randint(low, high, size=shape))

    return ensure_eager_tensor(get_backend_member(backend_module, "random").randint(low, high, size=shape))
