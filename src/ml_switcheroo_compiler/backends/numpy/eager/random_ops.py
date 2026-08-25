# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Numpy eager fallback implementations for random operations."""

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("Normal")
def _np_normal(backend_module: object, shape: object, **kwargs: object) -> object:
    """Evaluate _np_normal operation.

    Args:
        backend_module (object): The backend_module parameter.
        shape (object): The shape parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    dtype: object = kwargs.get("dtype", "float32")
    config: object = kwargs.get("config")
    mean: object = getattr(config, "mean", 0.0) if config else 0.0
    stddev: object = getattr(config, "stddev", 1.0) if config else 1.0
    return np.random.normal(loc=mean, scale=stddev, size=shape).astype(dtype)


@numpy_eager_registry.register("Uniform")
def _np_uniform(backend_module: object, shape: object, **kwargs: object) -> object:
    """Evaluate _np_uniform operation.

    Args:
        backend_module (object): The backend_module parameter.
        shape (object): The shape parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    dtype: object = kwargs.get("dtype", "float32")
    config: object = kwargs.get("config")
    minval: object = getattr(config, "minval", 0.0) if config else 0.0
    if minval is None:
        minval: object = 0.0
    maxval: object = getattr(config, "maxval", 1.0) if config else 1.0
    if maxval is None:
        maxval: object = 1.0
    return np.random.uniform(low=minval, high=maxval, size=shape).astype(dtype)


@numpy_eager_registry.register("StatelessSplit")
def _np_stateless_split(backend_module: object, seed: object, **kwargs: object) -> object:
    """Evaluate _np_stateless_split operation.

    Args:
        backend_module (object): The backend_module parameter.
        seed (object): The seed parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    num: object = kwargs.get("num", 2)
    s: object = np.asarray(seed).flatten()
    if len(s) > 0:
        if isinstance(s[0], (str, np.str_)):
            s_val: object = hash(s[0]) % (2**31 - 1)
        else:
            try:
                s_val: object = int(s[0])
            except ValueError:
                s_val: object = hash(str(s[0])) % (2**31 - 1)
    else:
        s_val: object = 0
    rng: object = np.random.RandomState(s_val)
    # Generate positive seeds to avoid overflow issues depending on platform
    return rng.randint(0, 2**31 - 1, size=(num, 2), dtype=np.int64)


@numpy_eager_registry.register("Lookup")
def _np_lookup(backend_module: object, table: object, keys: object, **kwargs: object) -> object:
    """Evaluate _np_lookup operation.

    Args:
        backend_module (object): The backend_module parameter.
        table (object): The table parameter.
        keys (object): The keys parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    arr: object = np.asarray(keys)
    default_value: object = kwargs.get("default_value", 0)
    if isinstance(table, dict):
        return np.vectorize(lambda k: table.get(k, default_value))(arr)
    if hasattr(table, "lookup"):
        return table.lookup(arr)
    return np.full_like(arr, default_value, dtype=np.int32)
