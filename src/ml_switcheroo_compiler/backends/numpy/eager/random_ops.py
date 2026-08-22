# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Numpy eager fallback implementations for random operations."""

from typing import Any

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("Normal")
def _np_normal(backend_module: Any, shape: Any, **kwargs: Any) -> Any:
    """Evaluate _np_normal operation.

    Args:
        backend_module (object): The backend_module parameter.
        shape (object): The shape parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    dtype = kwargs.get("dtype", "float32")
    config = kwargs.get("config")
    mean = getattr(config, "mean", 0.0) if config else 0.0
    stddev = getattr(config, "stddev", 1.0) if config else 1.0
    return np.random.normal(loc=mean, scale=stddev, size=shape).astype(dtype)


@numpy_eager_registry.register("Uniform")
def _np_uniform(backend_module: Any, shape: Any, **kwargs: Any) -> Any:
    """Evaluate _np_uniform operation.

    Args:
        backend_module (object): The backend_module parameter.
        shape (object): The shape parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    dtype = kwargs.get("dtype", "float32")
    config = kwargs.get("config")
    minval = getattr(config, "minval", 0.0) if config else 0.0
    if minval is None:
        minval = 0.0
    maxval = getattr(config, "maxval", 1.0) if config else 1.0
    if maxval is None:
        maxval = 1.0
    return np.random.uniform(low=minval, high=maxval, size=shape).astype(dtype)


@numpy_eager_registry.register("StatelessSplit")
def _np_stateless_split(backend_module: Any, seed: Any, **kwargs: Any) -> Any:
    """Evaluate _np_stateless_split operation.

    Args:
        backend_module (object): The backend_module parameter.
        seed (object): The seed parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    num = kwargs.get("num", 2)
    s = np.asarray(seed).flatten()
    if len(s) > 0:
        if isinstance(s[0], (str, np.str_)):
            s_val = hash(s[0]) % (2**31 - 1)
        else:
            try:
                s_val = int(s[0])
            except ValueError:
                s_val = hash(str(s[0])) % (2**31 - 1)
    else:
        s_val = 0
    rng = np.random.RandomState(s_val)
    # Generate positive seeds to avoid overflow issues depending on platform
    return rng.randint(0, 2**31 - 1, size=(num, 2), dtype=np.int64)


@numpy_eager_registry.register("Lookup")
def _np_lookup(backend_module: Any, table: Any, keys: Any, **kwargs: Any) -> Any:
    """Evaluate _np_lookup operation.

    Args:
        backend_module (object): The backend_module parameter.
        table (object): The table parameter.
        keys (object): The keys parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    arr = np.asarray(keys)
    default_value = kwargs.get("default_value", 0)
    if isinstance(table, dict):
        return np.vectorize(lambda k: table.get(k, default_value))(arr)
    if hasattr(table, "lookup"):
        return table.lookup(arr)
    return np.full_like(arr, default_value, dtype=np.int32)
