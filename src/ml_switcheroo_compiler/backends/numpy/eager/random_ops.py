"""Numpy eager fallback implementations for random operations."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import as_ndarray
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol


class RandomDistributionConfig(typing.Protocol):
    """Protocol describing configuration attributes for random distributions."""

    mean: float
    stddev: float
    minval: float
    maxval: float


class LookupTableProtocol(typing.Protocol):
    """Protocol for table objects supporting lookup."""

    def lookup(self, keys: np.ndarray[tuple[int, ...], np.dtype[np.float64]]) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """Lookup keys in table.

        Args:
            keys (np.ndarray): Input keys to query.

        Returns:
            np.ndarray: Looked up values.
        """
        ...


@numpy_eager_registry.register("Normal")
def _np_normal(
    backend_module: BackendModuleProtocol,
    shape: tuple[int, ...] | list[int] | int,
    **kwargs: str | float | int | RandomDistributionConfig | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate normal distribution random sampling.

    Args:
        backend_module (ModuleType): Active backend module.
        shape (Union[tuple[int, ...], list[int], int]): Target output shape.
        **kwargs (Union[str, float, int, RandomDistributionConfig, None]): Keyword arguments.

    Returns:
        np.ndarray: Random samples drawn from normal distribution.
    """
    dtype = kwargs.get("dtype", "float32")
    config = kwargs.get("config")
    mean = getattr(config, "mean", 0.0) if config else 0.0
    stddev = getattr(config, "stddev", 1.0) if config else 1.0
    return as_ndarray(getattr(np.random.normal(loc=mean, scale=stddev, size=shape), "astype")(dtype))


@numpy_eager_registry.register("Uniform")
def _np_uniform(
    backend_module: BackendModuleProtocol,
    shape: tuple[int, ...] | list[int] | int,
    **kwargs: str | float | int | RandomDistributionConfig | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate uniform distribution random sampling.

    Args:
        backend_module (ModuleType): Active backend module.
        shape (Union[tuple[int, ...], list[int], int]): Target output shape.
        **kwargs (Union[str, float, int, RandomDistributionConfig, None]): Keyword arguments.

    Returns:
        np.ndarray: Random samples drawn from uniform distribution.
    """
    dtype = kwargs.get("dtype", "float32")
    config = kwargs.get("config")
    minval = getattr(config, "minval", 0.0) if config else 0.0
    if minval is None:
        minval = 0.0
    maxval = getattr(config, "maxval", 1.0) if config else 1.0
    if maxval is None:
        maxval = 1.0
    return as_ndarray(getattr(np.random.uniform(low=minval, high=maxval, size=shape), "astype")(dtype))


@numpy_eager_registry.register("StatelessSplit")
def _np_stateless_split(
    backend_module: BackendModuleProtocol,
    seed: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | list[int | str] | int | str,
    **kwargs: int | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate stateless PRNG key split.

    Args:
        backend_module (ModuleType): Active backend module.
        seed (Union[np.ndarray, list[Union[int, str]], int, str]): Seed value or array.
        **kwargs (Union[int, None]): Keyword arguments such as num.

    Returns:
        np.ndarray: Split pseudo-random integer seeds.
    """
    num_val = kwargs.get("num", 2)
    num = int(num_val) if num_val is not None else 2
    s = np.asarray(seed).flatten()
    if len(s) > 0:
        if isinstance(s[0], (str, np.str_)):
            s_val = hash(s[0]) % (2**31 - 1)
        else:
            try:
                s_val = int(s[0])
            except (ValueError, TypeError):
                s_val = hash(str(s[0])) % (2**31 - 1)
    else:
        s_val = 0
    rng = np.random.RandomState(s_val)
    return as_ndarray(rng.randint(0, 2**31 - 1, size=(num, 2), dtype=np.int64))


@numpy_eager_registry.register("Lookup")
def _np_lookup(
    backend_module: BackendModuleProtocol,
    table: dict[int | str, int | float] | LookupTableProtocol | np.ndarray[tuple[int, ...], np.dtype[np.float64]] | int,
    keys: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | list[int | str] | int | str,
    **kwargs: int | float | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate lookup table query on keys.

    Args:
        backend_module (ModuleType): Active backend module.
        table (Union[dict[Union[int, str], Union[int, float]], LookupTableProtocol, np.ndarray, int]):
            Table dictionary or lookup object.
        keys (Union[np.ndarray, list[Union[int, str]], int, str]): Keys to lookup.
        **kwargs (Union[int, float, None]): Keyword arguments such as default_value.

    Returns:
        np.ndarray: Array of looked up values.
    """
    arr = np.asarray(keys)
    default_value = kwargs.get("default_value", 0)
    if isinstance(table, dict):
        return as_ndarray(np.vectorize(lambda k: table.get(k, default_value))(arr))
    if hasattr(table, "lookup"):
        return table.lookup(arr)
    return np.full_like(arr, default_value, dtype=np.int32)
