"""Array creation operations for numpy eager backend."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from types import ModuleType
from typing import Union

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.ops.linalg.linear_operator import (
    LinearOperatorIdentity,
    LinearOperatorScaledIdentity,
    LinearOperatorZeros,
)


@numpy_eager_registry.register("Fromfunction")
def _np_fromfunction_(
    backend_module: ModuleType,
    *args: Union[Callable[..., np.ndarray], tuple[int, ...], int, np.dtype],
    **kwargs: Union[np.dtype, tuple[int, ...], bool, None],
) -> np.ndarray:
    """Construct an array by executing a function over each coordinate.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[Callable[..., np.ndarray], tuple[int, ...], int, np.dtype]): Positional arguments.
        **kwargs (Union[np.dtype, tuple[int, ...], bool, None]): Keyword arguments.

    Returns:
        np.ndarray: The computed array from function coordinates.
    """
    return backend_module.fromfunction(*args, **kwargs)


@numpy_eager_registry.register("Fromiter")
def _np_fromiter_(
    backend_module: ModuleType,
    *args: Union[Iterable[Union[int, float]], np.dtype, int],
    **kwargs: Union[np.dtype, int, None],
) -> np.ndarray:
    """Create a new 1-dimensional array from an iterable object.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[Iterable[Union[int, float]], np.dtype, int]): Positional arguments.
        **kwargs (Union[np.dtype, int, None]): Keyword arguments.

    Returns:
        np.ndarray: Output 1-dimensional array.
    """
    return backend_module.fromiter(*args, **kwargs)


@numpy_eager_registry.register("Frompyfunc")
def _np_frompyfunc_(
    backend_module: ModuleType,
    *args: Union[Callable[..., Union[int, float]], int],
    **kwargs: Union[int, None],
) -> np.ufunc:
    """Takes an arbitrary Python function and returns a NumPy universal function.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[Callable[..., Union[int, float]], int]): Positional arguments.
        **kwargs (Union[int, None]): Keyword arguments.

    Returns:
        np.ufunc: Universal function wrapper.
    """
    return backend_module.frompyfunc(*args, **kwargs)


@numpy_eager_registry.register("Geomspace")
def _np_geomspace_(
    backend_module: ModuleType,
    *args: Union[float, int, tuple[int, ...], bool],
    **kwargs: Union[float, int, bool, np.dtype, None],
) -> np.ndarray:
    """Return numbers spaced evenly on a log scale (a geometric progression).

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[float, int, tuple[int, ...], bool]): Positional arguments (start, stop, num).
        **kwargs (Union[float, int, bool, np.dtype, None]): Keyword arguments.

    Returns:
        np.ndarray: Computed geometric progression array.
    """
    return backend_module.geomspace(*args, **kwargs)


@numpy_eager_registry.register("Zeros")
def _np_zeros_(
    backend_module: ModuleType,
    *args: Union[int, tuple[int, ...], list[int], np.dtype],
    **kwargs: Union[np.dtype, str, None],
) -> np.ndarray:
    """Return a new array of given shape and type, filled with zeros.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[int, tuple[int, ...], list[int], np.dtype]): Positional arguments.
        **kwargs (Union[np.dtype, str, None]): Keyword arguments.

    Returns:
        np.ndarray: Array of zeros with given shape.
    """
    return backend_module.zeros(*args, **kwargs)


@numpy_eager_registry.register("Ones")
def _np_ones_(
    backend_module: ModuleType,
    *args: Union[int, tuple[int, ...], list[int], np.dtype],
    **kwargs: Union[np.dtype, str, None],
) -> np.ndarray:
    """Return a new array of given shape and type, filled with ones.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[int, tuple[int, ...], list[int], np.dtype]): Positional arguments.
        **kwargs (Union[np.dtype, str, None]): Keyword arguments.

    Returns:
        np.ndarray: Array of ones with given shape.
    """
    return backend_module.ones(*args, **kwargs)


@numpy_eager_registry.register("Empty")
def _np_empty_(
    backend_module: ModuleType,
    *args: Union[int, tuple[int, ...], list[int], np.dtype],
    **kwargs: Union[np.dtype, str, None],
) -> np.ndarray:
    """Return a new array of given shape and type, without initializing entries.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[int, tuple[int, ...], list[int], np.dtype]): Positional arguments.
        **kwargs (Union[np.dtype, str, None]): Keyword arguments.

    Returns:
        np.ndarray: Uninitialized array with given shape.
    """
    return backend_module.empty(*args, **kwargs)


@numpy_eager_registry.register("Full")
def _np_full_(
    backend_module: ModuleType,
    *args: Union[int, float, tuple[int, ...], list[int], np.dtype],
    **kwargs: Union[np.dtype, str, None],
) -> np.ndarray:
    """Return a new array of given shape and type, filled with fill_value.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[int, float, tuple[int, ...], list[int], np.dtype]): Positional arguments.
        **kwargs (Union[np.dtype, str, None]): Keyword arguments.

    Returns:
        np.ndarray: Array filled with constant value.
    """
    return backend_module.full(*args, **kwargs)


@numpy_eager_registry.register("ZerosLike")
def _np_zeros_like_(
    backend_module: ModuleType,
    *args: Union[np.ndarray, list[int], list[float], np.dtype],
    **kwargs: Union[np.dtype, str, None],
) -> np.ndarray:
    """Return an array of zeros with the same shape and type as a given array.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, list[int], list[float], np.dtype]): Positional arguments.
        **kwargs (Union[np.dtype, str, None]): Keyword arguments.

    Returns:
        np.ndarray: Array of zeros like prototype.
    """
    return backend_module.zeros_like(*args, **kwargs)


@numpy_eager_registry.register("OnesLike")
def _np_ones_like_(
    backend_module: ModuleType,
    *args: Union[np.ndarray, list[int], list[float], np.dtype],
    **kwargs: Union[np.dtype, str, None],
) -> np.ndarray:
    """Return an array of ones with the same shape and type as a given array.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, list[int], list[float], np.dtype]): Positional arguments.
        **kwargs (Union[np.dtype, str, None]): Keyword arguments.

    Returns:
        np.ndarray: Array of ones like prototype.
    """
    return backend_module.ones_like(*args, **kwargs)


@numpy_eager_registry.register("EmptyLike")
def _np_empty_like_(
    backend_module: ModuleType,
    *args: Union[np.ndarray, list[int], list[float], np.dtype],
    **kwargs: Union[np.dtype, str, None],
) -> np.ndarray:
    """Return an empty array with the same shape and type as a given array.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, list[int], list[float], np.dtype]): Positional arguments.
        **kwargs (Union[np.dtype, str, None]): Keyword arguments.

    Returns:
        np.ndarray: Empty array like prototype.
    """
    return backend_module.empty_like(*args, **kwargs)


@numpy_eager_registry.register("FullLike")
def _np_full_like_(
    backend_module: ModuleType,
    *args: Union[np.ndarray, int, float, list[int], list[float], np.dtype],
    **kwargs: Union[np.dtype, str, None],
) -> np.ndarray:
    """Return a full array with the same shape and type as a given array.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, int, float, list[int], list[float], np.dtype]): Positional arguments.
        **kwargs (Union[np.dtype, str, None]): Keyword arguments.

    Returns:
        np.ndarray: Full array like prototype filled with value.
    """
    return backend_module.full_like(*args, **kwargs)


@numpy_eager_registry.register("Arange")
def _np_arange_(
    backend_module: ModuleType,
    *args: Union[int, float, np.dtype],
    **kwargs: Union[np.dtype, None],
) -> np.ndarray:
    """Return evenly spaced values within a given interval.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[int, float, np.dtype]): Positional arguments (start, stop, step).
        **kwargs (Union[np.dtype, None]): Keyword arguments.

    Returns:
        np.ndarray: Evenly spaced values array.
    """
    return backend_module.arange(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorIdentity")
def _np_linearoperatoridentity(
    backend_module: ModuleType,
    *args: Union[int, np.dtype, str],
    **kwargs: Union[int, np.dtype, str, None],
) -> LinearOperatorIdentity:
    """Implement LinearOperatorIdentity creation.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[int, np.dtype, str]): Positional arguments for LinearOperatorIdentity.
        **kwargs (Union[int, np.dtype, str, None]): Keyword arguments.

    Returns:
        LinearOperatorIdentity: Initialized linear operator identity instance.
    """
    return LinearOperatorIdentity(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorScaledIdentity")
def _np_linearoperatorscaledidentity(
    backend_module: ModuleType,
    *args: Union[int, float, np.ndarray, np.dtype, str],
    **kwargs: Union[int, float, np.ndarray, np.dtype, str, None],
) -> LinearOperatorScaledIdentity:
    """Implement LinearOperatorScaledIdentity creation.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[int, float, np.ndarray, np.dtype, str]): Positional arguments.
        **kwargs (Union[int, float, np.ndarray, np.dtype, str, None]): Keyword arguments.

    Returns:
        LinearOperatorScaledIdentity: Initialized linear operator scaled identity instance.
    """
    return LinearOperatorScaledIdentity(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorZeros")
def _np_linearoperatorzeros(
    backend_module: ModuleType,
    *args: Union[int, np.dtype, str],
    **kwargs: Union[int, np.dtype, str, None],
) -> LinearOperatorZeros:
    """Implement LinearOperatorZeros creation.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[int, np.dtype, str]): Positional arguments for LinearOperatorZeros.
        **kwargs (Union[int, np.dtype, str, None]): Keyword arguments.

    Returns:
        LinearOperatorZeros: Initialized linear operator zeros instance.
    """
    return LinearOperatorZeros(*args, **kwargs)


@numpy_eager_registry.register("FromDlpack")
def _np_fromdlpack(
    backend_module: ModuleType,
    *args: Union[np.ndarray, int, float],
    **kwargs: Union[int, float, str, None],
) -> Union[np.ndarray, int, float]:
    """Evaluate from_dlpack conversion or return object fallback.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, int, float]): Positional arguments (dlpack capsule/tensor).
        **kwargs (Union[int, float, str, None]): Keyword arguments.

    Returns:
        Union[np.ndarray, int, float]: Tensor converted from DLPack.
    """
    if hasattr(backend_module, "from_dlpack"):
        return backend_module.from_dlpack(*args, **kwargs)
    return args[0]


@numpy_eager_registry.register("Frombuffer")
def _np_frombuffer(
    backend_module: ModuleType,
    *args: Union[bytes, bytearray, np.dtype, int],
    **kwargs: Union[np.dtype, int, None],
) -> Union[np.ndarray, None]:
    """Interpret a buffer as a 1-dimensional array.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[bytes, bytearray, np.dtype, int]): Positional arguments (buffer, dtype, count, offset).
        **kwargs (Union[np.dtype, int, None]): Keyword arguments.

    Returns:
        Union[np.ndarray, None]: 1-D array from buffer or None if no arguments provided.
    """
    if not args:
        return None
    return np.frombuffer(args[0], **kwargs)
