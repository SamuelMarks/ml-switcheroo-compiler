"""Implementation of math nn operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_nn`)."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, numpy_eager_registry
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol

ArrayInput = EagerValue


@numpy_eager_registry.register("Convolve")
def _np_convolve(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Return the discrete, linear convolution of two one-dimensional sequences.

    Args:
        backend_module (ModuleType): The backend module.
        *args (ArrayInput): Variable positional arguments.
        **kwargs (Union[float, int, str, bool, None]): Arbitrary keyword arguments.

    Returns:
        np.ndarray: The computed result.
    """
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], backend_module.convolve(*args, **kwargs))


@numpy_eager_registry.register("ConvGeneralDilatedLocal")
def _np_convgeneraldilatedlocal(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Implement ConvGeneralDilatedLocal.

    Args:
        backend_module (ModuleType): The backend module.
        *args (ArrayInput): Positional args.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        np.ndarray: Result array.
    """
    import scipy.signal

    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], scipy.signal.convolve(np.asarray(args[0]), np.asarray(args[1]), mode="valid"))


@numpy_eager_registry.register("ConvGeneralDilatedPatches")
def _np_convgeneraldilatedpatches(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Implement ConvGeneralDilatedPatches.

    Args:
        backend_module (ModuleType): The backend module.
        *args (ArrayInput): Positional args.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        np.ndarray: Result array.
    """
    import scipy.signal

    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], scipy.signal.convolve(np.asarray(args[0]), np.asarray(args[1]), mode="valid"))


@numpy_eager_registry.register("ConvWithGeneralPadding")
def _np_convwithgeneralpadding(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Implement ConvWithGeneralPadding.

    Args:
        backend_module (ModuleType): The backend module.
        *args (ArrayInput): Positional args.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        np.ndarray: Result array.
    """
    import scipy.signal

    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], scipy.signal.convolve(np.asarray(args[0]), np.asarray(args[1]), mode="valid"))


@numpy_eager_registry.register("RawConv2D")
def _np_rawconv2d(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Implement RawConv2D.

    Args:
        backend_module (ModuleType): The backend module.
        *args (ArrayInput): Positional args.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        np.ndarray: Result array.
    """
    import scipy.signal

    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], scipy.signal.convolve(np.asarray(args[0]), np.asarray(args[1]), mode="valid"))
