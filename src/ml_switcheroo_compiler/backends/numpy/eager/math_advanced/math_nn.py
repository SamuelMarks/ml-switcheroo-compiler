"""math_nn module."""

from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType
from typing import Union

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry

ArrayInput = Union[np.ndarray, Sequence[float], Sequence[int], float, int]


@numpy_eager_registry.register("Convolve")
def _np_convolve(
    backend_module: ModuleType,
    *args: ArrayInput,
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Return the discrete, linear convolution of two one-dimensional sequences.

    Args:
        backend_module (ModuleType): The backend module.
        *args (ArrayInput): Variable positional arguments.
        **kwargs (Union[float, int, str, bool, None]): Arbitrary keyword arguments.

    Returns:
        np.ndarray: The computed result.
    """
    return backend_module.convolve(*args, **kwargs)


@numpy_eager_registry.register("ConvGeneralDilatedLocal")
def _np_convgeneraldilatedlocal(
    backend_module: ModuleType,
    *args: ArrayInput,
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Implement ConvGeneralDilatedLocal.

    Args:
        backend_module (ModuleType): The backend module.
        *args (ArrayInput): Positional args.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        np.ndarray: Result array.
    """
    import scipy.signal

    return scipy.signal.convolve(np.asarray(args[0]), np.asarray(args[1]), mode="valid")


@numpy_eager_registry.register("ConvGeneralDilatedPatches")
def _np_convgeneraldilatedpatches(
    backend_module: ModuleType,
    *args: ArrayInput,
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Implement ConvGeneralDilatedPatches.

    Args:
        backend_module (ModuleType): The backend module.
        *args (ArrayInput): Positional args.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        np.ndarray: Result array.
    """
    import scipy.signal

    return scipy.signal.convolve(np.asarray(args[0]), np.asarray(args[1]), mode="valid")


@numpy_eager_registry.register("ConvWithGeneralPadding")
def _np_convwithgeneralpadding(
    backend_module: ModuleType,
    *args: ArrayInput,
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Implement ConvWithGeneralPadding.

    Args:
        backend_module (ModuleType): The backend module.
        *args (ArrayInput): Positional args.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        np.ndarray: Result array.
    """
    import scipy.signal

    return scipy.signal.convolve(np.asarray(args[0]), np.asarray(args[1]), mode="valid")


@numpy_eager_registry.register("RawConv2D")
def _np_rawconv2d(
    backend_module: ModuleType,
    *args: ArrayInput,
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Implement RawConv2D.

    Args:
        backend_module (ModuleType): The backend module.
        *args (ArrayInput): Positional args.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        np.ndarray: Result array.
    """
    import scipy.signal

    return scipy.signal.convolve(np.asarray(args[0]), np.asarray(args[1]), mode="valid")
