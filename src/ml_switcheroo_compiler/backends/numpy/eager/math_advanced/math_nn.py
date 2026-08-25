# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""math_nn module."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_nan import _xlogy


@numpy_eager_registry.register("Convolve")
def _np_convolve(backend_module: object, *args: object, **kwargs: object) -> object:
    """Return the discrete, linear convolution of two one-dimensional sequences.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: object: The computed result.
    """
    return backend_module.convolve(*args, **kwargs)


@numpy_eager_registry.register("ConvGeneralDilatedLocal")
def _np_convgeneraldilatedlocal(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement ConvGeneralDilatedLocal.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.signal

    return scipy.signal.convolve(np.asarray(args[0]), np.asarray(args[1]), mode="valid")


@numpy_eager_registry.register("ConvGeneralDilatedPatches")
def _np_convgeneraldilatedpatches(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement ConvGeneralDilatedPatches.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.signal

    return scipy.signal.convolve(np.asarray(args[0]), np.asarray(args[1]), mode="valid")


@numpy_eager_registry.register("ConvWithGeneralPadding")
def _np_convwithgeneralpadding(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement ConvWithGeneralPadding.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.signal

    return scipy.signal.convolve(np.asarray(args[0]), np.asarray(args[1]), mode="valid")


@numpy_eager_registry.register("RawConv2D")
def _np_rawconv2d(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement RawConv2D.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.signal

    return scipy.signal.convolve(np.asarray(args[0]), np.asarray(args[1]), mode="valid")
