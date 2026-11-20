"""Thin NumPy/SciPy ndimage, sliding-window, and Poisson kernel bindings."""

from __future__ import annotations

import types

import numpy as np

from ml_switcheroo_compiler.backends.eager.nn_ops.polyfills import _stirling_gammaln
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_int_list,
    numpy_eager_registry,
)
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.backends.protocols import (
    BackendModuleProtocol,
)


@numpy_eager_registry.register("ExtractVolumePatches")
def _np_extract_volume_patches(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Extract 3D volume patches using NumPy `lib.stride_tricks.sliding_window_view`.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active NumPy backend module.
        *args (BackendArg): Positional arguments containing 5D input tensor and optional kernel/stride sizes.
        **kwargs (BackendArg): Keyword arguments (`ksizes`, `strides`).

    Returns:
        np.ndarray: Extracted 6D reshaped volume patches tensor.
    """
    _ = backend_module
    if not args:
        return np.asarray([])
    input_tensor = np.asarray(args[0])
    ksizes = coerce_int_list(kwargs.get("ksizes", args[1] if len(args) > 1 else [1, 1, 1, 1, 1]))
    strides = coerce_int_list(kwargs.get("strides", args[2] if len(args) > 2 else [1, 1, 1, 1, 1]))

    if input_tensor.ndim != 5:
        return input_tensor

    try:
        view = np.lib.stride_tricks.sliding_window_view(input_tensor, tuple(ksizes))
    except ValueError:
        return input_tensor

    step_view = view[
        :: int(strides[0]),
        :: int(strides[1]),
        :: int(strides[2]),
        :: int(strides[3]),
        :: int(strides[4]),
    ]
    out_shape = step_view.shape[:5] + (-1,)
    return step_view.reshape(out_shape)


@numpy_eager_registry.register("ConvGeneralDilatedLocal")
def _np_convgeneraldilatedlocal(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Implement ConvGeneralDilatedLocal.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The backend module.
        *args (BackendArg): Positional args.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        np.ndarray: Result array.
    """
    del backend_module, kwargs
    import scipy.signal

    return ensure_ndarray(scipy.signal.convolve(np.asarray(args[0]), np.asarray(args[1]), mode="valid"))


@numpy_eager_registry.register("ConvGeneralDilatedPatches")
def _np_convgeneraldilatedpatches(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Implement ConvGeneralDilatedPatches.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The backend module.
        *args (BackendArg): Positional args.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        np.ndarray: Result array.
    """
    del backend_module, kwargs
    import scipy.signal

    return ensure_ndarray(scipy.signal.convolve(np.asarray(args[0]), np.asarray(args[1]), mode="valid"))


@numpy_eager_registry.register("ConvWithGeneralPadding")
def _np_convwithgeneralpadding(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Implement ConvWithGeneralPadding.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The backend module.
        *args (BackendArg): Positional args.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        np.ndarray: Result array.
    """
    del backend_module, kwargs
    import scipy.signal

    return ensure_ndarray(scipy.signal.convolve(np.asarray(args[0]), np.asarray(args[1]), mode="valid"))


@numpy_eager_registry.register("RawConv2D")
def _np_rawconv2d(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Implement RawConv2D.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The backend module.
        *args (BackendArg): Positional args.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        np.ndarray: Result array.
    """
    del backend_module, kwargs
    import scipy.signal

    return ensure_ndarray(scipy.signal.convolve(np.asarray(args[0]), np.asarray(args[1]), mode="valid"))


@numpy_eager_registry.register("LogPoissonLoss")
def _np_log_poisson_loss(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    targets: BackendArg,
    predictions: BackendArg,
    **kwargs: BackendArg,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Numerically stable vectorized LogPoissonLoss computation.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        targets (BackendArg): Ground truth targets tensor.
        predictions (BackendArg): Predictions (log-scale if log_input=True).
        **kwargs (BackendArg): Hyperparameters including compute_full_loss and log_input.

    Returns:
        np.ndarray: Evaluated loss values.
    """
    t = np.asarray(targets)
    inp = np.asarray(predictions)
    is_log = kwargs.get("log_input", True)
    compute_full = kwargs.get("compute_full_loss", False)

    if is_log:
        res = np.exp(inp) - t * inp
    else:
        res = inp - t * np.log(np.maximum(inp, 1e-12))

    if compute_full:
        try:
            import scipy.special

            res = res + scipy.special.gammaln(t + 1.0)
        except ImportError:
            res = res + _stirling_gammaln(t + 1.0)
    return ensure_ndarray(res)


@numpy_eager_registry.register("MaxPoolWithArgmax")
def _np_max_pool_with_argmax(
    backend_module: BackendModuleProtocol | types.ModuleType | None, input: BackendArg, **kwargs: BackendArg
) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.float64]], np.ndarray[tuple[int, ...], np.dtype[np.float64]]]:
    """Eager fallback for _np_max_pool_with_argmax.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        input (BackendArg): Input `input` argument (input).
        **kwargs (BackendArg): Keyword args.

    Returns:
        tuple[np.ndarray, np.ndarray]: Computed `tuple[np.ndarray, np.ndarray]` output.
    """
    arr = np.asarray(input)
    try:
        import scipy.ndimage

        size = kwargs.get("pool_size", (2, 2))
        if isinstance(size, int):
            size = (1, size, size, 1) if arr.ndim == 4 else (size, size)
        maxes = scipy.ndimage.maximum_filter(arr, size=size)
        argmaxes = np.argmax(arr.reshape(arr.shape[0], -1), axis=1)
        return maxes, argmaxes.astype(np.int32)
    except ImportError:
        argmaxes = np.argmax(arr.reshape(arr.shape[0], -1), axis=1)
        return arr, argmaxes.astype(np.int32)
