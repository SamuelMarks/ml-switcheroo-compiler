"""Vision eager common operations."""

from types import ModuleType
from typing import cast

import numpy as np

from ml_switcheroo_compiler.backends.eager.audio import mel_filterbank_eager, mfcc_eager
from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerValue, numpy_eager_registry


@numpy_eager_registry.register("MelFilterbank")
def _np_mel_filterbank(backend_module: ModuleType, _: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np mel filterbank operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        _ (EagerValue): The _ parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, cast("EagerCallableProtocol", mel_filterbank_eager)(backend_module, None, kwargs.get("config", kwargs)))


@numpy_eager_registry.register("Mfcc")
def _np_mfcc(backend_module: ModuleType, spectrogram: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np mfcc operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        spectrogram (EagerValue): The spectrogram parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, cast("EagerCallableProtocol", mfcc_eager)(backend_module, spectrogram, kwargs.get("config", kwargs)))


@numpy_eager_registry.register("PowerIteration")
def _np_power_iteration(backend_module: ModuleType, w: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np power iteration operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        w (EagerValue): The w parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    num_iters = kwargs.get("num_iters", 1)
    u = kwargs.get("u", None)
    if u is None:
        u = np.ones(getattr(w, "shape")[:-2] + (getattr(w, "shape")[-2], 1), dtype=getattr(w, "dtype"))
    elif getattr(u, "ndim", 1) < getattr(w, "ndim", 2):
        u = getattr(np, "expand_dims")(u, axis=-1)
    for _ in range(cast("int", num_iters)):
        w_t = getattr(np, "swapaxes")(w, -1, -2)
        v = np.matmul(w_t, u)
        v = v / (np.linalg.norm(v, axis=-2, keepdims=True) + 1e-12)
        u = getattr(np, "matmul")(w, v)
        u = u / (np.linalg.norm(u, axis=-2, keepdims=True) + 1e-12)
    sigma = getattr(np, "matmul")(np.swapaxes(u, -1, -2), getattr(np, "matmul")(w, v))
    return (np.squeeze(v, axis=-1), np.squeeze(u, axis=-1), np.squeeze(np.squeeze(sigma, axis=-1), axis=-1))


__all__ = [
    "__cached__",
    "__doc__",
    "__file__",
    "__loader__",
    "__name__",
    "__package__",
    "__spec__",
    "_np_mel_filterbank",
    "_np_mfcc",
    "_np_power_iteration",
    "np",
    "numpy_eager_registry",
]
