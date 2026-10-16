"""Vision eager common operations."""

from types import ModuleType
from typing import Callable, cast

import numpy as np

from ml_switcheroo_compiler.backends.eager.audio import mel_filterbank_eager, mfcc_eager
from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue, numpy_eager_registry


@numpy_eager_registry.register("MelFilterbank")
def _np_mel_filterbank(backend_module: ModuleType, _: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_mel_filterbank operation.

    Args:
        backend_module (object): The backend_module parameter.
        _ (object): The _ parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, cast("Callable[..., EagerTensorProtocol]", mel_filterbank_eager)(backend_module, None, kwargs.get("config", kwargs)))


@numpy_eager_registry.register("Mfcc")
def _np_mfcc(backend_module: ModuleType, spectrogram: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_mfcc operation.

    Args:
        backend_module (object): The backend_module parameter.
        spectrogram (object): The spectrogram parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, cast("Callable[..., EagerTensorProtocol]", mfcc_eager)(backend_module, spectrogram, kwargs.get("config", kwargs)))


@numpy_eager_registry.register("PowerIteration")
def _np_power_iteration(backend_module: ModuleType, w: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_power_iteration operation.

    Args:
        backend_module (object): The backend_module parameter.
        w (object): The w parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    num_iters = kwargs.get("num_iters", 1)
    u = kwargs.get("u", None)
    if u is None:
        u = np.ones(getattr(w, "shape")[:-2] + (getattr(w, "shape")[-2], 1), dtype=getattr(w, "dtype"))
    else:
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
