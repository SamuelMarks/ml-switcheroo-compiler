# ruff: noqa: E501
"""Vision eager common operations."""

import numpy as np

from ml_switcheroo_compiler.backends.eager.audio import mel_filterbank_eager, mfcc_eager
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("MelFilterbank")
def _np_mel_filterbank(backend_module: object, _: object, **kwargs: object) -> object:
    """Evaluate _np_mel_filterbank operation.

    Args:
        backend_module (object): The backend_module parameter.
        _ (object): The _ parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return mel_filterbank_eager(backend_module, None, kwargs.get("config", kwargs))


@numpy_eager_registry.register("Mfcc")
def _np_mfcc(backend_module: object, spectrogram: object, **kwargs: object) -> object:
    """Evaluate _np_mfcc operation.

    Args:
        backend_module (object): The backend_module parameter.
        spectrogram (object): The spectrogram parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return mfcc_eager(backend_module, spectrogram, kwargs.get("config", kwargs))


@numpy_eager_registry.register("PowerIteration")
def _np_power_iteration(backend_module: object, w: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_power_iteration operation.

    Args:
        backend_module (object): The backend_module parameter.
        w (object): The w parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    num_iters = kwargs.get("num_iters", 1)
    u = kwargs.get("u", None)
    if u is None:
        u = np.ones(w.shape[:-2] + (w.shape[-2], 1), dtype=w.dtype)
    else:
        u = np.expand_dims(u, axis=-1)
    for _ in range(num_iters):
        w_t = np.swapaxes(w, -1, -2)
        v = np.matmul(w_t, u)
        v = v / (np.linalg.norm(v, axis=-2, keepdims=True) + 1e-12)
        u = np.matmul(w, v)
        u = u / (np.linalg.norm(u, axis=-2, keepdims=True) + 1e-12)
    sigma = np.matmul(np.swapaxes(u, -1, -2), np.matmul(w, v))
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
