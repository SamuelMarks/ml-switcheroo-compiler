# ruff: noqa: E501
"""Vision eager common operations."""

import numpy as np

from ml_switcheroo_compiler.backends.eager.audio import mel_filterbank_eager, mfcc_eager
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("MelFilterbank")
def _np_mel_filterbank(backend_module: object, _: object, **kwargs: object) -> object:
    """Evaluate the mel filterbank logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        _ (object): Required parameter for _.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return mel_filterbank_eager(backend_module, None, kwargs.get("config", kwargs))


@numpy_eager_registry.register("Mfcc")
def _np_mfcc(backend_module: object, spectrogram: object, **kwargs: object) -> object:
    """Evaluate the mfcc logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        spectrogram (object): Required parameter for spectrogram.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return mfcc_eager(backend_module, spectrogram, kwargs.get("config", kwargs))


@numpy_eager_registry.register("PowerIteration")
def _np_power_iteration(backend_module: object, w: object, *args: object, **kwargs: object) -> object:
    """Evaluate the power iteration logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        w (object): Required parameter for w.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
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
