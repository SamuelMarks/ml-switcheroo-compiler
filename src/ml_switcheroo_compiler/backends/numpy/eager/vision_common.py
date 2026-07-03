"""Vision eager common operations."""

import numpy as np

from ml_switcheroo_compiler.backends.eager.audio import istft_eager, mel_filterbank_eager, mfcc_eager, stft_eager
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("Istft")
def _np_istft(
    backend_module: object,
    stft_tensor: object,
    **kwargs: object,
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        stft_tensor: Arg.
        kwargs: Arg.
    """
    return istft_eager(backend_module, stft_tensor, **kwargs)


@numpy_eager_registry.register("MelFilterbank")
def _np_mel_filterbank(
    backend_module: object,
    _: object,
    **kwargs: object,
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        _: Arg.
        kwargs: Arg.
    """
    return mel_filterbank_eager(backend_module, None, kwargs.get("config", kwargs))


@numpy_eager_registry.register("Mfcc")
def _np_mfcc(
    backend_module: object,
    spectrogram: object,
    **kwargs: object,
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        spectrogram: Arg.
        kwargs: Arg.
    """
    return mfcc_eager(backend_module, spectrogram, kwargs.get("config", kwargs))


@numpy_eager_registry.register("PowerIteration")
def _np_power_iteration(backend_module: object, w: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        w: Arg.
        args: Arg.
        kwargs: Arg.
    """
    num_iters = kwargs.get("num_iters", 1)
    u = kwargs.get("u", None)
    if u is None:
        u = np.ones(w.shape[:-2] + (w.shape[-2], 1), dtype=w.dtype)
    for _ in range(num_iters):
        w_t = np.swapaxes(w, -1, -2)
        v = np.matmul(w_t, u)
        v = v / (np.linalg.norm(v, axis=-2, keepdims=True) + 1e-12)
        u = np.matmul(w, v)
        u = u / (np.linalg.norm(u, axis=-2, keepdims=True) + 1e-12)
    sigma = np.matmul(np.swapaxes(u, -1, -2), np.matmul(w, v))
    return np.squeeze(v, -1), np.squeeze(u, -1), np.squeeze(np.squeeze(sigma, -1), -1)


@numpy_eager_registry.register("Stft")
def _np_stft(
    np: object,
    input_tensor: object,
    **kwargs: object,
) -> object:
    """Function docstring.

    Args:
        np: Arg.
        input_tensor: Arg.
        kwargs: Arg.
    """
    return stft_eager(np, input_tensor, **kwargs)


__all__ = [
    "__cached__",
    "__doc__",
    "__file__",
    "__loader__",
    "__name__",
    "__package__",
    "__spec__",
    "_np_istft",
    "_np_mel_filterbank",
    "_np_mfcc",
    "_np_power_iteration",
    "_np_stft",
    "np",
    "numpy_eager_registry",
]
